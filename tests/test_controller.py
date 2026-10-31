import unittest
from unittest.mock import MagicMock, call, patch

from device.controller import DeviceController, TOUCH_EVENT_FORMAT


class TapJitterTests(unittest.TestCase):
    def test_zero_jitter_leaves_coordinates_exact(self):
        controller = DeviceController(MagicMock(), 100, 200)
        with patch.object(controller, "send_touch") as send, patch("device.controller.time.sleep"):
            controller.tap(10, 20)
        self.assertEqual(send.call_args_list, [call(0, 10, 20, 1.0), call(1, 10, 20, 0.0)])

    def test_jitter_stays_within_radius_and_clamps_to_display(self):
        controller = DeviceController(MagicMock(), 10, 10, tap_jitter_pixels=3)
        with patch("device.controller.random.randrange", side_effect=[-3, -3]) as rng:
            # 1-3 = -2, clamped to 0 on both axes.
            self.assertEqual(controller._apply_tap_jitter(1, 1), (0, 0))
        with patch("device.controller.random.randrange", side_effect=[3, 3]):
            # 8+3 = 11 clamps to display_width-1 = 9; 9+3 = 12 clamps to 9.
            self.assertEqual(controller._apply_tap_jitter(8, 9), (9, 9))
        with patch("device.controller.random.randrange", side_effect=[1, 2]):
            # Inside the display: no clamping.
            self.assertEqual(controller._apply_tap_jitter(5, 5), (6, 7))
        self.assertTrue(rng.call_args_list)

    def test_tap_applies_jitter_to_down_and_up_same_point(self):
        controller = DeviceController(MagicMock(), 100, 100, tap_jitter_pixels=5)
        with patch("device.controller.random.randrange", side_effect=[2, -1]) as rng, \
             patch.object(controller, "send_touch") as send, patch("device.controller.time.sleep"):
            controller.tap(50, 50)
        # x -> 50+2=52, y -> 50-1=49; both down and up use the jittered point.
        self.assertEqual(send.call_args_list, [call(0, 52, 49, 1.0), call(1, 52, 49, 0.0)])
        self.assertEqual([c.args for c in rng.call_args_list], [(-5, 6), (-5, 6)])

    def test_jitter_disabled_when_display_size_unset(self):
        # No display size means no clamping; jitter still applies to the raw point.
        controller = DeviceController(MagicMock(), 0, 0, tap_jitter_pixels=4)
        with patch("device.controller.random.randrange", side_effect=[4, -4]):
            self.assertEqual(controller._apply_tap_jitter(10, 10), (14, 6))

    def test_rejects_invalid_jitter_value(self):
        for value in (-1, 2.0, True, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DeviceController(MagicMock(), 100, 100, tap_jitter_pixels=value)


class DeviceControllerTests(unittest.TestCase):
    def test_display_power_sends_scrcpy_334_packets_on_current_connection(self):
        first, second = MagicMock(), MagicMock()
        controller = DeviceController(first)
        controller.set_display_power(False)
        controller.set_display_power(True)
        self.assertEqual(first.sendall.call_args_list, [call(b"\x0a\x00"), call(b"\x0a\x01")])

        controller.set_connection(second)
        controller.set_display_power(False)
        second.sendall.assert_called_once_with(b"\x0a\x00")
        self.assertEqual(first.sendall.call_count, 2)

    def test_display_power_requires_connection_and_propagates_send_failure(self):
        controller = DeviceController()
        for on in (False, True):
            with self.subTest(on=on), self.assertRaises(RuntimeError):
                controller.set_display_power(on)

        conn = MagicMock()
        failure = OSError("disconnected")
        conn.sendall.side_effect = failure
        controller.set_connection(conn)
        with self.assertRaises(OSError) as raised:
            controller.set_display_power(False)
        self.assertIs(raised.exception, failure)
        conn.sendall.assert_called_once_with(b"\x0a\x00")

        controller.set_connection(None)
        with self.assertRaises(RuntimeError):
            controller.set_display_power(True)

    def test_swipe_waits_for_full_duration_using_monotonic_clock(self):
        controller = DeviceController(MagicMock(), 100, 100)
        clock = [10.0]
        events = []

        def sleep(seconds):
            # Simulate early wakeups as well as time spent sending events.
            clock[0] += seconds / 2 if seconds > 0.001 else seconds

        def send(action, x, y, pressure):
            events.append((action, x, y, pressure, clock[0]))
            clock[0] += 0.005

        with patch.object(controller, "send_touch", side_effect=send), \
             patch("device.controller.time.monotonic", side_effect=lambda: clock[0]), \
             patch("device.controller.time.sleep", side_effect=sleep):
            controller.swipe(10, 20, 90, 80, duration_ms=100, steps=4)

        self.assertEqual([event[:4] for event in events], [
            (0, 10, 20, 1.0), (2, 30, 35, 1.0), (2, 50, 50, 1.0),
            (2, 70, 65, 1.0), (1, 90, 80, 0.0),
        ])
        self.assertGreaterEqual(events[-1][4] - events[0][4], 0.1)

    def test_swipe_rejects_invalid_endpoints_and_steps_before_down(self):
        controller = DeviceController(MagicMock(), 100, 100)
        with patch.object(controller, "send_touch") as send:
            for args in [(-1, 20, 90, 80), (10, 100, 90, 80),
                         (10, 20, 100, 80), (10, 20, 90, -1)]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    controller.swipe(*args)
            for steps in (0, 1, -2, 2.0, True, "2"):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    controller.swipe(10, 20, 90, 80, steps=steps)
            with self.assertRaises(TypeError):
                controller.swipe(10, 20, 90.5, 80)
            send.assert_not_called()

    def test_swipe_interrupt_releases_last_position_and_stops_following_action(self):
        controller = DeviceController(MagicMock(), 100, 100)
        clock = [0.0]

        def sleep(seconds):
            if clock[0] > 0:
                raise KeyboardInterrupt()
            clock[0] += seconds

        with patch.object(controller, "send_touch") as send, \
             patch("device.controller.time.monotonic", side_effect=lambda: clock[0]), \
             patch("device.controller.time.sleep", side_effect=sleep):
            with self.assertRaises(KeyboardInterrupt):
                controller.swipe(10, 20, 90, 80, duration_ms=100, steps=4)
                controller.tap(40, 40)

        self.assertEqual(send.call_args_list, [
            call(0, 10, 20, 1.0), call(2, 30, 35, 1.0), call(1, 30, 35, 0.0),
        ])

    def test_swipe_move_error_preserved_if_cleanup_fails(self):
        controller = DeviceController(MagicMock(), 100, 100)
        error = OSError("move failed")
        with patch.object(controller, "send_touch", side_effect=[None, error, OSError("up failed")]) as send, \
             patch("device.controller.time.monotonic", side_effect=[0, 1]):
            with self.assertRaises(OSError) as raised:
                controller.swipe(10, 20, 90, 80, steps=2)
        self.assertIs(raised.exception, error)
        self.assertEqual(send.call_args_list[-1], call(1, 10, 20, 0.0))

    def test_swipe_failed_down_does_not_send_up(self):
        controller = DeviceController(MagicMock(), 100, 100)
        with patch.object(controller, "send_touch", side_effect=OSError("down failed")) as send:
            with self.assertRaisesRegex(OSError, "down failed"):
                controller.swipe(10, 20, 90, 80)
        send.assert_called_once_with(0, 10, 20, 1.0)

    def test_swipe_retries_failed_up_at_last_position(self):
        controller = DeviceController(MagicMock(), 100, 100)
        error = OSError("up failed")
        with patch.object(controller, "send_touch", side_effect=[None, None, error, None]) as send, \
             patch("device.controller.time.monotonic", side_effect=[0, 1, 1]):
            with self.assertRaises(OSError) as raised:
                controller.swipe(10, 20, 90, 80, steps=2)
        self.assertIs(raised.exception, error)
        self.assertEqual(send.call_args_list, [
            call(0, 10, 20, 1.0), call(2, 50, 50, 1.0),
            call(1, 90, 80, 0.0), call(1, 50, 50, 0.0),
        ])

    def test_tap_sends_touch_packets(self):
        conn = MagicMock()
        controller = DeviceController(conn, 100, 100)
        with patch("device.controller.time.sleep") as sleep:
            controller.tap(10, 20, hold_ms=30)
        sleep.assert_called_once_with(0.03)
        packets = [TOUCH_EVENT_FORMAT.unpack(args.args[0]) for args in conn.sendall.call_args_list]
        self.assertEqual([(p[1], p[3], p[4], p[7]) for p in packets], [
            (0, 10, 20, 0xFFFF), (1, 10, 20, 0),
        ])

    def test_tap_interrupt_releases_and_preserves_original_exception(self):
        controller = DeviceController(MagicMock(), 100, 100)
        error = KeyboardInterrupt()
        with patch.object(controller, "send_touch", side_effect=[None, OSError("up failed")]) as send, \
             patch("device.controller.time.sleep", side_effect=error):
            with self.assertRaises(KeyboardInterrupt) as raised:
                controller.tap(10, 20)
        self.assertIs(raised.exception, error)
        self.assertEqual(send.call_args_list, [call(0, 10, 20, 1.0), call(1, 10, 20, 0.0)])

    def test_tap_retries_failed_up_and_does_not_cleanup_failed_down(self):
        controller = DeviceController(MagicMock(), 100, 100)
        error = OSError("up failed")
        with patch.object(controller, "send_touch", side_effect=[None, error, None]) as send, \
             patch("device.controller.time.sleep"):
            with self.assertRaises(OSError) as raised:
                controller.tap(10, 20)
        self.assertIs(raised.exception, error)
        self.assertEqual(send.call_args_list, [
            call(0, 10, 20, 1.0), call(1, 10, 20, 0.0), call(1, 10, 20, 0.0),
        ])
        with patch.object(controller, "send_touch", side_effect=OSError("down failed")) as send:
            with self.assertRaisesRegex(OSError, "down failed"):
                controller.tap(10, 20)
        send.assert_called_once_with(0, 10, 20, 1.0)

    def test_back_sends_down_and_up_on_existing_connection(self):
        conn = MagicMock()
        controller = DeviceController(conn)
        controller.back()
        conn.sendall.assert_called_once_with(bytes.fromhex(
            "00 00 00 00 00 04 00 00 00 00 00 00 00 00 "
            "00 01 00 00 00 04 00 00 00 00 00 00 00 00"
        ))

    def test_back_requires_connection_and_propagates_socket_error(self):
        controller = DeviceController()
        with self.assertRaisesRegex(RuntimeError, "Control connection"):
            controller.back()
        conn = MagicMock()
        conn.sendall.side_effect = OSError("disconnected")
        controller.set_connection(conn)
        with self.assertRaisesRegex(OSError, "disconnected"):
            controller.back()


if __name__ == "__main__":
    unittest.main()
