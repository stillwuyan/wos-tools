import argparse
import math
import socket
import subprocess
import threading
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

import app
from device.scrcpy_session import ScrcpySession


class ScrcpySessionTests(unittest.TestCase):
    def setUp(self):
        self.session = ScrcpySession(Path("unused"), serial="device-1", socket_name="test-scrcpy")

    def start_decoder(self):
        video, peer = socket.socketpair()
        self.session._video_conn = video
        self.session._stream_ended.clear()
        self.addCleanup(peer.close)
        self.addCleanup(self.session.stop)
        with patch("device.scrcpy_session.av.codec.CodecContext.create") as create:
            self.session._start_decode_thread()
        return peer, create.return_value

    def test_initial_and_normal_stop_state(self):
        self.assertTrue(self.session.screen_on)
        self.assertFalse(self.session.stay_awake)
        self.assertTrue(self.session.stream_ended)
        self.assertIsNone(self.session.stream_error)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.session.get_latest_frame()
        peer, _ = self.start_decoder()
        self.assertFalse(self.session.wait_for_stream_end(0.01))
        self.assertIsNone(self.session.get_latest_frame())
        self.session.stop()
        self.assertTrue(self.session.wait_for_stream_end(0))
        self.assertIsNone(self.session.stream_error)
        peer.close()

    def test_eof_and_truncated_packets_report_failure(self):
        for data in (b"", b"partial", bytes(8) + (4).to_bytes(4, "big") + b"ab"):
            with self.subTest(data=data):
                session = ScrcpySession(Path("unused"))
                video, peer = socket.socketpair()
                session._video_conn = video
                session._stream_ended.clear()
                session._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
                try:
                    with patch("device.scrcpy_session.av.codec.CodecContext.create"), \
                            self.assertLogs("scrcpy", level="ERROR"):
                        session._start_decode_thread()
                        peer.sendall(data)
                        peer.shutdown(socket.SHUT_WR)
                        self.assertTrue(session.wait_for_stream_end(2))
                    self.assertIsInstance(session.stream_error, ConnectionError)
                    self.assertIsNone(session._latest_frame)
                    for getter in (session.get_latest_frame, session.get_latest_size, session.get_frame_snapshot, session.check_stream_health):
                        with self.assertRaisesRegex(RuntimeError, "video stream failed") as raised:
                            getter()
                        self.assertIs(raised.exception.__cause__, session.stream_error)
                finally:
                    peer.close()
                    session.stop()

    def test_decode_error_is_exposed(self):
        peer, codec = self.start_decoder()
        error = ValueError("bad frame")
        codec.decode.side_effect = error
        with self.assertLogs("scrcpy", level="ERROR"):
            peer.sendall(bytes(8) + (1).to_bytes(4, "big") + b"x")
            self.assertTrue(self.session.wait_for_stream_end(2))
        self.assertIs(self.session.stream_error, error)
        with self.assertRaises(RuntimeError) as raised:
            self.session.check_stream_health()
        self.assertIs(raised.exception.__cause__, error)

    def test_frame_is_copied_and_stop_invalidates_it(self):
        peer, codec = self.start_decoder()
        self.session.display_width, self.session.display_height = 30, 20
        self.assertIsNone(self.session.get_frame_snapshot())
        frame_ready = threading.Event()
        rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        frame = MagicMock()
        frame.to_ndarray.return_value = rgba

        codec.decode.return_value = [frame]
        original_read = self.session._read_exact

        def read_exact(sock, size):
            if frame.to_ndarray.called:
                frame_ready.set()
            return original_read(sock, size)

        with patch.object(self.session, "_read_exact", side_effect=read_exact):
            peer.sendall(bytes(8) + (1).to_bytes(4, "big") + b"x")
            self.assertTrue(frame_ready.wait(2))
        result = self.session.get_latest_frame()
        self.assertIsNotNone(result)
        self.assertEqual(self.session.get_latest_size(), (3, 2))
        snapshot = self.session.get_frame_snapshot()
        self.assertEqual(snapshot.frame_id, 1)
        self.assertIsInstance(snapshot.decode_time, float)
        self.assertEqual(snapshot.frame_size, (3, 2))
        self.assertEqual(snapshot.device_size, (30, 20))
        snapshot.frame[:] = 128
        self.assertFalse(self.session.get_frame_snapshot().frame.any())
        result[:] = 255
        self.assertFalse(self.session.get_latest_frame().any())
        self.session.stop()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.session.get_latest_size()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.session.get_frame_snapshot()

    def test_snapshot_sequence_and_metadata_follow_decoded_frames(self):
        peer, codec = self.start_decoder()
        frames = []
        for height, width in ((2, 3), (4, 5)):
            frame = MagicMock()
            frame.to_ndarray.return_value = np.zeros((height, width, 4), dtype=np.uint8)
            frames.append(frame)
        codec.decode.side_effect = [[frames[0]], [frames[1]]]
        ready = [threading.Event(), threading.Event()]
        original_read = self.session._read_exact

        def read_exact(sock, size):
            if frames[1].to_ndarray.called:
                ready[1].set()
            elif frames[0].to_ndarray.called:
                ready[0].set()
            return original_read(sock, size)

        packet = bytes(8) + (1).to_bytes(4, "big") + b"x"
        with patch.object(self.session, "_read_exact", side_effect=read_exact):
            self.session.display_width, self.session.display_height = 30, 20
            peer.sendall(packet)
            self.assertTrue(ready[0].wait(2))
            first = self.session.get_frame_snapshot()
            self.session.display_width, self.session.display_height = 50, 40
            peer.sendall(packet)
            self.assertTrue(ready[1].wait(2))
            second = self.session.get_frame_snapshot()
        self.assertEqual((first.frame_id, second.frame_id), (1, 2))
        self.assertLessEqual(first.decode_time, second.decode_time)
        self.assertEqual((first.frame_size, first.device_size), ((3, 2), (30, 20)))
        self.assertEqual((second.frame_size, second.device_size), ((5, 4), (50, 40)))
        self.assertEqual(first.frame.shape, (2, 3, 3))
        self.assertEqual(second.frame.shape, (4, 5, 3))

    def test_frame_sequence_is_not_reset_on_restart(self):
        self.session._frame_id = 12
        with patch.object(self.session, "_open_listener"), \
                patch("device.scrcpy_session.subprocess.run"), \
                patch.object(self.session, "_run_adb"), \
                patch.object(self.session, "install_server"), \
                patch.object(self.session, "start_server_process"), \
                patch.object(self.session, "_accept", side_effect=[MagicMock(), MagicMock()]), \
                patch.object(self.session, "_read_stream_metadata"), \
                patch.object(self.session, "_start_decode_thread"):
            self.session.start()
            self.assertEqual(self.session._frame_id, 12)
            self.assertIsNone(self.session.get_frame_snapshot())
            self.session.stop()

    def test_accept_closes_socket_if_configuration_fails(self):
        listener, conn = MagicMock(), MagicMock()
        self.session._listener = listener
        listener.accept.return_value = (conn, None)
        conn.setsockopt.side_effect = OSError("socket configuration failed")
        with self.assertRaisesRegex(OSError, "socket configuration failed"):
            self.session._accept()
        conn.close.assert_called_once()

    def test_start_stop_removes_only_own_mapping_with_selected_device(self):
        listener = MagicMock()
        video, control = MagicMock(), MagicMock()
        with patch.object(self.session, "_open_listener", side_effect=lambda: setattr(self.session, "_listener", listener)), \
                patch.object(self.session, "_run_adb") as adb, \
                patch("device.scrcpy_session.subprocess.run") as run, \
                patch.object(self.session, "install_server"), \
                patch.object(self.session, "start_server_process"), \
                patch.object(self.session, "_accept", side_effect=[video, control]), \
                patch.object(self.session, "_read_stream_metadata"), \
                patch.object(self.session, "_start_decode_thread"):
            self.session.start()
            self.assertFalse(self.session.stream_ended)
            self.session.stop()
            self.session.stop()
        self.assertEqual(adb.call_args_list, [
            call("reverse", "localabstract:test-scrcpy", "tcp:27183"),
            call("reverse", "--remove", "localabstract:test-scrcpy"),
        ])
        self.assertEqual(run.call_args.args[0], ["adb", "-s", "device-1", "reverse", "--remove", "localabstract:test-scrcpy"])
        control.sendall.assert_not_called()
        listener.close.assert_called_once()
        video.close.assert_called_once()
        control.close.assert_called_once()

    def test_failed_reverse_creation_is_cleaned_and_original_error_preserved(self):
        listener = MagicMock()
        failure = RuntimeError("reverse creation failed")
        with patch.object(self.session, "_open_listener", side_effect=lambda: setattr(self.session, "_listener", listener)), \
                patch("device.scrcpy_session.subprocess.run"), \
                patch.object(self.session, "_run_adb", side_effect=[failure, None]) as adb:
            with self.assertRaises(RuntimeError) as raised:
                self.session.start()
        self.assertIs(raised.exception, failure)
        self.assertEqual(adb.call_args_list[-1], call("reverse", "--remove", "localabstract:test-scrcpy"))
        self.assertFalse(self.session._reverse_active)
        self.assertTrue(self.session.stream_ended)
        listener.close.assert_called_once()

    def test_restart_resets_failure_and_frame(self):
        self.session._stream_error = ConnectionError("old failure")
        self.session._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.session._latest_size = (2, 2)
        with patch.object(self.session, "_open_listener"), \
                patch("device.scrcpy_session.subprocess.run"), \
                patch.object(self.session, "_run_adb"), \
                patch.object(self.session, "install_server"), \
                patch.object(self.session, "start_server_process"), \
                patch.object(self.session, "_accept", side_effect=[MagicMock(), MagicMock()]), \
                patch.object(self.session, "_read_stream_metadata"), \
                patch.object(self.session, "_start_decode_thread"):
            self.session.start()
            self.assertFalse(self.session.stream_ended)
            self.assertIsNone(self.session.stream_error)
            self.assertIsNone(self.session.get_latest_frame())
            self.assertIsNone(self.session.get_latest_size())
            self.session.stop()

    def test_failure_after_reverse_and_failed_cleanup_can_be_retried(self):
        listener = MagicMock()
        with patch.object(self.session, "_open_listener", side_effect=lambda: setattr(self.session, "_listener", listener)), \
                patch("device.scrcpy_session.subprocess.run"), \
                patch.object(self.session, "_run_adb", side_effect=[None, RuntimeError("remove failed"), None]) as adb, \
                patch.object(self.session, "install_server", side_effect=ValueError("install failed")), \
                self.assertLogs("scrcpy", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "install failed"):
                self.session.start()
            self.assertTrue(self.session._reverse_active)
            self.session.stop()
        self.assertEqual(adb.call_count, 3)
        self.assertFalse(self.session._reverse_active)

    def test_server_command_reports_switchable_screen_options(self):
        for screen_on, stay_awake in ((True, True), (False, True), (True, False), (False, False)):
            with self.subTest(screen_on=screen_on, stay_awake=stay_awake):
                session = ScrcpySession(Path("unused"), screen_on=screen_on, stay_awake=stay_awake)
                with patch("device.scrcpy_session.subprocess.Popen") as popen:
                    session.start_server_process()
                cmd = popen.call_args.args[0]
                self.assertFalse(any(arg.startswith("turn_screen_off=") for arg in cmd))
                self.assertFalse(any(arg.startswith("screen_on=") for arg in cmd))
                self.assertIn(f"stay_awake={str(stay_awake).lower()}", cmd)
                self.assertIn("power_on=false", cmd)
                self.assertIn("cleanup=false", cmd)
                self.assertIn("max_fps=15", cmd)
                self.assertFalse(any(arg.startswith("max-fps=") for arg in cmd))

    def test_caller_reports_stream_failure_and_stops_session(self):
        args = argparse.Namespace(config="config.yaml", server_jar="unused", adb="adb", serial=None,
                                  host="127.0.0.1", port=27183, socket_name="scrcpy",
                                  output_dir="captures", window_scale=0.33)
        with patch.object(app, "load_server_jar", return_value=Path("unused")), \
                patch.object(app, "ScrcpySession") as session_type, \
                patch.object(app, "MarkerUI") as ui, \
                self.assertLogs("app", level="ERROR"):
            ui.return_value.run.side_effect = RuntimeError("video stream failed")
            self.assertEqual(app.run_mark_mode(args), 1)
        session_type.return_value.start.assert_called_once()
        session_type.return_value.stop.assert_called_once()


class ScrcpyDisplayPowerTests(unittest.TestCase):
    @contextmanager
    def session_with_resources(self, screen_on=False):
        session = ScrcpySession(Path("unused"), serial="device-1", socket_name="test-scrcpy",
                                screen_on=screen_on)
        events = MagicMock()
        video, control, listener, process = (MagicMock() for _ in range(4))
        process.poll.return_value = None
        process.wait.return_value = 0
        for name, resource in (("video", video), ("control", control),
                               ("listener", listener), ("process", process)):
            events.attach_mock(resource, name)
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                session, "_open_listener", side_effect=lambda: setattr(session, "_listener", listener)))
            stack.enter_context(patch.object(session, "_run_adb"))
            stack.enter_context(patch("device.scrcpy_session.subprocess.run"))
            stack.enter_context(patch.object(session, "install_server"))
            stack.enter_context(patch.object(
                session, "start_server_process", side_effect=lambda: setattr(session, "_server_proc", process)))
            stack.enter_context(patch.object(session, "_accept", side_effect=[video, control]))
            metadata = stack.enter_context(patch.object(session, "_read_stream_metadata"))
            decoder = stack.enter_context(patch.object(session, "_start_decode_thread"))
            events.attach_mock(metadata, "metadata")
            events.attach_mock(decoder, "decoder")
            yield session, events, video, control, listener, process

    def assert_bounded_wait(self, process):
        self.assertTrue(process.wait.called)
        args, kwargs = process.wait.call_args_list[0]
        timeout = kwargs.get("timeout", args[0] if args else None)
        self.assertIsInstance(timeout, (int, float))
        self.assertTrue(math.isfinite(timeout))
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 30)

    def assert_resources_cleaned(self, session, video, control, listener, process):
        video.close.assert_called_once()
        control.close.assert_called_once()
        listener.close.assert_called_once()
        self.assertIsNone(session._video_conn)
        self.assertIsNone(session._control_conn)
        self.assertIsNone(session._listener)
        self.assertIsNone(session._server_proc)
        self.assertTrue(session.stream_ended)
        session._run_adb.assert_any_call("reverse", "--remove", "localabstract:test-scrcpy")

    def test_start_and_stop_power_messages_and_graceful_shutdown_order(self):
        with self.session_with_resources() as (session, events, video, control, listener, process):
            session.start()
            self.assertEqual(control.sendall.call_args_list, [call(b"\x0a\x00")])
            self.assertLess(events.mock_calls.index(call.metadata()), events.mock_calls.index(call.decoder()))
            self.assertLess(events.mock_calls.index(call.decoder()),
                            events.mock_calls.index(call.control.sendall(b"\x0a\x00")))

            session.stop()
            self.assertEqual(control.sendall.call_args_list, [call(b"\x0a\x00"), call(b"\x0a\x01")])
            calls = events.mock_calls
            on = calls.index(call.control.sendall(b"\x0a\x01"))
            shutdown = calls.index(call.control.shutdown(socket.SHUT_WR))
            wait = next(i for i, item in enumerate(calls) if item[0] == "process.wait")
            self.assertLess(on, shutdown)
            self.assertLess(shutdown, wait)
            for name in ("video", "control", "listener"):
                self.assertLess(wait, calls.index(getattr(call, name).close()))
            self.assert_bounded_wait(process)
            process.kill.assert_not_called()
            self.assert_resources_cleaned(session, video, control, listener, process)
            session.stop()
            self.assertEqual(control.sendall.call_count, 2)

    def test_disabled_option_sends_no_power_messages(self):
        with self.session_with_resources(screen_on=True) as (session, _, video, control, listener, process):
            session.start()
            session.stop()
            session.stop()
            control.sendall.assert_not_called()
            self.assert_resources_cleaned(session, video, control, listener, process)

    def test_failed_power_off_attempts_restoration_and_preserves_start_error(self):
        for restore_fails in (False, True):
            with self.subTest(restore_fails=restore_fails), self.session_with_resources() as resources:
                session, events, video, control, listener, process = resources
                failure = OSError("power off failed")
                control.sendall.side_effect = [failure, OSError("power on failed") if restore_fails else None]
                with self.assertRaises(OSError) as raised:
                    session.start()
                self.assertIs(raised.exception, failure)
                self.assertEqual(control.sendall.call_args_list, [call(b"\x0a\x00"), call(b"\x0a\x01")])
                self.assertLess(events.mock_calls.index(call.control.sendall(b"\x0a\x01")),
                                events.mock_calls.index(call.control.shutdown(socket.SHUT_WR)))
                self.assert_bounded_wait(process)
                self.assert_resources_cleaned(session, video, control, listener, process)

    def test_failed_power_on_during_stop_still_cleans_resources(self):
        with self.session_with_resources() as (session, events, video, control, listener, process):
            session.start()
            control.sendall.side_effect = OSError("power on failed")
            try:
                session.stop()
            except OSError:
                pass
            self.assertEqual(control.sendall.call_args_list, [call(b"\x0a\x00"), call(b"\x0a\x01")])
            calls = events.mock_calls
            self.assertLess(calls.index(call.control.sendall(b"\x0a\x01")),
                            calls.index(call.control.shutdown(socket.SHUT_WR)))
            self.assertLess(calls.index(call.control.shutdown(socket.SHUT_WR)),
                            next(i for i, item in enumerate(calls) if item[0] == "process.wait"))
            self.assert_bounded_wait(process)
            self.assert_resources_cleaned(session, video, control, listener, process)

    def test_stop_kills_server_after_graceful_shutdown_timeout(self):
        with self.session_with_resources() as (session, _, video, control, listener, process):
            session.start()
            process.wait.side_effect = [subprocess.TimeoutExpired("adb", 2), 0]
            with self.assertLogs("scrcpy", level="WARNING"):
                session.stop()
            self.assertEqual(control.sendall.call_args_list, [call(b"\x0a\x00"), call(b"\x0a\x01")])
            process.kill.assert_called_once()
            self.assertEqual(process.wait.call_count, 2)
            self.assert_resources_cleaned(session, video, control, listener, process)

    def test_start_failure_before_power_off_does_not_send_power_messages(self):
        with self.session_with_resources() as (session, _, video, control, listener, process):
            failure = ValueError("metadata failed")
            session._read_stream_metadata.side_effect = failure
            with self.assertRaises(ValueError) as raised:
                session.start()
            self.assertIs(raised.exception, failure)
            control.sendall.assert_not_called()
            session._start_decode_thread.assert_not_called()
            self.assert_resources_cleaned(session, video, control, listener, process)


if __name__ == "__main__":
    unittest.main()
