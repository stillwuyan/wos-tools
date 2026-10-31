import logging
import unittest

from log import FixedPrefixFormatter


class FixedPrefixFormatterTests(unittest.TestCase):
    def test_level_names_are_exactly_four_characters(self):
        formatter = FixedPrefixFormatter()
        for level, expected in ((logging.DEBUG, "DEBU"), (logging.INFO, "INFO"),
                                (logging.WARNING, "WARN"), (logging.ERROR, "ERRO"),
                                (logging.CRITICAL, "CRIT")):
            with self.subTest(level=level):
                record = logging.LogRecord("test", level, __file__, 1, "message", (), None)
                result = formatter.format(record)
                self.assertIn(f"][{expected}][test  ] message", result)
                self.assertEqual(record.levelname, logging.getLevelName(level))

    def test_short_custom_level_is_padded(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
        record.levelname = "IO"
        self.assertIn("][IO  ][test  ] message", FixedPrefixFormatter().format(record))

    def test_module_name_is_limited_to_six_characters(self):
        record = logging.LogRecord("automation.engine", logging.INFO, __file__, 1, "message", (), None)
        self.assertIn("][INFO][automa] message", FixedPrefixFormatter().format(record))
        self.assertEqual(record.name, "automation.engine")


if __name__ == "__main__":
    unittest.main()
