import unittest
from defog.util import parse_update


class TestDefogUtil(unittest.TestCase):
    def test_parse_update_1_key_edit(self):
        update_str = ["--app_name", "AWS"]
        attributes_list = ["app_name"]
        config = {"app_name": "GCP"}
        expected_output = {"app_name": "AWS"}
        self.assertEqual(
            parse_update(update_str, attributes_list, config), expected_output
        )

    def test_parse_update_1_key_insert(self):
        update_str = ["--app_name", "AWS"]
        attributes_list = ["app_name"]
        config = {"version": "1.0"}
        expected_output = {"app_name": "AWS", "version": "1.0"}
        self.assertEqual(
            parse_update(update_str, attributes_list, config), expected_output
        )

    def test_parse_update_1_key_not_exists(self):
        update_str = ["--app_name", "AWS"]
        attributes_list = ["version"]
        config = {"app_name": "GCP"}
        expected_output = config
        self.assertEqual(
            parse_update(update_str, attributes_list, config), expected_output
        )
        config = {"version": "2.0"}
        expected_output = config
        self.assertEqual(
            parse_update(update_str, attributes_list, config), expected_output
        )
        config = {"random_key": "random_value"}
        expected_output = config
        self.assertEqual(
            parse_update(update_str, attributes_list, config), expected_output
        )


if __name__ == "__main__":
    unittest.main()
