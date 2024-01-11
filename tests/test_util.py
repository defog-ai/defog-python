import unittest
from unittest.mock import patch

from defog.util import parse_update, get_feedback


class TestParseUpdate(unittest.TestCase):
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


class TestGetFeedback(unittest.TestCase):
    @patch("defog.util.prompt", return_value="y")
    @patch("requests.post")
    def test_positive_feedback(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated")
        mock_post.assert_called_once()
        self.assertIn("good", mock_post.call_args.kwargs["json"]["feedback"])
        self.assertNotIn("feedback_text", mock_post.call_args.kwargs["json"])

    @patch("defog.util.prompt", side_effect=["n", "bad query"])
    @patch("requests.post")
    def test_negative_feedback_with_text(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated")
        mock_post.assert_called_once()
        self.assertIn("bad", mock_post.call_args.kwargs["json"]["feedback"])
        self.assertIn("bad query", mock_post.call_args.kwargs["json"]["feedback_text"])

    @patch("defog.util.prompt", side_effect=["n", ""])
    @patch("requests.post")
    def test_negative_feedback_without_text(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated")
        mock_post.assert_called_once()
        self.assertIn("bad", mock_post.call_args.kwargs["json"]["feedback"])
        self.assertNotIn("feedback_text", mock_post.call_args.kwargs["json"])

    @patch("defog.util.prompt", side_effect=["invalid", "y"])
    @patch("requests.post")
    def test_invalid_then_valid_input(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated")
        mock_post.assert_called_once()
        self.assertIn("good", mock_post.call_args.kwargs["json"]["feedback"])
        self.assertNotIn("feedback_text", mock_post.call_args.kwargs["json"])

    @patch("defog.util.prompt", side_effect=[""])
    @patch("requests.post")
    def test_skip_input(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated")
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
