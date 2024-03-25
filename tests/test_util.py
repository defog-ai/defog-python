import unittest
from unittest.mock import Mock, MagicMock, patch
from defog.util import identify_categorical_columns, parse_update, get_feedback


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


class TestIdentifyCategoricalColumns(unittest.TestCase):
    def setUp(self):
        self.cur = Mock()
        self.cur.execute = MagicMock()
        self.cur.fetchone = MagicMock()
        self.cur.fetchall = MagicMock()

    def test_identify_categorical_columns_succeed(self):
        # Mock 2 distinct values each with their respective occurence counts of 3, 30
        self.cur.fetchone.return_value = (2,)
        self.cur.fetchall.return_value = [("value1", 3), ("value2", 30)]
        rows = [
            {"column_name": "test_column", "data_type": "varchar"},
            {"column_name": "test_column_int", "data_type": "bigint"},
        ]

        # Call the function
        result = identify_categorical_columns(self.cur, "test_table", rows)

        # Assert the results
        self.assertEqual(len(result), len(rows))
        self.assertEqual(result[0]["top_values"], "value1,value2")

    def test_identify_categorical_columns_exceed_threshold(self):
        # Mock 20 distinct values each with their respective occurence counts of 1
        self.cur.fetchone.return_value = (20,)
        self.cur.fetchall.return_value = [(f"value{i}", 1) for i in range(20)]
        rows = [{"column_name": "test_column", "data_type": "varchar"}]

        # Call the function
        result = identify_categorical_columns(
            self.cur, "test_table", rows, distinct_threshold=10
        )

        # Assert results is still the same as rows (not modified)
        self.assertEqual(result, rows)
        self.assertIn("column_name", result[0])
        self.assertIn("data_type", result[0])
        self.assertNotIn("top_values", result[0])

    def test_identify_categorical_columns_within_modified_threshold(self):
        # Mock 20 distinct values each with their respective occurence counts of 1
        self.cur.fetchone.return_value = (20,)
        self.cur.fetchall.return_value = [(f"value{i}", 1) for i in range(20)]
        rows = [{"column_name": "test_column", "data_type": "varchar"}]

        # Call the function
        result = identify_categorical_columns(
            self.cur, "test_table", rows, distinct_threshold=20
        )

        # Assert that we get 20 distinct values as required
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["top_values"],
            "value0,value1,value10,value11,value12,value13,value14,value15,value16,value17,value18,value19,value2,value3,value4,value5,value6,value7,value8,value9",
        )


if __name__ == "__main__":
    unittest.main()


class TestGetFeedback(unittest.TestCase):
    @patch("defog.util.prompt", return_value="y")
    @patch("requests.post")
    def test_positive_feedback(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated", "base_url")
        assert mock_post.call_count == 1
        self.assertIn("good", mock_post.call_args.kwargs["json"]["feedback"])
        self.assertNotIn("feedback_text", mock_post.call_args.kwargs["json"])

    @patch("defog.util.prompt", side_effect=["n", "bad query", "", "", ""])
    @patch("requests.post")
    def test_negative_feedback_with_text(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated", "base_url")
        # 2 calls: 1 to /feedback, 1 to /reflect_on_error
        assert mock_post.call_count == 2
        self.assertIn("api_key", mock_post.call_args.kwargs["json"]["api_key"])
        self.assertIn("user_question", mock_post.call_args.kwargs["json"]["question"])
        self.assertIn(
            "sql_generated", mock_post.call_args.kwargs["json"]["sql_generated"]
        )
        self.assertIn("bad query", mock_post.call_args.kwargs["json"]["error"])

    @patch("defog.util.prompt", side_effect=["n", "", "", "", ""])
    @patch("requests.post")
    def test_negative_feedback_without_text(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated", "base_url")
        # 2 calls: 1 to /feedback, 1 to /reflect_on_error
        assert mock_post.call_count == 2
        self.assertIn("api_key", mock_post.call_args.kwargs["json"]["api_key"])
        self.assertIn("user_question", mock_post.call_args.kwargs["json"]["question"])
        self.assertIn(
            "sql_generated", mock_post.call_args.kwargs["json"]["sql_generated"]
        )
        self.assertIn("", mock_post.call_args.kwargs["json"]["error"])

    @patch("defog.util.prompt", side_effect=["invalid", "y"])
    @patch("requests.post")
    def test_invalid_then_valid_input(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated", "base_url")
        assert mock_post.call_count == 1
        self.assertIn("good", mock_post.call_args.kwargs["json"]["feedback"])
        self.assertNotIn("feedback_text", mock_post.call_args.kwargs["json"])

    @patch("defog.util.prompt", side_effect=[""])
    @patch("requests.post")
    def test_skip_input(self, mock_post, mock_prompt):
        get_feedback("api_key", "db_type", "user_question", "sql_generated", "base_url")
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
