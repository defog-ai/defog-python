import unittest
from unittest.mock import Mock, MagicMock
from defog.util import identify_categorical_columns


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
