import unittest
from unittest import mock

from defog.query import execute_query_once, execute_query


class ExecuteQueryOnceTestCase(unittest.TestCase):
    @mock.patch("psycopg2.connect")
    def test_postgres_query_execution(self, mock_connect):
        # Mock the psycopg2.connect function
        mock_cursor = mock_connect.return_value.cursor.return_value
        mock_cursor.description = [("col1",), ("col2",)]
        mock_cursor.fetchall.return_value = [("data1", "data2"), ("data3", "data4")]

        db_type = "postgres"
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
        }
        query = "SELECT * FROM table_name;"

        colnames, results = execute_query_once(db_type, db_creds, query)

        # Add your assertions here to validate the results
        self.assertEqual(colnames, ["col1", "col2"])
        self.assertEqual(results, [("data1", "data2"), ("data3", "data4")])
        print("Postgres query execution test passed!")

    @mock.patch("requests.post")
    @mock.patch("defog.query.execute_query_once")
    def test_execute_query_success(self, mock_execute_query_once, mock_requests_post):
        # Mock the execute_query_once function
        db_type = "postgres"
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
        }
        query1 = "SELECT * FROM table_name;"
        query2 = "SELECT * FROM new_table_name;"
        api_key = "your_api_key"
        question = "your_question"
        hard_filters = "your_hard_filters"
        retries = 3

        # Set up the mock responses
        mock_execute_query_once.return_value = (
            ["col1", "col2"],
            [("data1", "data2"), ("data3", "data4")],
        )
        mock_response = mock.Mock()
        mock_response.json.return_value = {"new_query": query2}
        mock_requests_post.return_value = mock_response

        # Call the function being tested
        colnames, results, rcv_query = execute_query(
            query1, api_key, db_type, db_creds, question, hard_filters, retries
        )

        # Assert the expected behavior
        mock_execute_query_once.assert_called_once_with(db_type, db_creds, query1)
        mock_requests_post.assert_not_called()
        self.assertEqual(colnames, ["col1", "col2"])
        self.assertEqual(results, [("data1", "data2"), ("data3", "data4")])
        self.assertEqual(rcv_query, query1)  # should return the original query

    @mock.patch("requests.post")
    @mock.patch("defog.query.execute_query_once")
    def test_execute_query_retry_limit_exceeded(
        self, mock_execute_query_once, mock_requests_post
    ):
        db_type = "postgres"
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
        }
        query1 = "SELECT * FROM table_name;"
        query2 = "SELECT * FROM table_name WHERE colour='blue';"
        api_key = "your_api_key"
        question = "your_question"
        hard_filters = "your_hard_filters"
        retries = 3
        colnames = (["col1", "col2"],)
        results = [("data1", "data2"), ("data3", "data4")]

        # Mock the execute_query_once function to raise an exception the first
        # time it is called and return the results the second time it is called
        err_msg = "Test exception"

        def side_effect(db_type, db_creds, query):
            if query == query1:
                raise Exception(err_msg)
            else:
                return colnames, results

        mock_execute_query_once.side_effect = side_effect
        # mock the response to return {"new_query": query2} when .json() is called
        mock_response = mock.Mock()
        mock_response.json.return_value = {"new_query": query2}
        mock_requests_post.return_value = mock_response

        # Call the function being tested
        ret = execute_query(
            query1, api_key, db_type, db_creds, question, hard_filters, retries
        )
        # should return new query2 instead of query1
        self.assertEqual(ret, (colnames, results, query2))

        # Assert the mock function calls
        mock_execute_query_once.assert_called_with(db_type, db_creds, query2)
        mock_requests_post.assert_called_with(
            "https://api.defog.ai/retry_query_after_error",
            json={
                "api_key": api_key,
                "previous_query": query1,
                "error": err_msg,
                "db_type": db_type,
                "hard_filters": hard_filters,
                "question": question,
            },
        )


if __name__ == "__main__":
    unittest.main()
