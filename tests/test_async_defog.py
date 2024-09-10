import shutil
import unittest
import asyncio
from defog import AsyncDefog
from defog.util import parse_update
import os
from unittest.mock import patch


class TestAsyncDefog(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # if connection.json exists, copy it to /tmp since we'll be overwriting it
        home_dir = os.path.expanduser("~")
        self.filepath = os.path.join(home_dir, ".defog", "connection.json")
        self.tmp_dir = os.path.join("/tmp")
        self.moved = False
        if os.path.exists(self.filepath):
            print("Moving connection.json to /tmp")
            if os.path.exists(os.path.join(self.tmp_dir, "connection.json")):
                os.remove(os.path.join(self.tmp_dir, "connection.json"))
            shutil.move(self.filepath, self.tmp_dir)
            self.moved = True

    @classmethod
    def tearDownClass(self):
        # copy back the original after all tests have completed
        if self.moved:
            print("Moving connection.json back to ~/.defog")
            shutil.move(os.path.join(self.tmp_dir, "connection.json"), self.filepath)

    def tearDown(self):
        # clean up connection.json created/saved after each test case
        if os.path.exists(self.filepath):
            print("Removing connection.json used for testing")
            os.remove(self.filepath)

    ### Case 1:
    def test_async_defog_bad_init_no_params(self):
        with self.assertRaises(ValueError):
            print("Testing AsyncDefog with no params")
            AsyncDefog()

    # test initialization with partial params
    def test_async_defog_good_init_no_db_creds(self):
        df = AsyncDefog("test_api_key", "redis")
        self.assertEqual(df.api_key, "test_api_key")
        self.assertEqual(df.db_type, "redis")
        self.assertEqual(df.db_creds, {})

    ### Case 2:
    # no connection file, no params
    def test_async_defog_bad_init_no_connection_file(self):
        with self.assertRaises(ValueError):
            print("Testing AsyncDefog with no connection file, no params")
            AsyncDefog()

    # no connection file, incomplete db_creds
    def test_async_defog_bad_init_incomplete_creds(self):
        with self.assertRaises(KeyError):
            AsyncDefog("test_api_key", "postgres", {"host": "some_host"})

    ### Case 3:
    def test_async_defog_good_init(self):
        print("testing AsyncDefog with good params")
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        df = AsyncDefog("test_api_key", "postgres", db_creds)
        self.assertEqual(df.api_key, "test_api_key")
        self.assertEqual(df.db_type, "postgres")
        self.assertEqual(df.db_creds, db_creds)

    ### Case 4:
    def test_async_defog_no_overwrite(self):
        db_creds = {
            "host": "host",
            "port": "port",
            "database": "database",
            "user": "user",
            "password": "password",
        }
        df1 = AsyncDefog("old_api_key", "postgres", db_creds)
        self.assertEqual(df1.api_key, "old_api_key")
        self.assertEqual(df1.db_type, "postgres")
        self.assertEqual(df1.db_creds, db_creds)
        self.assertTrue(os.path.exists(self.filepath))
        del df1
        df2 = AsyncDefog()  # should read connection.json
        self.assertEqual(df2.api_key, "old_api_key")
        self.assertEqual(df2.db_type, "postgres")
        self.assertEqual(df2.db_creds, db_creds)

    ### Case 5:
    @patch("builtins.input", lambda *args: "y")
    def test_async_defog_overwrite(self):
        db_creds = {
            "host": "host",
            "port": "port",
            "database": "database",
            "user": "user",
            "password": "password",
        }
        df = AsyncDefog("old_api_key", "postgres", db_creds)
        self.assertEqual(df.api_key, "old_api_key")
        self.assertEqual(df.db_type, "postgres")
        self.assertEqual(df.db_creds, db_creds)
        self.assertTrue(os.path.exists(self.filepath))
        df = AsyncDefog("new_api_key", "redshift")
        self.assertEqual(df.api_key, "new_api_key")
        self.assertEqual(df.db_type, "redshift")
        self.assertEqual(df.db_creds, {})

    @patch("builtins.input", lambda *args: "n")
    def test_async_defog_no_overwrite(self):
        db_creds = {
            "host": "host",
            "port": "port",
            "database": "database",
            "user": "user",
            "password": "password",
        }
        df = AsyncDefog("old_api_key", "postgres", db_creds)
        self.assertEqual(df.api_key, "old_api_key")
        self.assertEqual(df.db_type, "postgres")
        self.assertEqual(df.db_creds, db_creds)
        self.assertTrue(os.path.exists(self.filepath))
        df = AsyncDefog("new_api_key", "redshift")
        self.assertEqual(df.api_key, "new_api_key")
        self.assertEqual(df.db_type, "redshift")
        self.assertEqual(df.db_creds, {})

    # test check_db_creds with all the different supported db types
    def test_check_db_creds_postgres(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        AsyncDefog.check_db_creds("postgres", db_creds)
        AsyncDefog.check_db_creds("postgres", {})
        with self.assertRaises(KeyError):
            AsyncDefog.check_db_creds("postgres", {"host": "some_host"})

    def test_check_db_creds_redshift(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        AsyncDefog.check_db_creds("redshift", db_creds)
        AsyncDefog.check_db_creds("redshift", {})
        with self.assertRaises(KeyError):
            AsyncDefog.check_db_creds("redshift", {"host": "some_host"})

    def test_check_db_creds_mysql(self):
        db_creds = {
            "host": "some_host",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        AsyncDefog.check_db_creds("mysql", db_creds)
        AsyncDefog.check_db_creds("mysql", {})
        with self.assertRaises(KeyError):
            AsyncDefog.check_db_creds("mysql", {"host": "some_host"})

    async def test_check_db_creds_snowflake(self):
        db_creds = {
            "account": "some_account",
            "warehouse": "some_warehouse",
            "user": "some_user",
            "password": "some_password",
        }
        AsyncDefog.check_db_creds("snowflake", db_creds)
        AsyncDefog.check_db_creds("snowflake", {})
        with self.assertRaises(KeyError):
            AsyncDefog.check_db_creds("snowflake", {"account": "some_account"})

    def test_check_db_creds_mongo(self):
        db_creds = {"connection_string": "some_connection_string"}
        AsyncDefog.check_db_creds("mongo", db_creds)
        AsyncDefog.check_db_creds("mongo", {})
        with self.assertRaises(KeyError):
            AsyncDefog.check_db_creds("mongo", {"account": "some_account"})

    def test_check_db_creds_sqlserver(self):
        db_creds = {
            "server": "some_server",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        AsyncDefog.check_db_creds("sqlserver", db_creds)
        AsyncDefog.check_db_creds("sqlserver", {})
        with self.assertRaises(KeyError):
            AsyncDefog.check_db_creds("sqlserver", {"account": "some_account"})

    def test_check_db_creds_bigquery(self):
        db_creds = {"json_key_path": "some_json_key_path"}
        AsyncDefog.check_db_creds("bigquery", db_creds)
        AsyncDefog.check_db_creds("bigquery", {})
        with self.assertRaises(KeyError):
            AsyncDefog.check_db_creds("bigquery", {"account": "some_account"})

    def test_base64_encode_decode(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        df1 = AsyncDefog("test_api_key", "postgres", db_creds)
        df1_base64creds = df1.to_base64_creds()
        df2 = AsyncDefog(base64creds=df1_base64creds)
        self.assertEqual(df1.api_key, df2.api_key)
        self.assertEqual(df1.api_key, "test_api_key")
        self.assertEqual(df1.db_type, df2.db_type)
        self.assertEqual(df1.db_type, "postgres")
        self.assertEqual(df1.db_creds, df2.db_creds)
        self.assertEqual(df1.db_creds, db_creds)

    def test_save_json(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        _ = AsyncDefog("test_api_key", "postgres", db_creds, save_json=True)
        self.assertTrue(os.path.exists(self.filepath))

    def test_no_save_json(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        df_save = AsyncDefog("test_api_key", "postgres", db_creds, save_json=False)
        self.assertTrue(not os.path.exists(self.filepath))


if __name__ == "__main__":
    unittest.main()
