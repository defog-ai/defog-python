import shutil
import unittest
from defog import Defog
import os


class TestMyFunction(unittest.TestCase):
    def setUp(self) -> None:
        # if connection.json exists, copy it to /tmp since we'll be overwriting it
        home_dir = os.path.expanduser("~")
        self.filepath = os.path.join(home_dir, ".defog", "connection.json")
        self.tmp_dir = os.path.join("/tmp")
        self.moved = False
        if os.path.exists(self.filepath):
            print("Moving connection.json to /tmp")
            shutil.move(self.filepath, self.tmp_dir)
            self.moved = True
        return super().setUp()

    def tearDown(self) -> None:
        # clean up connection.json used for testing and copy back the original
        if os.path.exists(self.filepath):
            print("Removing connection.json used for testing")
            os.remove(self.filepath)
        if self.moved:
            print("Moving connection.json back to ~/.defog")
            shutil.move(os.path.join(self.tmp_dir, "connection.json"), self.filepath)
        return super().tearDown()

    # test that Defog raises errors when initialized with bad parameters
    def test_defog_bad_init(self):
        with self.assertRaises(ValueError):
            # no connection file, no params
            df = Defog(None)
        with self.assertRaises(ValueError):
            # no connection file, no db_creds
            df = Defog("test_api_key")
        with self.assertRaises(ValueError):
            # no connection file, wrong db_type
            df = Defog("test_api_key", "mysql")
        with self.assertRaises(KeyError):
            # no connection file, incomplete db_creds
            df = Defog("test_api_key", "postgres", {"host": "some_host"})

    # test that Defog initializes correctly when given good parameters
    def test_defog_good_init(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        df = Defog("test_api_key", "postgres", db_creds)
        self.assertEqual(df.api_key, "test_api_key")
        self.assertEqual(df.db_type, "postgres")
        self.assertEqual(df.db_creds, db_creds)

    # test check_db_creds with all the different supported db types
    def test_check_db_creds_postgres(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        Defog.check_db_creds("postgres", db_creds)
        with self.assertRaises(KeyError):
            Defog.check_db_creds("postgres", {})
        with self.assertRaises(KeyError):
            Defog.check_db_creds("postgres", {"host": "some_host"})

    def test_check_db_creds_redshift(self):
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        Defog.check_db_creds("redshift", db_creds)
        with self.assertRaises(KeyError):
            Defog.check_db_creds("redshift", {})
        with self.assertRaises(KeyError):
            # incomplete keys
            Defog.check_db_creds("redshift", {"host": "some_host"})

    def test_check_db_creds_mysql(self):
        db_creds = {
            "host": "some_host",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        Defog.check_db_creds("mysql", db_creds)
        with self.assertRaises(KeyError):
            Defog.check_db_creds("mysql", {})
        with self.assertRaises(KeyError):
            # incomplete keys
            Defog.check_db_creds("mysql", {"host": "some_host"})

    def test_check_db_creds_snowflake(self):
        db_creds = {
            "account": "some_account",
            "warehouse": "some_warehouse",
            "user": "some_user",
            "password": "some_password",
        }
        Defog.check_db_creds("snowflake", db_creds)
        with self.assertRaises(KeyError):
            Defog.check_db_creds("snowflake", {})
        with self.assertRaises(KeyError):
            # incomplete keys
            Defog.check_db_creds("snowflake", {"account": "some_account"})

    def test_check_db_creds_mongo(self):
        db_creds = {"connection_string": "some_connection_string"}
        Defog.check_db_creds("mongo", db_creds)
        with self.assertRaises(KeyError):
            Defog.check_db_creds("mongo", {})
        with self.assertRaises(KeyError):
            # wrong key
            Defog.check_db_creds("mongo", {"account": "some_account"})

    def test_check_db_creds_sqlserver(self):
        db_creds = {"connection_string": "some_connection_string"}
        Defog.check_db_creds("sqlserver", db_creds)
        with self.assertRaises(KeyError):
            Defog.check_db_creds("sqlserver", {})
        with self.assertRaises(KeyError):
            # wrong key
            Defog.check_db_creds("sqlserver", {"account": "some_account"})

    def test_check_db_creds_bigquery(self):
        db_creds = {"json_key_path": "some_json_key_path"}
        Defog.check_db_creds("bigquery", db_creds)
        with self.assertRaises(KeyError):
            Defog.check_db_creds("bigquery", {})
        with self.assertRaises(KeyError):
            # wrong key
            Defog.check_db_creds("bigquery", {"account": "some_account"})


if __name__ == "__main__":
    unittest.main()
