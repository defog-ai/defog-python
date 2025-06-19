import unittest
from defog import Defog


class TestDefog(unittest.TestCase):
    def test_defog_good_init(self):
        print("testing Defog with good params")
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        df = Defog(api_key="test_api_key", db_type="postgres", db_creds=db_creds)
        self.assertEqual(df.api_key, "test_api_key")
        self.assertEqual(df.db_type, "postgres")
        self.assertEqual(df.db_creds, db_creds)

    def test_defog_good_no_api_key(self):
        print("testing Defog with good params")
        db_creds = {
            "host": "some_host",
            "port": "some_port",
            "database": "some_database",
            "user": "some_user",
            "password": "some_password",
        }
        df = Defog(db_type="postgres", db_creds=db_creds)
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
        Defog.check_db_creds("snowflake", {})
        with self.assertRaises(KeyError):
            # incomplete keys
            Defog.check_db_creds("snowflake", {"account": "some_account"})

    def test_check_db_creds_sqlserver(self):
        db_creds = {
            "server": "some_server",
            "user": "some_user",
            "password": "some_password",
        }
        Defog.check_db_creds("sqlserver", db_creds)
        Defog.check_db_creds("sqlserver", {})
        with self.assertRaises(KeyError):
            # wrong key
            Defog.check_db_creds("sqlserver", {"account": "some_account"})

    def test_check_db_creds_bigquery(self):
        db_creds = {"json_key_path": "some_json_key_path"}
        Defog.check_db_creds("bigquery", db_creds)
        Defog.check_db_creds("bigquery", {})
        with self.assertRaises(KeyError):
            # wrong key
            Defog.check_db_creds("bigquery", {"account": "some_account"})


if __name__ == "__main__":
    unittest.main()
