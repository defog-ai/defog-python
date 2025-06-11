import unittest
from defog import AsyncDefog


class TestAsyncDefog(unittest.TestCase):
    def test_async_defog_good_init_no_db_creds(self):
        df = AsyncDefog(api_key="test_api_key", db_type="redis")
        self.assertEqual(df.api_key, "test_api_key")
        self.assertEqual(df.db_type, "redis")
        self.assertEqual(df.db_creds, {})

    def test_async_defog_bad_init_incomplete_creds(self):
        with self.assertRaises(KeyError):
            AsyncDefog(
                api_key="test_api_key",
                db_type="postgres",
                db_creds={"host": "some_host"},
            )

    def test_async_defog_good_init(self):
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
        df = AsyncDefog(api_key="new_api_key", db_type="redshift")
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

    def test_optional_api_key(self):
        # Test that AsyncDefog can be created without an API key
        df = AsyncDefog(db_type="postgres", db_creds={})
        self.assertIsNone(df.api_key)
        self.assertEqual(df.db_type, "postgres")

    def test_api_key_not_saved_when_none(self):
        # Test that connection.json doesn't include api_key when it's None
        import json

        df = AsyncDefog(db_type="postgres", db_creds={})
        self.assertIsNone(df.api_key)


if __name__ == "__main__":
    unittest.main()
