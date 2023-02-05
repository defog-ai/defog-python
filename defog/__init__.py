import requests
import json

class Defog:
    """
    The main class for Defog
    """

    def __init__(self, api_key: str, db_type: str = "postgres", db_creds: dict = None):
        """
        Initializes the Defog class.
        :param api_key: The API key for the defog account.
        """
        if db_type not in ["postgres", "redshift", "mysql", "bigquery", "mongo", "snowflake", "sqlserver"]:
            raise Exception(f"Database `{db_type}` is not supported right now. db_type must be one of 'postgres', 'redshift', 'mysql', 'bigquery', 'mongo', 'sqlserver'.")
        self.api_key = api_key
        self.db_type = db_type
        self.db_creds = db_creds
    
    def generate_postgres_schema(self, tables: list):
        try:
            import psycopg2
        except:
            raise Exception("psycopg2 not installed.")
        
        conn = psycopg2.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}
        
        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            cur.execute("SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s;", (table_name,))
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows
        
        conn.close()

        print("Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes...")
        # send the schemas dict to the defog servers
        r = requests.post("https://api.defog.ai/get_postgres_schema_gsheets",
            json={
                "api_key": self.api_key,
                "schemas": schemas
            }
        )
        resp = r.json()
        try:
            gsheet_url = resp['sheet_url']
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp['message']
    
    def generate_redshift_schema(self, tables: list):
        try:
            import psycopg2
        except:
            raise Exception("psycopg2 not installed.")
        
        conn = psycopg2.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}
        
        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            cur.execute("SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s;", (table_name,))
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows
        
        conn.close()

        print("Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes...")
        # send the schemas dict to the defog servers
        r = requests.post("https://api.defog.ai/get_postgres_schema_gsheets",
            json={
                "api_key": self.api_key,
                "schemas": schemas
            }
        )
        resp = r.json()
        try:
            gsheet_url = resp['sheet_url']
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp['message']

    def generate_mysql_schema(self, tables: list):
        try:
            import mysql.connector
        except:
            raise Exception("mysql-connector not installed.")
        
        conn = mysql.connector.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}
        
        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;", (table_name,))
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows
        
        conn.close()

        print("Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes...")
        # send the schemas dict to the defog servers
        r = requests.post("https://api.defog.ai/get_postgres_schema_gsheets",
            json={
                "api_key": self.api_key,
                "schemas": schemas
            }
        )
        resp = r.json()
        try:
            gsheet_url = resp['sheet_url']
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp['message']
    
    def generate_sqlserver_schema(self, tables: list):
        try:
            import pyodbc
        except:
            raise Exception("pyodbc not installed.")
        
        conn = pyodbc.connect(self.db_creds)
        cur = conn.cursor()
        schemas = {}

        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows
        
        conn.close()

        print("Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes...")
        # send the schemas dict to the defog servers
        r = requests.post("https://api.defog.ai/get_postgres_schema_gsheets",
            json={
                "api_key": self.api_key,
                "schemas": schemas
            }
        )
        resp = r.json()
        try:
            gsheet_url = resp['sheet_url']
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp['message']

    def generate_snowflake_schema(self, tables: list):
        try:
            import snowflake.connector
        except:
            raise Exception("snowflake-connector not installed.")
        
        conn = snowflake.connector.connect(
            user=self.db_creds['user'],
            password=self.db_creds['password'],
            account=self.db_creds['account'],
        )
        conn.cursor().execute(f"USE WAREHOUSE {self.db_creds['warehouse']}") # set the warehouse

        schemas = {}
        alt_types = {"DATE": "TIMESTAMP", "TEXT": "VARCHAR", "FIXED": "NUMERIC"}
        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            rows = []
            for row in conn.cursor().execute(f"SHOW COLUMNS IN {table_name};"):
                rows.append(row)
            rows = [{"column_name": i[2], "data_type": json.loads(i[3])['type'] } for i in rows]
            for idx, row in enumerate(rows):
                if row['data_type'] in alt_types:
                    row['data_type'] = alt_types[row['data_type']]
                rows[idx] = row
            print(rows)
            schemas[table_name] = rows
        
        conn.close()

        print("Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes...")
        # send the schemas dict to the defog servers
        r = requests.post("https://api.defog.ai/get_postgres_schema_gsheets",
            json={
                "api_key": self.api_key,
                "schemas": schemas
            }
        )
        resp = r.json()
        try:
            gsheet_url = resp['sheet_url']
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp['message']

    def generate_mongo_schema(self, collections: list):
        try:
            from pymongo import MongoClient
        except:
            raise Exception("pymongo not installed.")
        
        client = MongoClient(self.db_creds['connection_string'])
        db = client.get_database()
        
        schemas = {}
        
        print("Getting schema for each collection in your database...")
        # get the schema for each table
        for collection_name in collections:
            collection = db[collection_name]
            rows = collection.find_one()
            rows = [{"field_name": i, "data_type": type(rows[i]).__name__} for i in rows]
            schemas[collection_name] = rows
        
        client.close()

        print("Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes...")
        # send the schemas dict to the defog servers
        r = requests.post("https://api.defog.ai/get_mongo_schema_gsheets",
            json={
                "api_key": self.api_key,
                "schemas": schemas
            }
        )
        resp = r.json()
        try:
            gsheet_url = resp['sheet_url']
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp['message']

    def generate_bigquery_schema(self, tables: list):
        try:
            from google.cloud import bigquery
        except:
            raise Exception("google-cloud-bigquery not installed.")
        
        client = bigquery.Client.from_service_account_json(self.db_creds)
        schemas = {}
        
        print("Getting the schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            table = client.get_table(table_name)
            rows = table.schema
            rows = [{"column_name": i.name, "data_type": i.field_type} for i in rows]
            schemas[table_name] = rows
        
        client.close()

        print("Sending the schema to Defog servers and generating a Google Sheet. This might take up to 2 minutes...")
        # send the schemas dict to the defog servers
        r = requests.post("https://api.defog.ai/get_bigquery_schema_gsheets",
            json={
                "api_key": self.api_key,
                "schemas": schemas
            }
        )
        resp = r.json()
        try:
            gsheet_url = resp['sheet_url']
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp['message']
    
    def generate_db_schema(self, tables: list):
        if self.db_type == "postgres":
            return self.generate_postgres_schema(tables)
        elif self.db_type == "mysql":
            return self.generate_mysql_schema(tables)
        elif self.db_type == "mongo":
            return self.generate_mongo_schema(tables)
        elif self.db_type == "bigquery":
            return self.generate_bigquery_schema(tables)
        elif self.db_type == "redshift":
            return self.generate_redshift_schema(tables)
        elif self.db_type == "snowflake":
            return self.generate_snowflake_schema(tables)
        elif self.db_type == "sqlserver":
            return self.generate_sqlserver_schema(tables)
        else:
            raise Exception("Invalid database type. Valid types are: postgres, mysql, mongo, bigquery, and redshift")

    def update_mysql_schema(self, gsheet_url : str):
        """
        Updates the postgres schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post("https://api.defog.ai/update_postgres_schema",
            json={
                "api_key": self.api_key,
                "gsheet_url": gsheet_url
            }
        )
        resp = r.json()
        return resp
    
    def update_postgres_schema(self, gsheet_url : str):
        """
        Updates the postgres schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post("https://api.defog.ai/update_postgres_schema",
            json={
                "api_key": self.api_key,
                "gsheet_url": gsheet_url
            }
        )
        resp = r.json()
        return resp
    
    def update_redshift_schema(self, gsheet_url : str):
        """
        Updates the redshift schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post("https://api.defog.ai/update_postgres_schema",
            json={
                "api_key": self.api_key,
                "gsheet_url": gsheet_url
            }
        )
        resp = r.json()
        return resp
    
    def update_mongo_schema(self, gsheet_url : str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post("https://api.defog.ai/update_mongo_schema",
            json={
                "api_key": self.api_key,
                "gsheet_url": gsheet_url
            }
        )
        resp = r.json()
        return resp
    
    def update_bigquery_schema(self, gsheet_url : str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post("https://api.defog.ai/update_bigquery_schema",
            json={
                "api_key": self.api_key,
                "gsheet_url": gsheet_url
            }
        )
        resp = r.json()
        return resp
    
    def update_snowflake_schema(self, gsheet_url : str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post("https://api.defog.ai/update_snowflake_schema",
            json={
                "api_key": self.api_key,
                "gsheet_url": gsheet_url
            }
        )
        resp = r.json()
        return resp
    
    def update_sqlserver_schema(self, gsheet_url : str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post("https://api.defog.ai/update_postgres_schema",
            json={
                "api_key": self.api_key,
                "gsheet_url": gsheet_url
            }
        )
        resp = r.json()
        return resp
    
    def update_db_schema(self, gsheet_url: str):
        print("Updating the schema on the Defog servers. This might take a couple minutes...")
        if self.db_type == "postgres":
            return self.update_postgres_schema(gsheet_url)
        elif self.db_type == "mysql":
            return self.update_mysql_schema(gsheet_url)
        elif self.db_type == "mongo":
            return self.update_mongo_schema(gsheet_url)
        elif self.db_type == "bigquery":
            return self.update_bigquery_schema(gsheet_url)
        elif self.db_type == "redshift":
            return self.update_redshift_schema(gsheet_url)
        elif self.db_type == "snowflake":
            return self.update_snowflake_schema(gsheet_url)
        elif self.db_type == "sqlserver":
            return self.update_sqlserver_schema(gsheet_url)
        else:
            raise Exception("Invalid database type. Valid types are: postgres, mysql, mongo, bigquery, and redshift")

    def get_query(self, question: str, hard_filters: str = None):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :return: The response from the defog server.
        """
        r = requests.post("https://api.defog.ai/generate_query",
            json={
                "question": question,
                "api_key": self.api_key,
                "hard_filters": hard_filters,
                "db_type": self.db_type
            }
        )
        resp = r.json()
        query_generated = resp.get("query_generated")
        ran_successfully = resp["ran_successfully"]
        error_message =  resp.get("error_message")
        query_db = resp.get("query_db", 'postgres')
        return {
            "query_generated": query_generated,
            "ran_successfully": ran_successfully,
            "error_message": error_message,
            "query_db": query_db
        }
    
    def run_query(self, question: str, hard_filters: str = None):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :return: The response from the defog server.
        """
        print(f"Generating the query for your question: {question}...")
        query =  self.get_query(question, hard_filters)
        if query["ran_successfully"]:
            print("Query generated, now running it on your database...")
            if query['query_db'] == "postgres" or query['query_db'] == "redshift":
                try:
                    import psycopg2
                except:
                    raise Exception("psycopg2 not installed.")
                
                conn = psycopg2.connect(**self.db_creds)
                cur = conn.cursor()
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True
                    }
                except Exception as e:
                    print(f"Query generated was: {query['query_generated']}")
                    print(f"There was an error {str(e)} when running the previous query. Retrying with adaptive learning...")
                    # retry the query with the exception
                    r = requests.post("https://api.defog.ai/retry_query_after_error",
                        json={
                            "api_key": self.api_key,
                            "previous_query": query["query_generated"],
                            "error": str(e),
                            "db_type": self.db_type,
                            "hard_filters": hard_filters,
                            "question": question
                        }
                    )
                    query = r.json()
                    conn = psycopg2.connect(**self.db_creds)
                    cur = conn.cursor()
                    try:
                        cur.execute(query["new_query"])
                        colnames = [desc[0] for desc in cur.description]
                        result = cur.fetchall()
                        cur.close()
                        conn.close()
                        print("Query ran succesfully!")
                        return {
                            "columns": colnames,
                            "data": result,
                            "query_generated": query["new_query"],
                            "ran_successfully": True
                        }
                    except Exception as e:
                        return {"error_message": str(e), "ran_successfully": False}
            elif query['query_db'] == "mysql":
                try:
                    import mysql.connector
                except:
                    raise Exception("mysql.connector not installed.")
                conn = mysql.connector.connect(**self.db_creds)
                cur = conn.cursor()
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True
                    }
                except Exception as e:
                    print(f"Query generated was: {query['query_generated']}")
                    print(f"There was an error {str(e)} when running the previous query. Retrying with adaptive learning...")
                    # retry the query with the exception
                    r = requests.post("https://api.defog.ai/retry_query_after_error",
                        json={
                            "api_key": self.api_key,
                            "previous_query": query["query_generated"],
                            "error": str(e),
                            "db_type": self.db_type,
                            "hard_filters": hard_filters,
                            "question": question
                        }
                    )
                    query = r.json()
                    conn = mysql.connector.connect(**self.db_creds)
                    cur = conn.cursor()
                    try:
                        cur.execute(query["new_query"])
                        colnames = [desc[0] for desc in cur.description]
                        result = cur.fetchall()
                        cur.close()
                        conn.close()
                        print("Query ran succesfully!")
                        return {
                            "columns": colnames,
                            "data": result,
                            "query_generated": query["new_query"],
                            "ran_successfully": True
                        }
                    except Exception as e:
                        return {"error_message": str(e), "ran_successfully": False}
            elif query['query_db'] == "mongo":
                try:
                    from pymongo import MongoClient
                except:
                    raise Exception("pymongo not installed.")
                client = MongoClient(self.db_creds["connection_string"])
                db = client.get_database()
                try:
                    results = eval(f"{query['query_generated']}")
                    results = [i for i in results]
                    if len(results) > 0:
                        columns = results[0].keys()
                    else:
                        columns = []
                    return {
                        "columns": columns, # assumes that all objects have the same keys
                        "data": results,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True
                    }
                except Exception as e:
                    return {"error_message": str(e), "ran_successfully": False}
            elif query['query_db'] == "bigquery":
                try:
                    from google.cloud import bigquery
                except:
                    raise Exception("google.cloud.bigquery not installed.")
                
                json_key = self.db_creds
                client = bigquery.Client.from_service_account_json(json_key)
                query_job = client.query(query["query_generated"])
                results = query_job.result()
                columns = [i.name for i in results.schema]
                rows = []
                for row in results:
                    rows.append([row[i] for i in range(len(row))])
                
                return {
                    "columns": columns, # assumes that all objects have the same keys
                    "data": rows,
                    "query_generated": query["query_generated"],
                    "ran_successfully": True
                }
            elif query['query_db'] == "snowflake":
                try:
                    import snowflake.connector
                except:
                    raise Exception("snowflake.connector not installed.")
                conn = snowflake.connector.connect(
                    user=self.db_creds["user"],
                    password=self.db_creds["password"],
                    account=self.db_creds["account"],
                )
                cur = conn.cursor()
                cur.execute(f"USE WAREHOUSE {self.db_creds['warehouse']}") # set the warehouse
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True
                    }
                except Exception as e:
                    return {"error_message": str(e), "ran_successfully": False}
            elif query['query_db'] == "sqlserver":
                try:
                    import pyodbc
                except:
                    raise Exception("pyodbc not installed.")
                conn = pyodbc.connect(self.db_creds)
                cur = conn.cursor()
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True
                    }
                except Exception as e:
                    return {"error_message": str(e), "ran_successfully": False}
            else:
                raise Exception("Database type not yet supported.")
        else:
            raise Exception(query["error_message"])