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
        self.api_key = api_key
        self.db_type = db_type
        self.db_creds = db_creds
        if db_creds is None:
            self.update_db_creds()
    
    def update_db_creds(self):
        """
        Updates the database type.
        """
        print("Fetching database credentials...")
        r = requests.post("https://api.defog.ai/get_db_creds",
            json={
                "api_key": self.api_key,
            }
        )
        resp = r.json()
        db_type = resp['db_type']
        self.db_type = db_type
        if db_type == "postgres":
            creds = resp['postgres_creds']
            self.db_creds = {
                "host": creds["postgres_host"],
                "port": creds["postgres_port"],
                "database": creds["postgres_db"],
                "user": creds["postgres_username"],
                "password": creds["postgres_password"]
            }
            return True
        elif db_type == "mongo":
            creds = resp['mongo_creds']
            self.db_creds = {
                "connection_string": creds["mongo_connection_string"]
            }
            return True
        elif self.db_type == "bigquery":
            print("Please enter the path to your service account json file when initializing Defog\n\ndefog = Defog(api_key=key, db_type='bigquery', db_creds='path/to/service_account.json)")
            pass
        else:
            raise Exception("Database type not yet supported.")
    
    def generate_postgres_schema(self, tables: list, output_file: str = "schemas.json"):
        try:
            import psycopg2
        except:
            raise Exception("psycopg2 not installed.")
        
        conn = psycopg2.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}
        # get the schema for each table
        for table_name in tables:
            cur.execute("SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s;", (table_name,))
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows
        
        conn.close()

        with open(output_file, "w") as f:
            json.dump(schemas, f, indent=4)
        return schemas

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
        print("generating the query for your question...")
        query =  self.get_query(question, hard_filters)
        if query["ran_successfully"]:
            print("Query generated, now running it on your database...")
            if query['query_db'] == "postgres":
                try:
                    import psycopg2
                except:
                    raise Exception("psycopg2 not installed.")
                
                try:
                    conn = psycopg2.connect(**self.db_creds)
                    cur = conn.cursor()
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
            elif query['query_db'] == "mongo":
                try:
                    from pymongo import MongoClient
                except:
                    raise Exception("pymongo not installed.")
                try:
                    client = MongoClient(self.db_creds["connection_string"])
                    db = client.get_database()
                    results = eval(f"db.{query['query_generated']}")
                    return {
                        "columns": results[0].keys(), # assumes that all objects have the same keys
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
                
                json_key = self.db_creds["json_key"]
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
            else:
                raise Exception("Database type not yet supported.")
        else:
            raise Exception(query["error_message"])