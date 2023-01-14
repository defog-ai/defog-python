import requests

class Defog:
    """
    The main class for Defog
    """

    def __init__(self, api_key: str, db_type: str = None, db_creds: dict = None):
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
        r = requests.post("https://api.defog.ai/get_postgres_creds",
            json={
                "api_key": self.api_key,
            }
        )
        resp = r.json()
        creds = resp['postgres_creds']
        self.db_type = "postgres"
        self.db_creds = {
            "host": creds["postgres_host"],
            "port": creds["postgres_port"],
            "database": creds["postgres_db"],
            "user": creds["postgres_username"],
            "password": creds["postgres_password"]
        }
        return True
    
    def get_query(self, question: str):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :return: The response from the defog server.
        """
        r = requests.post("https://api.defog.ai/generate_query",
            json={
                "question": question,
                "api_key": self.api_key
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
    
    def run_query(self, question: str):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :return: The response from the defog server.
        """
        print("generating the SQL query for your question...")
        query =  self.get_query(question)
        if query["ran_successfully"]:
            print("SQL query generated, now running it on your database...")
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
                        "sql": query["query_generated"],
                        "ran_successfully": True
                    }
                except Exception as e:
                    return {"error_message": str(e), "ran_successfully": False}
            else:
                raise Exception("Database type not yet supported.")
        else:
            raise Exception(query["error_message"])