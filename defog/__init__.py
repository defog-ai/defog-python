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
    
    def get_query(self, question: str, hard_filters: list):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :param hard_filters: The hard filters to be applied.
        :return: The response from the defog server.
        """
        r = requests.post("https://api.defog.ai/generate_query",
            json={
                "question": question,
                "hard_filters": hard_filters,
                "api_key": self.api_key
            }
        )
        resp = r.json()
        query_generated = resp["query_generated"]
        ran_successfully = resp["ran_successfully"]
        error_message =  resp["error_message"]
        query_db = resp.get("query_db", 'postgres')
        return {
            "query_generated": query_generated,
            "ran_successfully": ran_successfully,
            "error_message": error_message,
            "query_db": query_db
        }
    
    def run_query(self, question: str, hard_filters: list = []):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :param hard_filters: The hard filters to be applied.
        :return: The response from the defog server.
        """
        query =  self.get_query(self.api_key, question, hard_filters)
        if query["ran_successfully"]:
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