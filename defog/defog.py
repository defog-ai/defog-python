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
    
    def get_query(self, question: str, hard_filters: list):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :param hard_filters: The hard filters to be applied.
        :return: The response from the defog server.
        """
        r = requests.post("https://api.defog.ai/generate_query",
            json={
                question: question,
                hard_filters: hard_filters
            }, headers={
                "x-api-key": self.api_key
            }
        )
        resp = r.json()
        query_generated = resp["query_generated"]
        ran_successfully = resp["ran_successfully"]
        error_message =  resp["error_message"]
        return {
            "query_generated": query_generated,
            "ran_successfully": ran_successfully,
            "error_message": error_message
        }
    
    def run_query(self, question: str, hard_filters: list):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :param hard_filters: The hard filters to be applied.
        :return: The response from the defog server.
        """
        if self.db_type is None or self.db_creds is None:
            raise Exception("Database type and credentials not provided.")
        query =  self.get_query(self.api_key, question, hard_filters)
        if query["ran_successfully"]:
            if self.db_type == "postgres":
                try:
                    import psycopg2
                except:
                    raise Exception("psycopg2 not installed.")
                
                try:
                    conn = psycopg2.connect(**self.db_creds)
                    cur = conn.cursor()
                    cur.execute(query["query_generated"])
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    return result
                except Exception as e:
                    raise Exception(e)
            else:
                raise Exception("Database type not yet supported.")
        else:
            raise Exception(query["error_message"])