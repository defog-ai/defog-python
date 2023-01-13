import requests

def get_defog_query(api_key: str, question: str, hard_filters: list):
    """
    Sends the query to the defog servers, and return the response.
    :param api_key: The API key for the defog account.
    :param question: The question to be asked.
    :param hard_filters: The hard filters to be applied.
    :return: The response from the defog server.
    """
    
    r = requests.post("https://api.defog.ai/generate_query",
      json={
        question: question,
        hard_filters: hard_filters
      }, headers={
        "x-api-key": api_key
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