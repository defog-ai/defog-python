import os
from chalice import Chalice

app = Chalice(app_name="defog-lambda")

from defog import Defog
import json

creds64_str = os.environ["DEFOG_CREDS_64"]
print(f"creds64_str = {creds64_str}")
defog = Defog(
    base64creds=creds64_str, save_json=False
)  # don't save the json file as no file system in lambda
print(f"Defog object created = {defog.__dict__}")


@app.route("/", methods=["POST"])
def answer():
    # This is the JSON body the user sent in their POST request.
    query = app.current_request.json_body
    print(f"query = {query}")
    question = query["question"]
    hard_filters = query.get("hard_filters")
    answer = defog.run_query(question, hard_filters=hard_filters)
    answer = json.dumps(answer, default=str)
    return answer
