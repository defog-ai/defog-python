# TL;DR
Defog converts your natural language text queries into SQL and other machine readable code

![](defog-python.gif)

# Installation
`pip install defog`

# Getting your API Key
You can get your API key by emailing founders@defog.ai.

# Usage

## Running after giving your database details to Defog

If you give us your database credentials, you can just run the following code

```
from defog import Defog

defog = Defog(api_key="YOUR_API_KEY")

question = "question asked by a user"

results = defog.run_query(
  question=question
)
# {"is_success": True, "error_message": "", "col_names": [...], "data": [[...], ...] "viz_type": "table", "generated_query": ""}
```


## Running without giving your database details to Defog

If you want to store your database credentials locally and do not want to give us access, you can run the query like this

```
from defog import Defog

# your credentials are never sent to our server, and always run locally
defog = Defog(
    api_key="YOUR_API_KEY",
    db_type="postgres",
    db_creds={
        "host": YOUR_DB_HOST,
        "port": YOUR_PORT, # usually, this is 5432 for Postgres DBs
        "database": YOUR_DATABASE_NAME,
        "user": YOUR_USER_NAME, # often `defogdata`, if you have followed our setup instructions
        "password": YOUR_PASSWORD
    }
)

question = "question asked by a user"

results = defog.run_query(
  question=question
)
# {"is_success": True, "error_message": "", "col_names": [...], "data": [[...], ...] "viz_type": "table", "generated_query": ""}
```
