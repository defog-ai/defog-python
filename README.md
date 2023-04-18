# TL;DR
Defog converts your natural language text queries into SQL and other machine readable code

![](defog-python.gif)

# Installation
`pip install --upgrade defog`

# Getting your API Key
You can get your API key by going to [https://defog.ai/account](https://defog.ai/account) and creating an account.

# Usage

You can either use our cli, which will take you through the setup step-by-step, or pass it in explicitly in python to the `Defog` class. The CLI uses the python api's behind the hood, and is just an interactive wrapper over it that does some extra validation on your behalf.

## CLI

### Connection Setup
To get started, you can run the following cli command, which will prompt you for your defog api key, database type, and the corresponding database credentials required.
```
defog init
```
If this is your first time running, we will write these information into a json config file, which will be stored in `~/.defog/connection.json`. If we detect a file present already, we will ask you if you intend to re-initialize the file. You can always delete the file and `defog init` all over again. Note that your credentials are _never_ sent to defog's servers.

Once you have setup the connection settings, we will ask you for the names of the tables that you would like to register (space separated), generate the schema for each of them, upload the schema to defog, and print out the generate gsheets url in your console. 

### Regenerating Schema

If you would like to include new tables to be indexed, you can run the following to regenerate the schema for your tables:
```
defog gen <table1> <table2> ...
```
This will generate a gsheets url as before.

### Updating Schema

If you spot some mistakes, and have updated the schema generated in the gsheet, you can run the following to update the schema with defog:
```
defog update <url>
```

### Querying

You can now run queries directly:
```
defog query "<your query>"
```
Happy querying!

## Python API's

If you prefer to use the python API's to integrate the calls directly within your application, you may do so with the following examples provided.

## Postgres
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

# generate a schema of your selected postgres tables
# feel free to make changes to the google sheet url generated
gsheets_url = defog.generate_postgres_schema(tables=['your_table_name_1', 'your_table_name_2']) 

# update the postgres schema in our database
defog.update_postgres_schema(gsheets_url)

question = "question asked by a user"
results = defog.run_query(
  question=question
)

print(results)
```

## Mongo
```
from defog import Defog

# your credentials are never sent to our server, and always run locally
defog = Defog(
    api_key="YOUR_API_KEY",
    db_type="mongo",
    db_creds={
        "connection_string": YOUR_CONNECTION_STRING,
    }
)

# generate a schema of your selected mongo collections
# feel free to make changes to the google sheet url generated
gsheets_url = defog.generate_mongo_schema(collections=['collection_name_1', 'collection_name_2'])

# update the mongo schema in our database
defog.update_mongo_schema(gsheets_url)

question = "question asked by a user"
results = defog.run_query(
  question=question
)

print(results)
```

## MySQL
```
from defog import Defog

# your credentials are never sent to our server, and always run locally
defog = Defog(
    api_key="YOUR_API_KEY",
    db_type="mysql",
    db_creds={
        "host": YOUR_DB_HOST,
        "database": YOUR_DATABASE_NAME,
        "user": YOUR_USER_NAME, # often `defogdata`, if you have followed our setup instructions
        "password": YOUR_PASSWORD
    }
)

# generate a schema of your selected mysql tables
# feel free to make changes to the google sheet url generated
gsheets_url = defog.generate_mysql_schema(tables=['your_table_name_1', 'your_table_name_2']) 

# update the mysql schema in our database
defog.update_mysql_schema(gsheets_url)

question = "question asked by a user"
results = defog.run_query(
  question=question
)

print(results)
```


## BigQuery
```
from defog import Defog

# your credentials are never sent to our server, and always run locally
defog = Defog(
    api_key="YOUR_API_KEY",
    db_type="bigquery",
    db_creds={
      "json_key_path": "/path/to/service/json.key"
    },
)

# generate a schema of your selected Bigquery tables
# feel free to make changes to the google sheet url generated
gsheets_url = defog.generate_bigquery_schema(tables=['your_table_name_1', 'your_table_name_2'])

# update the postgres schema in our database
defog.update_biguery_schema(gsheets_url)

question = "question asked by a user"
results = defog.run_query(
  question=question
)

print(results)
```

## Redshift
```
from defog import Defog

# your credentials are never sent to our server, and always run locally
defog = Defog(
    api_key="YOUR_API_KEY",
    db_type="redshift",
    db_creds={
        "host": YOUR_DB_HOST,
        "port": YOUR_PORT, # usually, this is 5439 for Redshift
        "database": YOUR_DATABASE_NAME,
        "user": YOUR_USER_NAME, # often `defogdata`, if you have followed our setup instructions
        "password": YOUR_PASSWORD
    }
)

# generate a schema of your selected Redshift tables
# feel free to make changes to the google sheet url generated
gsheets_url = defog.generate_redshift_schema(tables=['your_table_name_1', 'your_table_name_2']) 

# update the redshift schema in our database
defog.update_redshift_schema(gsheets_url)

question = "question asked by a user"
previous_context = []
# previous_context is an array of previous questions asked and SQL generated
# an example is this: previous_context = ['who are our best users?', "SELECT u.userid, u.username, count(distinct s.eventid) as num_events, sum(s.qtysold) as total_tickets_sold\nFROM users u join sales s ON u.userid = s.buyerid\nGROUP BY u.userid, u.username\nORDER BY total_tickets_sold desc limit 10;"]

results = defog.run_query(
  question=question,
  previous_context=previous_context
)

print(results)
```

## Snowflake
```
from defog import Defog

# your credentials are never sent to our server, and always run locally
defog = Defog(
    api_key="YOUR_API_KEY",
    db_type="snowflake",
    db_creds={
        "user": YOUR_USER_NAME,
        "password": YOUR_PASSWORD,
        "account": YOUR_ACCOUNT_NAME,
        "warehouse": YOUR_WAREHOUSE_NAME
        
    }
)

# generate a schema of your selected Snowflake tables
# feel free to make changes to the google sheet url generated
tables_to_query = ['your_table_name_1', 'your_table_name_2']
gsheets_url = defog.generate_snowflake_schema(tables=tables_to_query)

# update the snowflake schema in our database
defog.update_snowflake_schema(gsheets_url)

question = "question asked by a user"
results = defog.run_query(
  question=question
)

print(results)
```

# Testing

For developers who want to test or add tests for this client, you can run:
```
python -m unittest test_defog
```

Note that we will transfer the existing .defog/connection.json file over to /tmp (if at all), and transfer the original file back once the tests are done to avoid messing with the original config.
