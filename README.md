# Defog Python
Defog converts your natural language text queries into SQL and other machine readable code. This library allows you to easily integrate Defog into your python application, and has a CLI to help you get started.

https://user-images.githubusercontent.com/4327467/236758074-042bc5d7-4452-46ce-bb26-e2da2a0223c6.mp4


# Installation
`pip install --upgrade defog`

# Getting your API Key
You can get your API key by going to [https://defog.ai/account](https://defog.ai/account) and creating an account.

# Integration

You can either use our cli, which will take you through the setup step-by-step, or pass it in explicitly in python to the `Defog` class. The CLI uses the python api's behind the hood, and is just an interactive wrapper over it that does some extra validation on your behalf.

## Connection Setup
To get started, you can run the following cli command, which will prompt you for your defog api key, database type, and the corresponding database credentials required.

If you are just checking the suitability of your database, just hit enter when prompted for your defog api key.

```
defog init
```
If this is your first time running, we will write these information into a json config file, which will be stored in `~/.defog/connection.json`. If we detect a file present already, we will ask you if you intend to re-initialize the file. You can always delete the file and `defog init` all over again. Note that your credentials are _never_ sent to defog's servers.

Once you have setup the connection settings, we will ask you for the names of the tables that you would like to register (space separated), generate the schema for each of them, upload the schema to defog, and print out the generate gsheets url in your console. If you do not wish to provide those at this point, you can exit this prompt by hitting `ctrl+c`

## Checking the suitability of your database
To check whether Defog will be suitable for your database, you can run the following:

```
defog check <table_name_1> <table_name_1> ...
```

Where the table names are just the names of the table that you would want defog to query

## Regenerating Schema

If you would like to include new tables to be indexed, you can run the following to regenerate the schema for your tables:
```
defog gen <table1> <table2> ...
```
This will generate a gsheets url as before.

## Updating Schema

If you spot some mistakes, and have updated the schema generated in the gsheet, you can run the following to update the schema with defog:
```
defog update <url>
```

## Querying

You can now run queries directly:
```
defog query "<your query>"
```
Happy querying!

## Quota

You can check your quota usage by running:
```
defog quota
```
Free-tier users have 100 queries per month, while premium users have unlimited queries.

# Usage

You can use the API from within Python

```
from defog import Defog

# your credentials are never sent to our server, and always run locally
defog = Defog() # your credentials will automatically be loaded after you install defog

question = "question asked by a user"

# run chat version of query
results = defog.run_query(
  question=question,
  mode="chat",
  language=None # replace with a string, like 'Japanese', or 'Traditional Chinese' if using a non English language
)

print(results)
```

# Testing

For developers who want to test or add tests for this client, you can run:
```
python -m unittest discover -s tests -p "test_*.py"
```

Note that we will transfer the existing .defog/connection.json file over to /tmp (if at all), and transfer the original file back once the tests are done to avoid messing with the original config.
