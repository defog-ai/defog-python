# create a FastAPI app
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from defog import Defog
import psycopg2
import pandas as pd
import requests
import os
import json
import sqlparse
from tqdm import tqdm
import sys

try:
    from llama_cpp import Llama

    home_dir = os.path.expanduser("~")
    filepath = os.path.join(home_dir, ".defog", "sqlcoder-7b-q4_k_m.gguf")

    if not os.path.exists(filepath):
        print(
            "Downloading the SQLCoder-7b GGUF file. This is a 4GB file and may take up to 10 minutes to download..."
        )

        # download the gguf file from the internet and save it
        url = "https://storage.googleapis.com/defog-ai/sqlcoder-7b/v2/sqlcoder-7b-q4_k_m.gguf"
        response = requests.get(url, stream=True)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024)

        with open(filepath, "wb") as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)

        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong while downloading the file")

    llm = Llama(model_path=filepath, n_gpu_layers=1, n_ctx=2048)
except Exception as e:
    print("An error occured when trying to load the model!")
    sys.exit(1)


app = FastAPI()
defog = Defog()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def generate(request: Request):
    params = await request.json()
    question = params.get("question")
    previous_context = params.get("previous_context")
    resp = defog.run_query(question, previous_context=previous_context)
    return resp


@app.post("/get_tables")
async def get_tables(request: Request):
    params = await request.json()
    db_host = params.get("host")
    username = params.get("username")
    password = params.get("password")
    port = params.get("port")
    database = params.get("database")

    conn = psycopg2.connect(
        host=db_host,
        dbname=database,
        user=username,
        password=password,
        port=port,
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    tables = [row[0] for row in cur.fetchall()]
    return {"tables": tables}


@app.post("/get_metadata")
async def get_metadata(request: Request):
    params = await request.json()
    db_host = params.get("host")
    username = params.get("username")
    password = params.get("password")
    port = params.get("port")
    database = params.get("database")
    tables = params.get("tables")

    conn = psycopg2.connect(
        host=db_host,
        dbname=database,
        user=username,
        password=password,
        port=port,
    )
    cur = conn.cursor()
    print("Getting schema for each table in your database...")

    schema = []
    # get the schema for each table
    for table_name in tables:
        cur.execute(
            "SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s;",
            (table_name,),
        )
        rows = cur.fetchall()
        rows = [row for row in rows]
        rows = [
            {"table_name": table_name, "column_name": i[0], "data_type": i[1]}
            for i in rows
        ]
        schema += rows

    schema = pd.DataFrame(schema)
    schema["column_description"] = ""

    # save the credentials to a file in ~/.defog/
    home_dir = os.path.expanduser("~")
    filepath = os.path.join(home_dir, ".defog", "gguf_credentials.json")
    with open(filepath, "w") as f:
        creds = {
            "host": db_host,
            "database": database,
            "user": username,
            "password": password,
            "port": port,
        }
        json.dump(creds, f)

    return {"schema": schema.to_dict(orient="records")}


@app.post("/make_gguf_request")
async def make_gguf_request(request: Request):
    params = await request.json()
    prompt = params.get("prompt")
    completion = llm(
        prompt,
        max_tokens=100,
        temperature=0,
        top_p=1,
        stop=["\n"],
        echo=False,
        repeat_penalty=1.0,
    )
    completion = completion["choices"][0]["text"]
    return {"completion": completion}


@app.post("/update_metadata")
async def update_metadata(request: Request):
    params = await request.json()
    metadata = params.get("metadata")
    allowed_joins = params.get("allowed_joins")
    # this in a list where each item is a dictionary
    # with the format {"table_name": ..., "column_name": ..., "data_type": ..., "column_description": ...}

    # let's convert this into DDL statements
    # first we need to get the tables
    tables = list(set([i["table_name"] for i in metadata]))
    table_ddl = []
    for table in tables:
        table_ddl.append(f"CREATE TABLE {table} (\n")
        for column in metadata:
            if column["table_name"] == table:
                if column["column_description"]:
                    desc = f"-- {column['column_description']}"
                else:
                    desc = ""
                table_ddl.append(
                    f"{column['column_name']} {column['data_type']}{desc},\n"
                )
        table_ddl.append(");\n\n")
    table_ddl = "".join(table_ddl)

    if allowed_joins is None or allowed_joins == "":
        prompt = f"""# Task
Your task is to identify all valid joins between tables in a Postgres Database. Give your answers in the format
-- table1.column1 can be joined with table2.column2
-- table1.column3 can be joined with table2.column4
etc.

# Database Schema
The database has the following schema:
{table_ddl}

# Allowed Joins
Based on the database schema, the following joins are valid:
--"""
        completion = llm(
            prompt,
            max_tokens=500,
            temperature=0,
            top_p=1,
            echo=False,
            repeat_penalty=1.0,
        )
        completion = completion["choices"][0]["text"]
        allowed_joins = completion
        table_ddl += "\n" + completion
    else:
        table_ddl += "\n" + allowed_joins

    home_dir = os.path.expanduser("~")
    filepath = os.path.join(home_dir, ".defog", "metadata.sql")
    # save the DDL statements to a file in ~/.defog/
    with open(filepath, "w") as f:
        f.write(table_ddl)

    return {
        "success": True,
        "message": "Metadata updated successfully!",
        "suggested_joins": allowed_joins,
    }


@app.post("/query_db")
async def query_db(request: Request):
    params = await request.json()
    home_dir = os.path.expanduser("~")
    filepath = os.path.join(home_dir, ".defog", "metadata.sql")
    with open(filepath, "r") as f:
        ddl = f.read()

    user_question = params.get("question")
    prompt = f"""# Task
Generate a SQL query to answer the following question:
`{user_question}`

# Database Schema
The query will run on a database with the following schema:
{ddl}

# Instructions
- only use the tables and columns in the schema above
- you can use CTEs to create temporary tables that can be used in your query

# SQL
```"""
    completion = llm(
        prompt,
        max_tokens=100,
        temperature=0,
        repeat_penalty=1.0,
        echo=False,
    )
    completion = completion["choices"][0]["text"]
    completion = completion.split("```")[0].split(";")[0].strip()
    completion = completion + ";"
    print(completion)
    # now we have the SQL query, let's run it on the database

    # first we need to get the credentials
    filepath = os.path.join(home_dir, ".defog", "gguf_credentials.json")
    with open(filepath, "r") as f:
        creds = json.load(f)

    conn = psycopg2.connect(
        host=creds["host"],
        database=creds["database"],
        user=creds["user"],
        password=creds["password"],
        port=creds["port"],
    )
    cur = conn.cursor()
    try:
        cur.execute(completion)
        rows = cur.fetchall()
        rows = [row for row in rows]
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
    except Exception as e:
        error_message = str(e)
        prompt = f"""# Task
Fix SQL queries that have errors.

# Database Schema
The query will run on a database with the following schema:
{ddl}

# Original SQL
```{completion}```

# Error Message
{error_message}

# New SQL
```"""
        completion = llm(
            prompt,
            max_tokens=100,
            temperature=0,
            repeat_penalty=1.0,
            echo=False,
        )
        completion = completion["choices"][0]["text"]
        completion = completion.split("```")[0].split(";")[0].strip()
        completion = completion + ";"
        print(completion)
        conn = psycopg2.connect(
            host=creds["host"],
            database=creds["database"],
            user=creds["user"],
            password=creds["password"],
            port=creds["port"],
        )
        cur = conn.cursor()
        cur.execute(completion)
        rows = cur.fetchall()
        rows = [row for row in rows]
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

    completion = sqlparse.format(completion, reindent_aligned=True)
    return {
        "columns": columns,
        "data": rows,
        "query_generated": completion,
        "ran_successfully": True,
    }
