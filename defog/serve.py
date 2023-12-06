# create a FastAPI app
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from defog import Defog
import psycopg2
import pandas as pd
import requests
import os
import json

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
        database=database,
        user=username,
        password=password,
        port=port,
    )
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
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
        database=database,
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
        rows = [{"table_name": table_name, "column_name": i[0], "data_type": i[1]} for i in rows]
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
    r = requests.post("http://localhost:8081/v1/completions", json={"prompt": prompt, "max_tokens": 100, "temperature": 0, "top_p": 1, "n": 1, "stop": ["\n"]})
    completion = r.json()["choices"][0]["text"]
    return {"completion": completion}

@app.post("/update_metadata")
async def update_metadata(request: Request):
    params = await request.json()
    metadata = params.get("metadata")
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
                table_ddl.append(f"{column['column_name']} {column['data_type']} -- {column['column_description']},\n")
        table_ddl.append(");\n\n")
    table_ddl = "".join(table_ddl)
    home_dir = os.path.expanduser("~")
    filepath = os.path.join(home_dir, ".defog", "metadata.sql")
    # save the DDL statements to a file in ~/.defog/
    with open(filepath, "w") as f:
        f.write(table_ddl)
    
    return {"success": True, "message": "Metadata updated successfully!"}

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

# SQL
```"""
    r = requests.post("http://localhost:8081/v1/completions", json={"prompt": prompt, "max_tokens": 600, "temperature": 0, "n": 1, "stop": [";", "```"]})
    completion = r.json()["choices"][0]["text"]
    completion = completion.split("```")[0].split(";")[0].strip()
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
    cur.execute(completion)
    rows = cur.fetchall()
    rows = [row for row in rows]
    columns = [desc[0] for desc in cur.description]
    return {
        "columns": columns,
        "data": rows,
        "query_generated": completion,
        "ran_successfully": True,
    }