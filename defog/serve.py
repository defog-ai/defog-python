# create a FastAPI app
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from defog import Defog
import pandas as pd
import os
import json
from io import StringIO

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

home_dir = os.path.expanduser("~")
defog_path = os.path.join(home_dir, ".defog")


@app.get("/")
async def root():
    return {"message": "Hello, I am Defog"}


@app.post("/generate_query")
async def generate(request: Request):
    params = await request.json()
    question = params.get("question")
    previous_context = params.get("previous_context")
    defog = Defog()
    resp = defog.run_query(question, previous_context=previous_context)
    return resp


@app.post("/integration/get_tables_db_creds")
async def get_tables_db_creds(request: Request):
    try:
        defog = Defog()
    except:
        return {"error": "no defog instance found"}

    try:
        with open(os.path.join(defog_path, "tables.json"), "r") as f:
            table_names = json.load(f)
    except:
        table_names = []

    try:
        with open(os.path.join(defog_path, "selected_tables.json"), "r") as f:
            selected_table_names = json.load(f)
    except:
        selected_table_names = []

    db_type = defog.db_type
    db_creds = defog.db_creds
    api_key = defog.api_key

    return {
        "tables": table_names,
        "db_creds": db_creds,
        "db_type": db_type,
        "selected_tables": selected_table_names,
        "api_key": api_key,
    }


@app.post("/integration/get_metadata")
async def get_metadata(request: Request):
    try:
        with open(os.path.join(defog_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        return {"metadata": metadata}
    except:
        return {"error": "no metadata found"}


@app.post("/integration/generate_tables")
async def get_tables(request: Request):
    params = await request.json()
    api_key = params.get("api_key")
    db_type = params.get("db_type")
    db_creds = params.get("db_creds")
    for k in ["api_key", "db_type"]:
        if k in db_creds:
            del db_creds[k]

    defog = Defog(api_key, db_type, db_creds)
    table_names = defog.generate_db_schema(tables=[], return_tables_only=True)

    with open(os.path.join(defog_path, "tables.json"), "w") as f:
        json.dump(table_names, f)

    return {"tables": table_names}


@app.post("/integration/generate_metadata")
async def generate_metadata(request: Request):
    params = await request.json()
    tables = params.get("tables")

    with open(os.path.join(defog_path, "selected_tables.json"), "w") as f:
        json.dump(tables, f)

    defog = Defog()
    table_metadata = defog.generate_db_schema(
        tables=tables, scan=True, upload=True, return_format="csv_string"
    )
    metadata = (
        pd.read_csv(StringIO(table_metadata)).fillna("").to_dict(orient="records")
    )

    with open(os.path.join(defog_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    defog.update_db_schema(StringIO(table_metadata))
    return {"metadata": metadata}


@app.post("/integration/update_metadata")
async def update_metadata(request: Request):
    params = await request.json()
    metadata = params.get("metadata")
    defog = Defog()
    metadata = pd.DataFrame(metadata).to_csv(index=False)
    defog.update_db_schema(StringIO(metadata))
    return {"status": "success"}


@app.post("/instruct/get_glossary_golden_queries")
async def update_glossary(request: Request):
    defog = Defog()
    glossary = defog.get_glossary()
    golden_queries = defog.get_golden_queries(format="json")
    return {"glossary": glossary, "golden_queries": golden_queries}


@app.post("/instruct/update_glossary")
async def update_glossary(request: Request):
    params = await request.json()
    glossary = params.get("glossary")
    defog = Defog()
    defog.update_glossary(glossary=glossary)
    return {"status": "success"}


@app.post("/instruct/update_golden_queries")
async def update_golden_queries(request: Request):
    params = await request.json()
    golden_queries = params.get("golden_queries")
    golden_queries = [
        x
        for x in golden_queries
        if x["sql"] != "" and x["question"] != "" and x["user_validated"]
    ]
    defog = Defog()
    defog.update_golden_queries(golden_queries=golden_queries)
    return {"status": "success"}
