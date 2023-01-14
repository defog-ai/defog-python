# this script generates a json schema for selected tables in a postgres database

import json
import psycopg2

# connect to the database
conn = psycopg2.connect(
    host="YOUR_POSTGRES_HOST",
    port=5432,
    database="YOUR_POSTGRES_DB",
    user="YOUR_POSTGRES_USERNAME",
    password="YOUR_POSTGRES_PASSWORD"
)

# get the cursor
cur = conn.cursor()

# specity the tables you want to generate a schema for
tables = [ "YOUR_TABLE_NAME",  "YOUR_TABLE_NAME"]

schemas = {}
# get the schema for each table
for table_name in tables:
    cur.execute("SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s;", (table_name,))
    rows = cur.fetchall()
    rows = [row for row in rows]
    rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
    schemas[table_name] = rows

# write the schema to a json file
with open("schemas.json", "w") as f:
    json.dump(schemas, f, indent=4)