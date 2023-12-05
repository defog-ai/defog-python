# create a FastAPI app
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from defog import Defog

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
