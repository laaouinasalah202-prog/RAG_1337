from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from src.lib.chunker import Chunker
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lib import models
app = FastAPI()
from src.__main__ import Cli

cli = Cli()



@app.post("/max_chunk_size/{m}")
def index(m: int):
    chunker = Chunker(m)
    chunks = chunker.chunker()
    # print(f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed")
    return {"message": "the data is indexed and saved in disk"}


@app.get("/answer/queyr/{q}")
def answear_query(query: str, k: int):
    response = cli.answear(query, k)
    return {"response": response}