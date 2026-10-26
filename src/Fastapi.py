from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from src.lib.chunker import Chunker
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lib import models
app = FastAPI()
from src.__main__ import Cli

cli = Cli()



@app.get("/max_chunk_size/{m}")
def index(max_chunk_size: int):
    chunker = Chunker(max_chunk_size)
    chunks = chunker.chunker()
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed")
    return {"chunks": chunks}


@app.get("/answer/queyr/{q}")
def answear_query(query: str, k: int):
    response = cli.answear(query, k)
    return {"response": response}