from fastapi import FastAPI
from src.lib.chunker import Chunker
from src.__main__ import Cli


app = FastAPI()
cli = Cli()


@app.post("/max_chunk_size/{m}")
def index(m: int):
    chunker = Chunker(m)
    chunker.chunker()
    return {"message": "the data is indexed and saved in disk"}


@app.get("/answer/queyr/{q}")
def answear_query(query: str, k: int):
    response = cli.answear(query, k)
    return {"response": response}
