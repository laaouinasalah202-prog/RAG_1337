from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from src.lib.chunker import Chunker

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


@app.get("/max_chunk_size/{m}")
def index(max_chunk_size: int):
    chunker = Chunker(max_chunk_size)
    chunks = chunker.chunker()
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed")
    return chunks
