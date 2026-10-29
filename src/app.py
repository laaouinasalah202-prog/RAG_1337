from fastapi import FastAPI
from src.lib.chunker import Chunker
from src.__main__ import Cli


app = FastAPI()
cli = Cli()


@app.post("/max_chunk_size/{m}")
def index(m: int) -> dict:
    """Chunk the documents and save the index to disk.

    Args:
        m (int): Maximum size of a chunk in characters.

    Returns:
        dict: A message confirming that the data was indexed.
    """

    chunker = Chunker(m)
    chunker.chunker(m)
    return {"message": "the data is indexed and saved "
            "under data/processed/index.json"}


@app.get("/answer/queyr/{query}")
def answear_query(query: str, k: int) -> dict:
    """Answer a question using the retrieved chunks as context.

    Args:
        query (str): The question to answer.
        k (int): Number of chunks to retrieve as context.

    Returns:
        dict: The generated answer, under the key ``response``.
    """

    response = cli.answer(query, k)
    return {"response": response}
