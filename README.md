*This project has been created as part of the 42 curriculum by slaaouin.*

# RAG Against the Machine

## Description

**RAG against the machine** is a Retrieval-Augmented Generation (RAG) system that answers questions about a real-world codebase: the vLLM folder provided with the project.

A language model only knows what it saw during training. Instead of retraining it, a RAG system lets the model look up the right information at answer time. The pipeline has four stages:

1. **Indexing**: split every file of the vLLM repository into chunks and save a searchable index.
2. **Retrieving**: given a question, find the most relevant chunks (file path and character range).
3. **Augmenting**: place the best snippets in the context window of the model.
4. **Generating**: let a small local model (`Qwen/Qwen3-0.6B`) write an answer based on those snippets.

Retrieval quality is measured with **recall@k**. The targets are at least 80% recall@5 on docs questions and 50% on code questions, with indexing in under 5 minutes and 200 questions searched in under 90 seconds.

### Features

- Two chunking strategies (Python code and Markdown/text) with a configurable chunk size
- Custom BM25 implemented from scratch (keyword search)
- Semantic search with `all-MiniLM-L6-v2` embeddings
- Hybrid retrieval that combines both rankings
- Answer generation with `Qwen/Qwen3-0.6B`
- Interactive CLI built with Fire and `tqdm` progress bars
- A simple HTTP API built with FastAPI
- Pydantic models for all data exchanged between stages

---

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) as project and package manager
- Several GB of free disk space (PyTorch and model weights)

### Installation

```bash
git clone <your-repo-url>
cd <your-repo>
make install
```

Place the vLLM repository in `data/raw/` at the root of the project, so that file paths look like `data/raw/vllm-0.10.1/...`. The grader compares paths verbatim, so do not rename or move them.

### Makefile rules

- `make install`: install the dependencies with `uv sync`
- `make run`: run the main script
- `make debug`: run the main script with `pdb`
- `make clean`: remove `__pycache__`, `.mypy_cache` and other temporary files
- `make lint`: run `flake8` and `mypy` with the required flags

### CLI commands

Every command is run as `uv run python -m src <command> [options]`.

**index**: builds the index from `data/raw/` and saves it under `data/processed/`.

```bash
uv run python -m src index --max_chunk_size 2000
```

The overlap between chunks is `max_chunk_size / 20` (5%). The default chunk size is 2000 characters, which is also the maximum accepted by the grader.

**search**: returns the top-k sources for one query.

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

**search_dataset**: runs the search over a whole dataset and writes a `StudentSearchResults` JSON file.

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

**answer**: retrieves the context for one query and answers it with Qwen3-0.6B.

```bash
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

**answer_dataset**: generates answers for a whole dataset and writes a `StudentSearchResultsAndAnswer` JSON file.

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

**evaluate**: reports recall@k against a ground-truth dataset, for personal testing.

```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

All input and output paths are CLI arguments and nothing is hard-coded. Bad inputs (empty query, `k=0`, missing file, malformed JSON) are handled with clear error messages instead of crashes.

### HTTP API

A small FastAPI server lets you search and answer questions without the CLI.

```bash
uv run fastapi dev
```

The interactive documentation is then available at `http://127.0.0.1:8000/docs`.

---

## System Architecture

The pipeline has two phases that share the same index.

**Indexing phase.** The loader reads the files of `data/raw/`. The chunker splits each file (one strategy for Python, one for Markdown and text). Each chunk is then stored with its text, its position in the file, and its embedding in `data/processed/index.json`. The BM25 statistics are built from the same chunks.

**Query phase.** A question is scored by BM25 and by semantic search. The two results are combined into a single ranking and the top-k chunks are returned. For the `answer` commands, those chunks are put in a prompt and sent to Qwen3-0.6B, which writes the final answer.

All data moving between these steps is validated with the Pydantic models of the subject (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`).

---

## Chunking Strategy

Code and prose do not break apart the same way, so two strategies are used, both based on LangChain's `RecursiveCharacterTextSplitter`:

- **Python files** are split on code structure first (classes and functions), then on blank lines and lines, so a chunk usually holds a complete piece of code.
- **Markdown and text files** are split on headings first, then on paragraphs and lines, so a chunk usually holds a complete section.

The maximum chunk size is set with `--max_chunk_size` (default 2000 characters) and is never exceeded, because the grader rejects any source longer than 2000 characters. The overlap is 5% of the chunk size, so information at a boundary is not lost. Each chunk keeps its `file_path`, `first_character_index` and `last_character_index`, which is exactly what the output format needs.

---

## Retrieval Method

**Keyword search (BM25).** BM25 is implemented from scratch. It ranks chunks by how often the query terms appear in them, giving more weight to rare terms and correcting for chunk length. so questions that quote a name from the code still match.

**Semantic search.** Chunks and queries are embedded with `all-MiniLM-L6-v2` from sentence-transformers, a lightweight model that runs on CPU. Chunks are ranked by cosine similarity to the query. This finds answers when the question paraphrases an idea instead of using the exact words.

**Hybrid ranking.** First, BM25 and semantic search are both run on the query. Their scores are normalized so they can be compared, and weighted with alpha = 0.7. The two rankings are then merged with Reciprocal Rank Fusion (RRF) into one final list, and the top-k chunks are returned.

---

## Performance Analysis

For each question, **recall@k** is the share of its correct sources that appear in the top-k results. A source counts as found when a result comes from the same file and its character range overlaps the reference (IoU of at least 0.05):

recall@k = correct sources found in top-k / total correct sources

I use recall instead of precision because the goal is to make sure the right evidence reaches the model. A missed source cannot be recovered later, while an extra irrelevant chunk only adds some noise. Each question also has very few correct sources, so precision would be limited by design when k is large.

Recall is computed for k = 1, 3, 5 and 10.

**Results**:

- Docs: recall@1 = 59.0 %, recall@3 = 84.0%, recall@5 = 86.0%, recall@10 = 93.0%
- Code: recall@1 = 32.3 %, recall@3 = 48.5%, recall@5 = 58.6%, recall@10 = 62.6%

**Speed**:

- Indexing the whole corpus: 260 seconds (limit: 5 minutes)
- Searching 200 questions: 48 seconds (limit: 90 seconds)

My first BM25 version only reached 76% recall@5 because the tokenization did not suit the data. After rewriting the tokenizer regex more than five times, recall went above the target. On a codebase, the tokenizer matters more than any other BM25 setting.

---

## Design Decisions

- **Hybrid retrieval**: BM25 is strong on exact identifiers (common in code questions) and semantic search is strong on paraphrased questions (common in docs questions). Using both covers each case.
- **BM25 from scratch**: full control over tokenization and speed, and a better understanding of the ranking.
- **Two chunking strategies**: respecting the structure of code and documents keeps chunks self-contained.
- **Embeddings computed at indexing time**: at query time only the question has to be embedded, so batch search stays fast.
- **Small embedding model**: `all-MiniLM-L6-v2` runs on CPU with little memory.
- **Pydantic models everywhere**: malformed data fails early and clearly.
- **Results scoped by dataset**: the public datasets share file names, so each dataset has its own output folder to avoid overwriting.

---

## Challenges Faced

1. **Project structure and Pydantic models**: organizing the code so that every stage uses the models required by the subject.
2. **Speed and accuracy of a from-scratch BM25**: it had to be fast enough for 200 questions in 90 seconds. Tokenization was the biggest problem: I first got 76% recall@5, and only after changing the regex more than five times did it fit the vocabulary of the code and docs.
3. **Prompting Qwen3-0.6B**: the model is small and tends to answer even when the context does not contain the answer. I had to write the prompt so that it says it cannot answer when the retrieved context does not match the question.

---

## Example Usage

```bash
# Install dependencies
make install

# Build the index
uv run python3 -m src index --max_chunk_size 2000

# Search a single query
uv run python -m src search "How to configure the OpenAI server?" --k 5

# Search a whole dataset
 uv run python -m src search_dataset --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_result
s/UnansweredQuestions



# Evaluate your own recall@k
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# Generate answers for a whole dataset
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_private.json --save_directory data/outpu
t/search_results_and_answer/UnansweredQuestions
```

---

## Resources

- [Sentence-Transformers documentation](https://www.sbert.net/) and [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [Pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [uv](https://docs.astral.sh/uv/), [FastAPI](https://fastapi.tiangolo.com/)

### Use of AI

AI assistants were used for:

- Explaining RAG concepts and the difference between retrieval metrics
- Brainstorming test cases and edge cases for the CLI

All the code was written, reviewed, and tested by me, and I can explain every part of it.