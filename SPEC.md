# Spec — PDF Q&A RAG pipeline

**Status:** approved, not yet implemented
**Source plan:** `C:\Users\Archna Jain\.claude\plans\synthetic-seeking-stroustrup.md`

---

## 1. Purpose

A minimal, readable Retrieval-Augmented Generation pipeline built on LangChain and LangGraph, using OpenAI models. The user points the tool at a PDF; the tool chunks and embeds it, then answers questions about it in a terminal loop. Answers are grounded in retrieved passages, cite page numbers, and support follow-up questions.

The primary audience is a **reader learning the RAG pattern**. Where a choice exists between a clever implementation and an obvious one, the spec mandates the obvious one.

## 2. Scope

**In scope**

- Single PDF per run, supplied as a command-line argument
- Text extraction, chunking, embedding into an in-memory vector store
- Multi-turn question answering with conversation memory
- Page-number citations for every answer
- Terminal REPL interface

**Out of scope** (explicitly not to be built)

- Persisting the index across runs
- Multiple PDFs in one session
- Streaming token output
- Reranking or hybrid search
- Any web or GUI interface
- OCR for scanned PDFs
- Automated tests

## 3. Environment and dependencies

| Item | Value |
|---|---|
| Python | 3.11.15 (existing `venv/` in project root) |
| Platform | Windows 11, PowerShell |
| LLM provider | OpenAI, via API key |

`requirements.txt` (already created):

```
langchain-core>=1.5,<2
langchain-openai>=1.4,<2
langchain-text-splitters>=1.1,<2
langgraph>=1.2,<2
pypdf>=6,<7
python-dotenv>=1,<2
```

**D-1.** The `langchain` umbrella package MUST NOT be added — nothing imports from it.

**D-2.** `langchain-community` MUST NOT be used. PDF loading uses `pypdf` directly, per current LangChain 1.x documentation. This avoids pulling `langchain-classic`, `sqlalchemy`, and `aiohttp` into the tree for a single loader.

**D-3.** The following import paths are verified correct for these versions and MUST be used verbatim:

```python
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
```

**D-4.** The checkpointer class is `InMemorySaver`. `MemorySaver` is the older name and MUST NOT be used.

## 4. File layout

```
AI_tool_ntoes/
├── main.py             # ingest + graph + CLI, all in one file    [TO BUILD]
├── requirements.txt                                               [DONE]
├── .env.example                                                   [DONE]
├── .gitignore                                                     [DONE]
├── SPEC.md             # this file                                [DONE]
├── README.md           # setup + usage, ~20 lines                 [TO BUILD]
└── venv/                                                          [EXISTS]
```

**L-1.** All pipeline code lives in a single `main.py`. It MUST NOT be split into modules.

**L-2.** `main.py` is organised as five commented sections, in this order: (1) imports and configuration, (2) ingestion, (3) graph, (4) CLI, (5) entrypoint guard.

**L-3.** Target size is roughly 180 lines including comments. This is guidance, not a hard limit.

## 5. Configuration

**C-1.** Tunables are module-level named constants at the top of `main.py`:

| Constant | Value | Meaning |
|---|---|---|
| `CHAT_MODEL` | `"gpt-4o-mini"` | Chat completion model |
| `EMBEDDING_MODEL` | `"text-embedding-3-small"` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per question |
| `HISTORY_LIMIT` | `6` | Messages of history passed to the LLM |

**C-2.** The API key is read from the `OPENAI_API_KEY` environment variable, loaded from a `.env` file via `python-dotenv`. The key MUST NOT appear in source, in committed files, or in any printed output.

## 6. Command-line interface

**CLI-1.** Invocation:

```
python main.py <pdf_path> [--model MODEL] [--k K]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `pdf_path` | yes | — | Path to a text-based PDF. Paths with spaces are quoted by the caller. |
| `--model` | no | `CHAT_MODEL` | Override the chat model |
| `--k` | no | `TOP_K` | Override chunks retrieved per question |

**CLI-2.** Startup sequence, in this order:

1. Load `.env`
2. Parse arguments
3. Validate the key and the PDF path (§9) — **before** any billable call
4. Build the vector store, printing progress
5. Build the graph
6. Enter the REPL

**CLI-3.** Ingestion prints a summary in the form `N pages → M chunks`, so the user can immediately tell whether extraction worked.

**CLI-4.** The REPL prompts for a question, and:
- blank input is skipped without an API call
- `exit` or `quit` (case-insensitive) terminates cleanly with exit code 0
- EOF (Ctrl-D / Ctrl-Z) and Ctrl-C terminate cleanly, without a traceback

**CLI-5.** Each answer is followed by a sources line listing deduplicated, ascending page numbers:

```
Sources: p.1, p.7, p.12
```

**CLI-6.** Errors raised during a question (e.g. an API failure) are printed as a readable message and the REPL continues. A transient failure MUST NOT discard the loaded index.

## 7. Ingestion

`build_vector_store(pdf_path) -> InMemoryVectorStore`

**I-1.** Read the PDF with `pypdf.PdfReader`. For each page, call `page.extract_text()`.

**I-2.** Build one `Document` per page whose extracted text is non-empty, with metadata:

```python
{"source": <filename>, "page": <1-based page number>}
```

Page numbers are 1-based because they are shown to humans.

**I-3.** If **every** page extracts empty, raise an error stating that the PDF appears to be scanned images and requires OCR, which is out of scope. Rationale: without this guard the run silently embeds nothing, every answer becomes "I don't know", and the cause is invisible to the user. This is the most likely first-run failure.

**I-4.** Split with `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)`. Page metadata propagates to every chunk.

**I-5.** Build the store with `InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model=EMBEDDING_MODEL))`.

**I-6.** The function returns the store and prints nothing. Progress reporting belongs to the CLI section.

## 8. The graph

### 8.1 State

**G-1.** State schema:

```python
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation
    search_query: str                        # history-aware standalone question
    context: list[Document]                  # chunks retrieved this turn
```

### 8.2 Construction

**G-2.** `build_graph(vector_store, llm)` returns a compiled graph. The node functions are defined **inside** `build_graph` so they close over `vector_store` and `llm` and take no arguments beyond state.

**G-3.** Topology is linear:

```
START → contextualize → retrieve → generate → END
```

**G-4.** Compiled with `builder.compile(checkpointer=InMemorySaver())`, so conversation history survives across separate `invoke` calls within one process.

### 8.3 Nodes

**G-5 — `contextualize`.** Produces `{"search_query": ...}`.

- If `messages` contains only the current question (no prior turns), pass the question through unchanged and make **no** LLM call.
- Otherwise, ask the LLM to rewrite the latest question into a standalone question given recent history. The prompt instructs it to rewrite only, not to answer.

Rationale: embedding the literal string `"why?"` retrieves nothing useful. This node is what makes multi-turn RAG work, and is the one behaviour a single-shot pipeline cannot reproduce.

**G-6 — `retrieve`.** `vector_store.similarity_search(state["search_query"], k=TOP_K)` → `{"context": docs}`.

**G-7 — `generate`.** Produces `{"messages": [AIMessage(answer)]}`; the `add_messages` reducer appends it to history.

- Retrieved chunks are formatted as `[p.N] <text>` blocks.
- Those blocks go into a `SystemMessage` carrying the grounding instruction: *"Answer using only the context below. If the answer is not in the context, say you don't know."*
- Recent chat history is appended after the system message.

**G-8 — History trimming.** Both `contextualize` and `generate` pass at most the last `HISTORY_LIMIT` messages to the LLM. Trimming happens **at read time only**; stored history is never mutated. Rationale: with a checkpointer, `messages` grows without bound, and per-question cost must stay flat.

### 8.4 Invocation

**G-9.** Per question:

```python
graph.invoke(
    {"messages": [HumanMessage(q)]},
    {"configurable": {"thread_id": "cli-session"}},
)
```

**G-10.** The `thread_id` is the fixed literal `"cli-session"` — one process, one conversation. This is the seam where multi-session support would later attach.

**G-11.** The answer is `result["messages"][-1].content`; the sources line is built from `result["context"]`.

## 9. Error handling

Every case below MUST produce a readable message, never a raw traceback.

| Condition | When detected | Behaviour |
|---|---|---|
| `OPENAI_API_KEY` unset or empty | Startup, before ingestion | Exit with a message pointing at `.env.example` |
| PDF path does not exist | Startup, before ingestion | Exit with the offending path in the message |
| Path is not a `.pdf` | Startup, before ingestion | Exit with a message naming the expected extension |
| PDF is unreadable / corrupt | During ingestion | Exit with the underlying reason |
| PDF yields no extractable text | During ingestion (I-3) | Exit; state that OCR is needed and is out of scope |
| API error on a question | During REPL | Print the error, keep the REPL and the index alive |
| Blank question | During REPL | Re-prompt; no API call |

The first three checks run **before** embedding, so a typo never costs money.

## 10. Non-functional requirements

**N-1 — Cost.** Embedding is the only bulk cost. With `text-embedding-3-small`, a 50-page PDF costs well under one cent. The README states this so the user is not surprised.

**N-2 — Import safety.** No work executes at module scope. All behaviour sits behind `main()`, called under `if __name__ == "__main__":`. `import main` MUST be side-effect free, so the smoke test in §11 works without a key or network.

**N-3 — Legibility.** The file reads top to bottom. Comments explain *why*, not *what*. Section boundaries are the seams for the out-of-scope features in §2.

**N-4 — Secrets.** `.env` is git-ignored. The key is never printed, logged, or echoed.

## 11. Acceptance criteria

Run from the project root with the venv active (`.\venv\Scripts\Activate.ps1`).

**A-1 — Install.** `pip install -r requirements.txt` resolves without conflicts on Python 3.11.

**A-2 — Import smoke test.** Requires no key and no network:

```
python -c "import main; print('imports ok')"
```

Passing proves the import paths of §3 are correct and that N-2 holds.

**A-3 — Guards (free, no API calls).** Running with a nonexistent path, and running with `OPENAI_API_KEY` unset, each fail with a readable message and no traceback.

**A-4 — Ingestion.** With a real key and any text-based PDF, `python main.py "path\to\some.pdf"` prints a plausible page → chunk count.

**A-5 — Grounded answer with correct citations.** A factual question is answered correctly, and the pages named on the `Sources:` line genuinely contain the answer when the PDF is opened and checked. This is the real test of retrieval quality — matching page numbers are not enough on their own.

**A-6 — Follow-up resolution.** Ask a specific question, then a bare follow-up (`why?` or `say more`). The answer is relevant to the prior turn. This is the acceptance test for G-5.

**A-7 — Refusal to invent.** A question clearly unrelated to the PDF returns "I don't know" rather than a confident fabrication. This is the acceptance test for the grounding prompt in G-7.

**A-8 — Clean exit.** `exit` terminates without a traceback.

## 12. Open items

None. All design decisions are settled; §2 records what was deliberately deferred.
