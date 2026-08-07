"""Ask questions about a single PDF: chunk it, embed it, answer from what was retrieved.

Reads top to bottom in five sections: configuration, ingestion, graph, CLI, entrypoint.
Nothing runs at import time, so `import main` is safe without a key or a network.
"""

# ---------------------------------------------------------------------------
# 1. Imports and configuration
# ---------------------------------------------------------------------------

import argparse
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pypdf
from dotenv import load_dotenv
from pypdf.errors import PyPdfError
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
HISTORY_LIMIT = 6  # messages handed to the LLM, so per-question cost stays flat

# One process, one conversation. This literal is the seam multi-session support attaches to.
THREAD_ID = "cli-session"

GROUNDING_PROMPT = (
    "You are answering questions about a PDF. Answer using only the context below. "
    "If the answer is not in the context, say you don't know. "
    "Cite the pages you used inline, like (p.3).\n\n"
    "Context:\n{context}"
)

CONTEXTUALIZE_PROMPT = (
    "Given the conversation so far, rewrite the user's latest message as a standalone "
    "question that makes sense without the conversation. Do not answer it. "
    "If it already stands alone, return it unchanged. Return only the question."
)


# ---------------------------------------------------------------------------
# 2. Ingestion
# ---------------------------------------------------------------------------


def build_vector_store(pdf_path: Path) -> tuple[InMemoryVectorStore, int, int]:
    """Extract, chunk and embed the PDF.

    Returns the store plus the page and chunk counts, which the CLI reports. This
    function prints nothing itself.
    """
    reader = pypdf.PdfReader(str(pdf_path))

    # One Document per page, so the page number rides along into every chunk's
    # metadata. Numbers are 1-based because they get shown to a human.
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": number},
                )
            )

    # Without this guard the run embeds nothing, every answer becomes "I don't know",
    # and the cause is invisible. It is also the likeliest first-run failure.
    if not pages:
        raise ValueError(
            f"No extractable text in {pdf_path.name} - it is probably a scan of images. "
            "Reading that needs OCR, which is out of scope for this tool."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    store = InMemoryVectorStore.from_documents(
        chunks, OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )
    return store, len(pages), len(chunks)


# ---------------------------------------------------------------------------
# 3. The graph
# ---------------------------------------------------------------------------


class RAGState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation, kept by the checkpointer
    search_query: str  # history-aware standalone question
    context: list[Document]  # chunks retrieved this turn


def build_graph(vector_store: InMemoryVectorStore, llm: ChatOpenAI, top_k: int):
    """Compile START → contextualize → retrieve → generate → END.

    The nodes are defined in here so they close over the store and the model, and
    so each one takes nothing but state.
    """

    def contextualize(state: RAGState) -> dict:
        messages = state["messages"]
        # First turn: the question already stands alone, so skip the LLM call.
        if len(messages) <= 1:
            return {"search_query": messages[-1].content}

        # Embedding the literal string "why?" retrieves nothing useful, so a
        # follow-up gets rewritten into a question that can stand on its own.
        # Trimming happens here at read time; stored history is never mutated.
        rewritten = llm.invoke(
            [SystemMessage(CONTEXTUALIZE_PROMPT), *messages[-HISTORY_LIMIT:]]
        )
        return {"search_query": rewritten.content.strip()}

    def retrieve(state: RAGState) -> dict:
        return {"context": vector_store.similarity_search(state["search_query"], k=top_k)}

    def generate(state: RAGState) -> dict:
        blocks = "\n\n".join(
            f"[p.{doc.metadata['page']}] {doc.page_content}" for doc in state["context"]
        )
        answer = llm.invoke(
            [
                SystemMessage(GROUNDING_PROMPT.format(context=blocks)),
                *state["messages"][-HISTORY_LIMIT:],
            ]
        )
        # The add_messages reducer appends this to the stored conversation.
        return {"messages": [AIMessage(answer.content)]}

    builder = StateGraph(RAGState)
    builder.add_node("contextualize", contextualize)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # The checkpointer is what lets history survive across separate invoke() calls.
    return builder.compile(checkpointer=InMemorySaver())


def format_sources(docs: list[Document]) -> str:
    pages = sorted({doc.metadata["page"] for doc in docs})
    return "Sources: " + ", ".join(f"p.{page}" for page in pages)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="path to a text-based PDF (default: the one PDF sitting beside main.py)",
    )
    parser.add_argument("--model", default=CHAT_MODEL, help=f"chat model (default: {CHAT_MODEL})")
    parser.add_argument(
        "--k", type=int, default=TOP_K, help=f"chunks retrieved per question (default: {TOP_K})"
    )
    return parser.parse_args()


def resolve_pdf(pdf_path: str | None) -> Path:
    """An explicit path wins; otherwise fall back to a lone PDF next to this script."""
    if pdf_path:
        return Path(pdf_path)

    folder = Path(__file__).parent
    candidates = sorted(folder.glob("*.pdf"))
    if not candidates:
        raise SystemExit(f"No PDF given, and none found in {folder}. Add one, or pass a path.")
    # Guessing between several would be a coin flip, and the wrong guess costs money.
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise SystemExit(f"Several PDFs in {folder}: {names}. Name the one you want.")
    return candidates[0]


def validate(args: argparse.Namespace) -> Path:
    """Everything that is free to check, checked before any billable call."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    pdf_path = resolve_pdf(args.pdf_path)
    if not pdf_path.is_file():
        raise SystemExit(f"No such file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Expected a .pdf file, got: {pdf_path.name}")
    return pdf_path


def repl(graph) -> None:
    print("Ask a question, or type 'exit' to quit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            continue  # no API call for an empty line
        if question.lower() in {"exit", "quit"}:
            return

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(question)]},
                {"configurable": {"thread_id": THREAD_ID}},
            )
        except KeyboardInterrupt:
            print()
            return
        except Exception as error:
            # A failed question must not cost the user the index they paid to build.
            print(f"\nThat question failed: {error}\n")
            continue

        print(f"\n{result['messages'][-1].content}")
        print(f"{format_sources(result['context'])}\n")


def main() -> None:
    # Piped output on Windows defaults to cp1252, which cannot encode the degree
    # signs and Greek letters a technical PDF will put in its answers. Without this
    # the REPL dies mid-answer, after the document has already been paid for.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()
    args = parse_args()
    pdf_path = validate(args)

    print(f"Reading {pdf_path.name} ...")
    try:
        vector_store, page_count, chunk_count = build_vector_store(pdf_path)
    except ValueError as error:
        raise SystemExit(str(error))  # the no-text guard, already phrased for a human
    except PyPdfError as error:
        raise SystemExit(f"Could not read {pdf_path.name}: {error}")
    except Exception as error:
        # The file parsed, so anything left is the embedding call — key, network, quota.
        # Saying "could not read" here would send the user hunting through their PDF.
        raise SystemExit(f"Could not embed {pdf_path.name}: {error}")

    # ASCII arrow on purpose: this line must survive a cp1252 console even if the
    # reconfigure above could not be applied.
    print(f"{page_count} pages -> {chunk_count} chunks\n")

    graph = build_graph(vector_store, ChatOpenAI(model=args.model), args.k)
    repl(graph)


# ---------------------------------------------------------------------------
# 5. Entrypoint
# ---------------------------------------------------------------------------

# SystemExit raised with a string prints it to stderr and exits 1 — no traceback,
# which is what §9 asks for, so main() is called plainly here.
if __name__ == "__main__":
    main()
