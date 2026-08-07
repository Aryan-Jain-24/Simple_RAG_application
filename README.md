# PDF Q&A

Ask questions about a PDF from the terminal. The tool chunks and embeds the file,
retrieves the passages most relevant to each question, and answers from those
passages only — citing the pages it used. Follow-up questions work: `why?` is
rewritten into a standalone question before retrieval.

## Setup

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env    # then paste your OpenAI key into .env
```

## Use

```powershell
python main.py                                   # uses the PDF sitting beside main.py
python main.py "path\to\some.pdf"
python main.py "path\to\some.pdf" --model gpt-4o --k 8
```

Drop a PDF in this folder and `python main.py` will find it, so long as there is
exactly one. With several, name the one you want. PDFs are git-ignored.

Type a question at the prompt; `exit` or `quit` ends the session.

`--model` overrides the chat model, `--k` the number of chunks retrieved per
question. Both default to the constants at the top of `main.py`.

## Cost

Embedding the PDF is the only bulk cost, and it happens once per run — the index
is not saved between runs. With `text-embedding-3-small` a 50-page PDF costs well
under a cent. Each question then costs a couple of `gpt-4o-mini` calls.

## Limits

Text-based PDFs only. A scanned PDF has no extractable text and the tool will say
so rather than answering from nothing; OCR is out of scope. One PDF per run.
