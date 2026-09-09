# rag-document-intelligence-v2
# RAG Document Intelligence System

An end-to-end Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask natural-language questions about them. Built with an explicit, agentic LangGraph pipeline, automated LLM-based answer evaluation, and containerized for cloud deployment.

## Features

- **PDF upload & semantic search** — upload one or more PDFs, extract and chunk text, and retrieve relevant passages using FAISS vector search with `all-MiniLM-L6-v2` embeddings.
- **LLM-powered answering** — Groq-hosted LLMs generate context-grounded answers, with query rewriting to improve retrieval on vague or reference-heavy questions.
- **Agentic pipeline (LangGraph)** — the RAG flow is built as an explicit state graph (rewrite → retrieve → generate → evaluate) with a conditional edge that automatically widens retrieval and retries if the initial answer scores low on faithfulness.
- **Automated answer evaluation** — every answer is scored for faithfulness (is it grounded in the retrieved context?) and relevancy (does it address the question?) using an LLM-as-judge approach, displayed live in the UI.
- **Input validation & guardrails** — handles empty/invalid queries, caps file count and size per session, and gracefully skips PDFs with no extractable text (e.g. scanned images).
- **Structured logging** — key pipeline events (generation latency, evaluation scores, retry triggers) are logged as structured JSON for observability.
- **Containerized** — packaged with Docker for portable, reproducible deployment; includes an AWS ECS/Fargate task definition for cloud deployment.

## Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq API (configurable model, defaults to `openai/gpt-oss-120b`)
- **Orchestration:** LangGraph (agentic state graph with conditional retry logic)
- **Retrieval:** LangChain, FAISS, HuggingFace Sentence Transformers
- **PDF parsing:** PyMuPDF (fitz)
- **Containerization:** Docker
- **Cloud-ready:** AWS ECS/Fargate task definition included

## Running Locally

1. Clone the repo:
```bash
git clone https://github.com/<your-username>/rag-document-intelligence-v2.git
cd rag-document-intelligence-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Groq API key:
```bash
export GROQ_API_KEY=your_key_here   # macOS/Linux
set GROQ_API_KEY=your_key_here      # Windows cmd
```

4. Run the app:
```bash
streamlit run app.py
```

## Running with Docker

```bash
docker build -t rag-pdf-app .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key_here rag-pdf-app
```

Then open `http://localhost:8501`.

## Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key (required) | — |
| `GROQ_MODEL` | Groq model to use for generation & evaluation | `openai/gpt-oss-120b` |

## Evaluation Framework

Each answer is automatically scored on two dimensions using an LLM-as-judge approach:
- **Faithfulness (1–5):** does the answer stay grounded in the retrieved context, without hallucinated claims?
- **Relevancy (1–5):** does the answer directly address the question asked?

If faithfulness scores ≤2, the pipeline automatically retries with a wider retrieval window (`k`) before returning a final answer — a lightweight form of self-correction built into the graph itself.

## Notes

- Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).
- Upload limits: 10 PDFs per session, 200MB per file (configurable in `app.py`).
- An AWS ECS/Fargate task definition (`task-definition.json`) is included for teams wanting to deploy on dedicated cloud infrastructure beyond Streamlit Cloud.

## License

MIT

