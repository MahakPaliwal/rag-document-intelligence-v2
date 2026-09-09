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

## Architecture
User Question
│
▼
┌─────────────┐
│ Rewrite │ → reformulates the question for better retrieval
└──────┬──────┘
▼
┌─────────────┐
│ Retrieve │ → semantic search over FAISS vector store
└──────┬──────┘
▼
┌─────────────┐
│ Generate │ → Groq LLM answers using retrieved context only
└──────┬──────┘
▼
┌─────────────┐
│ Evaluate │ → scores faithfulness & relevancy (LLM-as-judge)
└──────┬──────┘
▼
Low faithfulness? ──Yes──► Widen retrieval (larger k) ──► back to Retrieve
│
No
▼
Final Answer
