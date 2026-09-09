import streamlit as st
import fitz
import os
import time
import json
import logging
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# ---------- Logging setup ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_app")

def log_event(event_type, **kwargs):
    logger.info(json.dumps({"event": event_type, **kwargs}))

# ---------- Streamlit + Groq setup ----------
st.set_page_config(page_title="PDF Q&A System", layout="wide")
st.title("RAG Document Intelligence System")
st.caption("Upload PDFs and ask questions about them")

client = Groq()

# Configurable model name — change via env var instead of editing code
# when Groq deprecates/renames a model (e.g. llama-3.3-70b-versatile was
# decommissioned Aug 16, 2026).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

MAX_RETRIES = 1
MAX_FILES = 10  # cap number of PDFs per session to control processing time/cost

# ---------- PDF loading & indexing ----------
def load_pdfs(uploaded_files):
    documents = []
    for file in uploaded_files:
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            if not text.strip():
                st.warning(f"'{file.name}' has no extractable text (scanned image PDF?). Skipping.")
                continue
            documents.append({"name": file.name, "text": text})
        except Exception as e:
            st.error(f"Failed to process '{file.name}': {e}")
            log_event("pdf_load_error", file=file.name, error=str(e))
    return documents

def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    all_chunks, all_metadatas = [], []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": doc["name"]}] * len(chunks))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadatas)

# ---------- LLM helper functions ----------
def rewrite_query(question):
    prompt = f"""Rewrite the following question to make it more specific and easier to search for in a document, while keeping all named entities exactly as given. Return ONLY the rewritten question.

Question: {question}"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return question

def evaluate_faithfulness(answer, context):
    if answer.strip().lower().rstrip(".") == "i don't know":
        return 5, "Model correctly declined to answer — no unsupported claims made."
    judge_prompt = f"""Context:
{context}

Answer:
{answer}

Is every claim in the Answer supported by the Context?
Respond in EXACTLY this format:
score: <number 1-5>
reason: <one short sentence>"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        score_line = [l for l in raw.split("\n") if l.lower().startswith("score")][0]
        reason_line = [l for l in raw.split("\n") if l.lower().startswith("reason")][0]
        score = int("".join(filter(str.isdigit, score_line.split(":")[1])))
        reason = reason_line.split(":", 1)[1].strip()
        return score, reason
    except Exception:
        return None, "Could not score this answer."

def evaluate_relevancy(question, answer):
    judge_prompt = f"""Question: {question}
Answer: {answer}

On a scale of 1-5, how directly does the Answer address the Question?
Respond in EXACTLY this format:
score: <number 1-5>
reason: <one short sentence>"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        score_line = [l for l in raw.split("\n") if l.lower().startswith("score")][0]
        reason_line = [l for l in raw.split("\n") if l.lower().startswith("reason")][0]
        score = int("".join(filter(str.isdigit, score_line.split(":")[1])))
        reason = reason_line.split(":", 1)[1].strip()
        return score, reason
    except Exception:
        return None, "Could not score this answer."

# ---------- LangGraph pipeline ----------
class RAGState(TypedDict):
    question: str
    search_query: str
    context: str
    sources: List[str]
    answer: str
    faithfulness_score: Optional[int]
    faithfulness_reason: str
    relevancy_score: Optional[int]
    relevancy_reason: str
    k: int
    retry_count: int
    error: str

def rewrite_node(state: RAGState) -> RAGState:
    try:
        return {**state, "search_query": rewrite_query(state["question"])}
    except Exception as e:
        return {**state, "search_query": state["question"], "error": str(e)}

def retrieve_node(state: RAGState, vectorstore) -> RAGState:
    docs = vectorstore.similarity_search(state["search_query"], k=state["k"])
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata["source"] for d in docs]))
    return {**state, "context": context, "sources": sources}

def generate_node(state: RAGState) -> RAGState:
    start = time.time()
    prompt = f"""Use only the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{state['context']}

Question: {state['question']}
Answer:"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        log_event("generate_success", question=state["question"],
                   latency_ms=int((time.time() - start) * 1000))
        return {**state, "answer": answer}
    except Exception as e:
        log_event("generate_error", question=state["question"], error=str(e))
        return {**state, "answer": f"⚠️ Error generating answer: {e}", "error": str(e)}

def evaluate_node(state: RAGState) -> RAGState:
    faith_score, faith_reason = evaluate_faithfulness(state["answer"], state["context"])
    rel_score, rel_reason = evaluate_relevancy(state["question"], state["answer"])
    log_event("eval_complete", question=state["question"],
               faithfulness=faith_score, relevancy=rel_score)
    return {**state, "faithfulness_score": faith_score, "faithfulness_reason": faith_reason,
            "relevancy_score": rel_score, "relevancy_reason": rel_reason}

def widen_retrieval_node(state: RAGState) -> RAGState:
    log_event("retry_triggered", question=state["question"],
               old_k=state["k"], faithfulness=state["faithfulness_score"])
    return {**state, "k": state["k"] + 4, "retry_count": state["retry_count"] + 1}

def should_retry(state: RAGState) -> str:
    low_faithfulness = state["faithfulness_score"] is not None and state["faithfulness_score"] <= 2
    can_retry = state["retry_count"] < MAX_RETRIES
    return "retry" if (low_faithfulness and can_retry) else "done"

def build_rag_graph(vectorstore):
    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, vectorstore))
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("widen_retrieval", widen_retrieval_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_retry, {"retry": "widen_retrieval", "done": END})
    graph.add_edge("widen_retrieval", "retrieve")
    return graph.compile()

# ---------- Streamlit UI ----------
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Guardrail: cap number of files per session
    if uploaded_files and len(uploaded_files) > MAX_FILES:
        st.error(f"Please upload {MAX_FILES} or fewer PDFs at a time. You selected {len(uploaded_files)}.")
        uploaded_files = None

    if uploaded_files:
        total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
        st.success(f"{len(uploaded_files)} file(s) uploaded ({total_size_mb:.1f} MB total)")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / (1024*1024):.1f} MB)")

    st.divider()
    show_eval = st.checkbox("Show answer quality evaluation", value=True)

if uploaded_files:
    if "vectorstore" not in st.session_state or st.session_state.get("files") != [f.name for f in uploaded_files]:
        with st.spinner("Processing PDFs... please wait"):
            documents = load_pdfs(uploaded_files)
            if not documents:
                st.error("No valid text could be extracted. Please try a different file.")
                st.stop()
            st.session_state.vectorstore = create_vectorstore(documents)
            st.session_state.files = [f.name for f in uploaded_files]
        st.success("PDFs processed! Ask your questions below.")
    else:
        st.success("PDFs ready! Ask your questions below.")

    question = st.text_input("Ask a question about your PDFs")

    if question and not question.strip():
        st.warning("Please enter a valid question.")
    elif question:
        rag_graph = build_rag_graph(st.session_state.vectorstore)
        with st.spinner("Thinking..."):
            result = rag_graph.invoke({
                "question": question, "search_query": "", "context": "", "sources": [],
                "answer": "", "faithfulness_score": None, "faithfulness_reason": "",
                "relevancy_score": None, "relevancy_reason": "", "k": 6,
                "retry_count": 0, "error": ""
            })

        st.markdown("### Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.markdown("### Sources")
            for s in result["sources"]:
                st.caption(f"- {s}")

        if result["retry_count"] > 0:
            st.caption(f"🔄 Retrieval was automatically widened {result['retry_count']} time(s) to improve grounding.")

        if show_eval:
            with st.expander("📊 Answer Quality Evaluation", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Faithfulness", f"{result['faithfulness_score']}/5" if result['faithfulness_score'] else "N/A")
                    st.caption(result["faithfulness_reason"])
                with col2:
                    st.metric("Relevancy", f"{result['relevancy_score']}/5" if result['relevancy_score'] else "N/A")
                    st.caption(result["relevancy_reason"])
                if result["faithfulness_score"] and result["faithfulness_score"] <= 2:
                    st.warning("⚠️ This answer may contain claims not well-supported by the source documents.")
else:
    st.info(f"Please upload at least one PDF from the sidebar to get started (up to {MAX_FILES} files, 200MB each).")