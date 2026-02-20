

# ============================================================
# 1. Import pacakgaes needed
# ============================================================
import hashlib
import difflib
import numpy as np
import streamlit as st
from openai import OpenAI
from langchain_community.retrievers import WikipediaRetriever
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO
# ============================================================
# 2. Configuration
# ============================================================

# ChatGPT Model 40 mini for speed
MODEL_LLM = "gpt-4o-mini"
MODEL_EMBED = "text-embedding-3-small"

# Initialize model
TEMPERATURE = 0.5

# keeps output < 500 words in practice
MAX_OUTPUT_TOKENS = 600

TOP_K_PAGES = 5                  # Q2 requirement
TOP_K_EVIDENCE_CHUNKS = 5

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MAX_CHUNKS_TO_EMBED = 40
EMBED_BATCH_SIZE = 64

# ============================================================
# 3. Application Setup
# ============================================================

st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Market Research Assistant")
st.caption(
    "Enter an industry. The assistant retrieves the top 5 relevant Wikipedia pages "
    "and generates a concise industry report based only on those sources."
)


# ============================================================
# 4. Sidebar (LLM + API Key)
# ============================================================
st.sidebar.header("Settings")

llm_choice = st.sidebar.selectbox(
    "Choose LLM",
    options=[MODEL_LLM],
)

api_key = st.sidebar.text_input("Enter API key", type="password")

# ============================================================
# 5.  Text Chunking, Embedding and Evidence Ranking
# ============================================================

def safe_stop(message: str):
    """Display an error message and halt execution."""
    st.error(message)
    st.stop()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split long Wikipedia text into overlapping chunks so that
    semantic embeddings retain context.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def stable_hash(items: list[str]) -> str:
    """Generate a stable hash for embedding cache keys."""
    return hashlib.md5("||".join(items).encode("utf-8")).hexdigest()


def embed_batched(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """Embed texts in batches to avoid API rate issues."""
    vectors = []
    try:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            resp = client.embeddings.create(
                model=MODEL_EMBED,
                input=batch
            )
            vectors.extend(
                [np.array(e.embedding, dtype=np.float32) for e in resp.data]
            )
        return vectors
    except Exception as e:
        safe_stop(f"Embedding failed: {e}")


@st.cache_data(show_spinner=False)
def cached_embeddings(texts: list[str], api_key: str, cache_key: str):
    """
    Cache embeddings so repeated runs with the same industry
    do not recompute vectors.
    """
    _ = cache_key
    client = OpenAI(api_key=api_key)
    vectors = embed_batched(client, texts)
    return [v.tolist() for v in vectors]

# ============================================================
# 6. Query Normalisation And Spelling Correction
# ============================================================

def suggest_industry_correction(user_input: str) -> str | None:
    """
    Suggest a corrected industry spelling using Wikipedia titles.
    Does NOT auto-correct without user confirmation.
    """
    retriever = WikipediaRetriever(top_k_results=3)
    docs = retriever.invoke(user_input)

    if not docs:
        return None

    top_title = docs[0].metadata.get("title", "").lower()
    similarity = difflib.SequenceMatcher(
        None, user_input.lower(), top_title
    ).ratio()

    if similarity >= 0.75 and user_input.lower() != top_title:
        return top_title

    return None

# ============================================================
# 7. EVIDENCE RETRIEVAL (industry-only, stable)
# ============================================================

def retrieve_relevant_context(client, docs, query, api_key):
    """
    Select the most relevant evidence chunks from the retrieved
    Wikipedia pages using semantic similarity.
    """
    chunks, metas = [], []

    for d in docs:
        title = d.metadata.get("title", "Wikipedia")
        source = d.metadata.get("source", "")
        for ch in chunk_text(d.page_content, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(ch)
            metas.append({
                "title": title,
                "source": source,
                "preview": ch[:200].replace("\n", " ") + "..."
            })

    if not chunks:
        return "", []

    # Limit for performance and determinism
    chunks = chunks[:MAX_CHUNKS_TO_EMBED]
    metas = metas[:MAX_CHUNKS_TO_EMBED]

    # Embed query and chunks
    query_vec = np.array(
        cached_embeddings([query], api_key, stable_hash([query]))[0],
        dtype=np.float32
    )

    chunk_vecs = [
        np.array(v, dtype=np.float32)
        for v in cached_embeddings(chunks, api_key, stable_hash(chunks))
    ]

    scores = [cosine_similarity(query_vec, v) for v in chunk_vecs]
    top_idx = np.argsort(scores)[::-1][:TOP_K_EVIDENCE_CHUNKS]

    evidence_text, evidence_meta = [], []
    for i, idx in enumerate(top_idx, start=1):
        evidence_text.append(f"[Evidence {i}] {chunks[idx]}")
        evidence_meta.append({**metas[idx], "score": scores[idx]})

    return "\n\n".join(evidence_text), evidence_meta

# ============================================================
# 8. Report Generation (industry-only)
# ============================================================

def generate_report(client, industry, evidence, llm_choice):
    """
    Generate a <500-word industry report using only Wikipedia evidence.
    """
    system_prompt = (
        "You are a market research assistant.\n"
        "Rules:\n"
        "- The report must be fewer than 500 words.\n"
        "- Use ONLY the provided Wikipedia evidence.\n"
        "- Do NOT invent facts, statistics, or citations.\n"
        "- If evidence is insufficient, say so.\n"
        "- Use clear headings and bullet points.\n"
        "- If you dont know the exact answer, apologize and say that you do not know. \n"
        "- When making a claim, reference it with (Evidence X).\n"

    )

    user_prompt = f"""
Industry:
{industry}

Wikipedia evidence:
{evidence}

Task:
Write a concise industry report with:
1) Industry overview
2) Market structure
3) Key trends
4) Risks
5) Opportunities
6) 3 follow-up research questions
"""

    resp = client.chat.completions.create(
        model=llm_choice,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.content

# --- Evidence highlight function ---
import re

def highlight_evidence(text):
    return re.sub(
        r"\(Evidence (\d+)\)",
        r"<span style='color:#2563EB; font-weight:600;'>(Evidence \1)</span>",
        text
    )


# ============================================================

# ============================================================
# 9. API KEY INITIALISATION
# ============================================================

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


# ============================================================
# 10. USER INTERFACE (Q1–Q3)
# ============================================================

with st.form("industry_form"):
    industry = st.text_input("Industry", placeholder="e.g. Jpop industry")
    submitted = st.form_submit_button("Generate report")

if submitted:
    industry = (industry or "").strip()

    # Q1: validate industry
    if not industry:
        st.warning("Please enter an industry.")
        st.stop()

    # Q0: validate API key
    if not api_key:
        safe_stop("Please enter an API key in the sidebar (Q0).")

    # Typo suggestion
    suggestion = suggest_industry_correction(industry)
    if suggestion:
        st.info(f"Did you mean **{suggestion}**?")
        if st.button("Use suggested spelling"):
            industry = suggestion

    # Fixed retrieval query (industry-only)
    retrieval_query = f"Provide an industry overview of the {industry}."

    # Q2: retrieve Wikipedia pages
    with st.status("Retrieving top Wikipedia pages…", expanded=False):
        retriever = WikipediaRetriever(top_k_results=TOP_K_PAGES)
        docs = retriever.invoke(industry)



    if not docs:
        safe_stop("No Wikipedia pages found.")

    if len(docs) < TOP_K_PAGES:
        st.warning(
            f"Only {len(docs)} relevant Wikipedia page(s) were found. "
            "This may be due to ambiguity or spelling."
        )

    docs = docs[:TOP_K_PAGES]

    st.subheader("Top 5 relevant Wikipedia pages (URLs)")
    for i, d in enumerate(docs, start=1):
        st.write(f"{i}. {d.metadata.get('title')}: {d.metadata.get('source')}")

    # Q3: generate report
    with st.status("Generating industry report…", expanded=False):
        context, evidence_meta = retrieve_relevant_context(
            client, docs, retrieval_query, api_key
        )
        report = generate_report(
            client, industry, context, llm_choice
        )
        st.session_state["report"] = report
        st.session_state["evidence_meta"] = evidence_meta
        st.session_state["industry"] = industry

if "report" in st.session_state:

    report = st.session_state["report"]

    st.subheader("Industry report (<500 words)")
    highlighted_report = highlight_evidence(report)

    st.markdown(highlighted_report, unsafe_allow_html=True)


    # ---- Generate PDF ----
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    report = st.session_state["report"]

    for line in report.split("\n"):
        elements.append(Paragraph(line, normal_style))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        "Download report (PDF)",
        data=buffer,
        file_name=f"{industry}_report.pdf",
        mime="application/pdf"
    )

    # --- report length check ---
    word_count = len((report or "").split())
    st.caption(f"Word count: {word_count}")

    if word_count > 500:
        st.warning(
            "The report exceeds the 500-word limit. Consider reducing MAX_OUTPUT_TOKENS or lowering TEMPERATURE.")

    if "evidence_meta" in st.session_state:
        with st.expander("Show evidence used (optional)"):
            for i, ev in enumerate(st.session_state["evidence_meta"], start=1):
                st.markdown(f"**Evidence {i}** — score: `{ev['score']:.3f}` — *{ev['title']}*")
                st.caption(ev["source"])
                st.write(ev["preview"])
                st.divider()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
