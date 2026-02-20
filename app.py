

# ============================================================
# 1. Import packages needed
# ============================================================
import difflib
import streamlit as st
import re
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

# Initialize model
TEMPERATURE = 0.5

# keeps output < 500 words in practice
REPORT_MAX_TOKENS = 600

# summary length control per Wikipedia page
SUMMARY_MAX_TOKENS = 300

# limit how much of each Wikipedia page we send to the LLM (cost + stability)
MAX_PAGE_CHARACTERS = 6000

MAX_SOURCE_PAGES = 5                 # Q2 requirement

SUMMARY_TEMPERATURE = 0.2

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
# Utility Functions
# ============================================================

def safe_stop(message: str):
    st.error(message)
    st.stop()


def suggest_industry_correction(user_input: str) -> str | None:
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
# 4. Sidebar (LLM + API Key)
# ============================================================
st.sidebar.header("Settings")

llm_choice = st.sidebar.selectbox(
    "Choose LLM",
    options=[MODEL_LLM],
)

api_key = st.sidebar.text_input("Enter API key", type="password")

# ============================================================
# 5–7. Source Summaries (replace embedding + evidence ranking)
# ============================================================

def build_source_summaries(client: OpenAI, docs, llm_choice: str):
    """
    Create short, analyst-oriented summaries for each of the 5 Wikipedia pages.
    This replaces chunking + embeddings + cosine ranking.
    """
    summaries = []

    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "Wikipedia")
        url = d.metadata.get("source", "")
        content = (d.page_content or "").strip()

        # protect cost + keep stable
        content = content[:MAX_PAGE_CHARACTERS]

        prompt = f"""
You are helping with market research.
Summarize the Wikipedia page for a business analyst.

[Page {i}]
Title: {title}
URL: {url}

Content:
{content}

Write 90–120 words with:
- What this page suggests about the industry/market context
- Key entities/terms mentioned
- Any constraints/limitations implied by the page
Return plain text only.
"""

        resp = client.chat.completions.create(
            model=llm_choice,
            temperature=SUMMARY_TEMPERATURE,
            max_tokens=SUMMARY_MAX_TOKENS,
            messages=[
                {"role": "system", "content": "You summarize accurately and do not invent facts."},
                {"role": "user", "content": prompt},
            ],
        )

        summaries.append({
            "idx": i,
            "title": title,
            "url": url,
            "summary": resp.choices[0].message.content.strip()
        })

    return summaries




# ============================================================
# 8. Report Generation (industry-only)
# ============================================================

def generate_report(client, industry, sources_block, llm_choice):
    system_prompt = (
        "You are a market research assistant.\n"
        "Rules:\n"
        "- Output must be fewer than 500 words.\n"
        "- Use ONLY the provided Sources.\n"
        "- Do NOT invent facts, numbers, or citations.\n"
        "- If sources are insufficient, say so.\n"
        "- Use clear headings and bullet points.\n"
        "- When making a claim, cite it as (Source X).\n"
    )

    user_prompt = f"""
Industry:
{industry}

Sources:
{sources_block}

Task:
Write a concise industry report with:
1) Market overview
2) Value chain / segments
3) Competitive landscape / ecosystem notes (only if supported by sources)
4) Key trends
5) Risks & unknowns
6) Opportunities
7) 3 follow-up research questions
""".strip()

    resp = client.chat.completions.create(
        model=llm_choice,
        temperature=TEMPERATURE,
        max_tokens=REPORT_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# --- Source highlight function ---
def highlight_sources(text, source_summaries):
    source_map = {str(s["idx"]): s["url"] for s in source_summaries}

    def replace(match):
        source_number = match.group(1)
        url = source_map.get(source_number, "#")
        return (
            f"<a href='{url}' target='_blank' "
            f"style='color:#2563EB; font-weight:600; text-decoration:none;'>"
            f"(Source {source_number})"
            f"</a>"
        )

    return re.sub(r"\(Source (\d+)\)", replace, text)
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
    industry = st.text_input("Industry", placeholder="e.g. J-pop industry")
    submitted = st.form_submit_button("Generate report")

if submitted:
    industry = (industry or "").strip()

    # Q1: validate industry
    if not industry:
        st.warning("Please enter an industry.")
        st.stop()

    # Q0: validate API key
    if not api_key:
        safe_stop("Please enter an API key in the sidebar.")

    # Typo suggestion
    suggestion = suggest_industry_correction(industry)
    if suggestion:
        st.info(f"Did you mean **{suggestion}**?")
        if st.button("Use suggested spelling"):
            industry = suggestion



    # Q2: retrieve Wikipedia pages
    with st.status("Retrieving top Wikipedia pages…", expanded=False):
        retriever = WikipediaRetriever(top_k_results=MAX_SOURCE_PAGES)
        docs = retriever.invoke(industry)



    if not docs:
        safe_stop("No Wikipedia pages found.")

    if len(docs) < MAX_SOURCE_PAGES:
        st.warning(
            f"Only {len(docs)} relevant Wikipedia page(s) were found. "
            "This may be due to ambiguity or spelling."
        )

    docs = docs[:MAX_SOURCE_PAGES]

    st.subheader("Top 5 relevant Wikipedia pages (URLs)")
    for i, d in enumerate(docs, start=1):
        st.write(f"{i}. {d.metadata.get('title')}: {d.metadata.get('source')}")

    # Q3: generate report
    with st.status("Generating industry report…", expanded=False):
        source_summaries = build_source_summaries(client, docs, llm_choice)

        sources_block = "\n\n".join(
            [f"[Source {s['idx']}] {s['title']} - {s['url']}\n{s['summary']}"
             for s in source_summaries]
        )

        report = generate_report(client, industry, sources_block, llm_choice)

        st.session_state["report"] = report
        st.session_state["source_summaries"] = source_summaries
        st.session_state["industry"] = industry


if "report" in st.session_state:

    report = st.session_state["report"]

    st.subheader("Industry report (<500 words)")
    highlighted_report = highlight_sources(
        report,
        st.session_state["source_summaries"]
    )
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
            "The report exceeds the 500-word limit. Consider reducing REPORT_MAX_TOKENS or lowering TEMPERATURE.")

    if "source_summaries" in st.session_state:
        with st.expander("Show sources used (optional)"):
            for s in st.session_state["source_summaries"]:
                st.markdown(f"**Source {s['idx']}** — *{s['title']}*")
                st.caption(s["url"])
                st.write(s["summary"])
                st.divider()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
