import os
import re
from typing import List, Literal, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()

# =====================================================================
# 1. Page Configuration & Custom CSS (Greenish 2D Blur + White-in-Green Buttons)
# =====================================================================
st.set_page_config(
    page_title="Chess Class Pedagogical Auditor",
    page_icon="♟️",
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* --- 2D BLURRED GREENISH AMBIENT BACKGROUND --- */
[data-testid="stAppViewContainer"] {
    background-color: #0b170e;
    background-image: 
        radial-gradient(circle at 12% 18%, rgba(46, 125, 50, 0.45) 0%, transparent 40%),
        radial-gradient(circle at 85% 25%, rgba(67, 160, 71, 0.35) 0%, transparent 45%),
        radial-gradient(circle at 50% 75%, rgba(27, 94, 32, 0.50) 0%, transparent 50%),
        radial-gradient(circle at 80% 85%, rgba(102, 187, 106, 0.25) 0%, transparent 40%);
    background-attachment: fixed;
    background-size: cover;
    color: #e8f5e9;
}

/* Sidebar Glassmorphism */
[data-testid="stSidebar"] {
    background-color: rgba(9, 21, 12, 0.82) !important;
    backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(76, 175, 80, 0.25) !important;
}

/* --- ALL BUTTONS: WHITE IN GREEN --- */
div.stButton > button, 
button[kind="primary"], 
button[kind="secondary"],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"],
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    border: 1.5px solid #66bb6a !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    box-shadow: 0 4px 15px rgba(46, 125, 50, 0.40) !important;
    transition: all 0.25s ease-in-out !important;
}

div.stButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #ffffff !important;
    color: #1b5e20 !important;
    border: 1.5px solid #2e7d32 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(102, 187, 106, 0.55) !important;
}

/* Dropzone container */
[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(18, 43, 24, 0.55) !important;
    border: 1.5px dashed #4caf50 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
}

/* Text & Headers */
h1, h2, h3, h4, h5, h6 {
    color: #e8f5e9 !important;
    font-weight: 700 !important;
}

.stMarkdown p, .stMarkdown span {
    color: #d1e7dd;
}

/* Input Fields */
input, textarea, select, [data-baseweb="select"] {
    background-color: rgba(18, 43, 24, 0.7) !important;
    color: #ffffff !important;
    border: 1px solid rgba(76, 175, 80, 0.35) !important;
    border-radius: 6px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================================
# 2. Pydantic Output Schemas
# =====================================================================
FeatureState = Literal["Yes", "No", "Blank"]

class FeatureAuditResult(BaseModel):
    f1_individual_questions: FeatureState = Field(
        description="Yes: Asked questions individually by name / included everyone / maintained one to one queries. No: Open room only. Blank: Neutral."
    )
    f1_evidence: str

    f2_homework_given: FeatureState = Field(
        description="Yes: HW was given / assigned homework / gave homework. No: Missed HW / should give HW. Blank: Neutral."
    )
    f2_evidence: str

    f_rapport_building: FeatureState = Field(
        description="Yes: Good rapport building (social check-in, casual banter). No: Skipped rapport / Directly entered class without building rapport. Blank: Neutral."
    )
    f_rapport_evidence: str

    f3_solid_intro: FeatureState = Field(
        description="Yes: Included a solid introduction / Good intro. No: Lacked good intro. Blank: Neutral."
    )
    f3_evidence: str

    f4_setup_and_puzzles: FeatureState = Field(
        description="Yes: Addressed learners during setup and puzzles. No: Silent or left unguided. Blank: No setup/puzzles observed."
    )
    f4_evidence: str

    f5_interactive_flow: FeatureState = Field(
        description="Yes: General interactive flow was present. No: Monologues/dead air. Blank: Neutral."
    )
    f5_evidence: str

    f6_conclusion_quality: FeatureState = Field(
        description="No: Lacked conclusion (ended abruptly). Yes: Proper conclusion/wrap-up given. Blank: Neutral."
    )
    f6_evidence: str

    f7_joiner_intros: FeatureState = Field(
        description="Yes: Joiner introductions present. No: Missed joiner intro (new student present but skipped). Blank: No new joiners."
    )
    f7_evidence: str

    students_called: List[str]

class RubricScoreResult(BaseModel):
    score: Literal[1, 3, 4] = Field(description="Deterministic final score: strictly 1, 3, or 4.")
    tier: Literal["Low Value", "Medium Value", "High Value"] = Field(description="Evaluation tier matching the score.")
    matched_rule: str = Field(description="The exact matched combinatorial rule from the rubric.")
    reasoning: str = Field(description="Brief explanation of why this specific score was awarded based on the feature states.")

# =====================================================================
# 3. Extraction Prompt & Model JSON Parsers
# =====================================================================
EXTRACTION_SYSTEM_PROMPT = """You are an expert pedagogical auditor evaluating online children's chess classes.
Audit the transcript dialogue and return a single valid JSON object strictly matching this format:

{
  "f1_individual_questions": "Yes" | "No" | "Blank",
  "f1_evidence": "direct quote or observation",
  "f2_homework_given": "Yes" | "No" | "Blank",
  "f2_evidence": "direct quote or observation",
  "f_rapport_building": "Yes" | "No" | "Blank",
  "f_rapport_evidence": "direct quote or observation",
  "f3_solid_intro": "Yes" | "No" | "Blank",
  "f3_evidence": "direct quote or observation",
  "f4_setup_and_puzzles": "Yes" | "No" | "Blank",
  "f4_evidence": "direct quote or observation",
  "f5_interactive_flow": "Yes" | "No" | "Blank",
  "f5_evidence": "direct quote or observation",
  "f6_conclusion_quality": "Yes" | "No" | "Blank",
  "f6_evidence": "direct quote or observation",
  "f7_joiner_intros": "Yes" | "No" | "Blank",
  "f7_evidence": "direct quote or observation",
  "students_called": ["Name1", "Name2"]
}

Strict Rules:
- f1_individual_questions: 'Yes' ONLY if tutor calls at least two different students by name to answer. If whole room only, mark 'No'.
- f2_homework_given: 'Yes' ONLY if explicit home practice tasks or puzzles are assigned before ending. If class ends without tasks, mark 'No'.
- f_rapport_building: 'Yes' ONLY if authentic social check-in took place (school, weekend, hobbies). If tutor jumped straight into chess, mark 'No'.
- f6_conclusion_quality: 'No' if the session lacked conclusion / ended abruptly. 'Yes' if wrap-up was delivered.
- f7_joiner_intros: 'Yes' if new student was invited to introduce themselves. 'No' if new joiner was ignored. 'Blank' if no new joiners.
- Return ONLY the JSON object. Do not include introductory text, explanations, or trailing commentary.
"""

SCORING_SYSTEM_PROMPT = """You are an expert pedagogical audit engine evaluating online children's chess classes.
You will receive the classified states (Yes, No, Blank) of 7 pedagogical features + Rapport.
Your job is to evaluate these features strictly according to the combinatorial rubric below and return a valid JSON object.

### COMBINATORIAL SCORING RULES:

[LOW VALUE COMBINATIONS -> SCORE = 1]
Evaluate these veto conditions FIRST:
1. (Lacked good intro + lacked conclusion + joiner introductions)
   Trigger condition: solid_intro == 'No' AND conclusion_quality == 'No' AND joiner_intros == 'Yes'
2. (Skipped rapport + missed joiner intro)
   Trigger condition: rapport_building == 'No' AND joiner_intros == 'No'
3. (Directly entered class without building rapport.)
   Trigger condition: rapport_building == 'No'

[HIGH VALUE COMBINATIONS -> SCORE = 4]
IMPORTANT: If homework_given == 'No', you CANNOT award Score 4. Missing homework is a hard veto for Score 4.
1. (Questions asked individually by name + HW was given)
   Trigger condition: individual_questions == 'Yes' AND homework_given == 'Yes'
2. (Good rapport building + asked questions individually)
   Trigger condition: rapport_building == 'Yes' AND individual_questions == 'Yes' AND homework_given != 'No'

[MEDIUM VALUE COMBINATIONS -> SCORE = 3]
If none of the Low or High combinations apply:
1. (Included a solid introduction + assigned homework)
2. (Good intro + included everyone + missed HW)
3. (Addressed learners during setup and puzzles + gave homework)
4. (Maintained one to one queries + missed Homeworks)
5. (General interactive flow was present)

Return ONLY a valid JSON object strictly matching this format:
{
  "score": 1 | 3 | 4,
  "tier": "Low Value" | "Medium Value" | "High Value",
  "matched_rule": "The exact rule string matched from above",
  "reasoning": "Brief explanation of the decision"
}
"""

def extract_compact_payload(raw_vtt: str) -> str:
    lines = [
        l.strip() for l in raw_vtt.splitlines()
        if l.strip() and not l.startswith("WEBVTT") and not l.isdigit() and "-->" not in l
    ]
    full_text = "\n".join(lines)
    
    opening = full_text[:4000]       # First ~10 mins
    closing = full_text[-3500:]      # Last ~10 mins
    
    mid_lines = full_text[4000:-3500].splitlines()
    mid_sample = [
        l for l in mid_lines
        if any(w in l.lower() for w in ["?", "puzzle", "move", "can you", "what is", "try", "white to play"])
    ][:35]
    mid_sample_text = "\n".join(mid_sample)
    
    return f"""=== CLASS OPENING (MINUTES 0-10) ===
{opening}

=== MID-CLASS INTERACTIVE & PUZZLE EXCHANGES ===
{mid_sample_text}

=== CLASS CLOSING & HOMEWORK (FINAL 10 MINUTES) ===
{closing}"""

def extract_raw_text(raw_response) -> str:
    if hasattr(raw_response, "content"):
        content = raw_response.content
        if isinstance(content, list):
            content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        if not content.strip():
            add_kw = getattr(raw_response, "additional_kwargs", {})
            content = add_kw.get("reasoning_content", "") or add_kw.get("reasoning", "")
    else:
        content = str(raw_response)
    return content

def parse_llm_json_response(raw_response) -> FeatureAuditResult:
    content = extract_raw_text(raw_response)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not locate JSON block in response: {content[:300]}")
    return FeatureAuditResult.model_validate_json(match.group(0))

# =====================================================================
# 4. Model-Driven Scoring Engine (LLM-as-a-Judge)
# =====================================================================
def compute_score_and_rule_llm(audit: FeatureAuditResult, llm: ChatGroq) -> Tuple[int, str, str]:
    """Uses LLM-as-a-judge to evaluate extracted features against rubric combinations."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SCORING_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("""Classified Feature States:
- individual_questions (F1): {f1}
- homework_given (F2): {f2}
- rapport_building: {rap}
- solid_intro (F3): {f3}
- setup_and_puzzles (F4): {f4}
- interactive_flow (F5): {f5}
- conclusion_quality (F6): {f6}
- joiner_intros (F7): {f7}
- students_called: {students}

Evaluate against the combinatorial rules and output JSON.""")
    ])

    chain = prompt | llm
    raw_response = chain.invoke({
        "f1": audit.f1_individual_questions,
        "f2": audit.f2_homework_given,
        "rap": audit.f_rapport_building,
        "f3": audit.f3_solid_intro,
        "f4": audit.f4_setup_and_puzzles,
        "f5": audit.f5_interactive_flow,
        "f6": audit.f6_conclusion_quality,
        "f7": audit.f7_joiner_intros,
        "students": ", ".join(audit.students_called) if audit.students_called else "None"
    })

    content = extract_raw_text(raw_response)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse scoring JSON: {content[:300]}")

    result = RubricScoreResult.model_validate_json(match.group(0))
    return result.score, result.tier, result.matched_rule

# =====================================================================
# 5. Application Header & UI Layout
# =====================================================================
st.title("♟️ Pedagogical Quality Audit Engine")
st.markdown("Automated evaluation of chess coaching transcripts (`.vtt`) against the **7 atomic pedagogical features** and strict combinatorial decision rules.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Engine Settings")
    groq_api_key = st.text_input(
        "Groq API Key", 
        value=os.environ.get("GROQ_API_KEY", ""), 
        type="password",
        help="Reads from .env by default or enter manually."
    )
    selected_model = st.selectbox(
        "Groq Model", 
        ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Scoring Model Mechanism:**")
    st.markdown("- 🤖 **LLM Perception:** Classifies the 7 features & quotes")
    st.markdown("- ⚖️ **LLM Judge:** Evaluates rubric combinations & veto gates")
    st.markdown("---")
    st.markdown("**Pedagogical Tiers:**")
    st.markdown("- 🟢 **Score 4 (High):** Questions by name + HW / Rapport")
    st.markdown("- 🟠 **Score 3 (Medium):** Solid intro + HW / Interactive flow")
    st.markdown("- 🔴 **Score 1 (Low):** Skipped rapport / Missed joiners")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload Class Transcript (.vtt)", 
    type=["vtt"], 
    accept_multiple_files=True
)

if uploaded_files:
    if not groq_api_key:
        st.error("Please provide a valid Groq API Key in the sidebar to proceed.")
    else:
        llm = ChatGroq(
            model_name=selected_model,
            groq_api_key=groq_api_key,
            temperature=0.0
        )
        extraction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("Class Transcript Dialogue:\n\n{transcript_text}")
        ])
        extraction_chain = extraction_prompt | llm

        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
            file_name = uploaded_file.name

            with st.spinner(f"Auditing `{file_name}`..."):
                try:
                    compact_payload = extract_compact_payload(file_content)
                    
                    # 1. Feature Extraction (LLM Perception)
                    raw_response = extraction_chain.invoke({"transcript_text": compact_payload})
                    extracted = parse_llm_json_response(raw_response)
                    
                    # 2. Score Prediction (LLM as Judge)
                    score, tier, rule = compute_score_and_rule_llm(extracted, llm)

                    score_color = "#4ade80" if score == 4 else ("#fb923c" if score == 3 else "#f87171")
                    badge_bg = "rgba(74, 222, 128, 0.15)" if score == 4 else ("rgba(251, 146, 60, 0.15)" if score == 3 else "rgba(248, 113, 113, 0.15)")

                    st.markdown("---")
                    
                    # Top Metric Summary Card with Glassmorphism
                    card_html = f"""
                    <div style="background: rgba(18, 43, 25, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(76, 175, 80, 0.35); border-left: 6px solid {score_color}; border-radius: 10px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                            <div>
                                <h3 style="margin: 0; color: #f1f8e9; font-size: 20px;">📄 {file_name}</h3>
                                <p style="margin: 6px 0 0 0; color: #a5d6a7; font-size: 14px;">
                                    <b>Students Identified:</b> {', '.join(extracted.students_called) if extracted.students_called else 'None'}
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <div style="display: inline-block; background: {badge_bg}; border: 1px solid {score_color}; padding: 6px 16px; border-radius: 8px;">
                                    <span style="font-size: 24px; font-weight: 800; color: {score_color};">{score} / 4</span>
                                    <span style="font-size: 15px; font-weight: 600; color: {score_color}; margin-left: 6px;">({tier})</span>
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(76, 175, 80, 0.2); font-size: 14px; color: #c8e6c9;">
                            <b>Matched Combinatorial Rule:</b> <i style="color: #ffffff;">"{rule}"</i>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                    # Table Data: Sequential 1 to 8 (Rapport properly indexed at 3)
                    table_data = [
                        ("1", "Individual Questioning (1-on-1 / Name)", extracted.f1_individual_questions, extracted.f1_evidence),
                        ("2", "Homework Assignment / Tasks", extracted.f2_homework_given, extracted.f2_evidence),
                        ("3", "Rapport Building Check-in", extracted.f_rapport_building, extracted.f_rapport_evidence),
                        ("4", "Solid Session Introduction", extracted.f3_solid_intro, extracted.f3_evidence),
                        ("5", "Setup & Puzzles Engagement", extracted.f4_setup_and_puzzles, extracted.f4_evidence),
                        ("6", "General Interactive Flow", extracted.f5_interactive_flow, extracted.f5_evidence),
                        ("7", "Session Conclusion Quality", extracted.f6_conclusion_quality, extracted.f6_evidence),
                        ("8", "Joiner Introductions", extracted.f7_joiner_intros, extracted.f7_evidence),
                    ]

                    rows_list = []
                    for num, name, state, ev in table_data:
                        if state == "Yes":
                            badge = '<span style="background: rgba(46, 125, 50, 0.35); color: #81c784; border: 1px solid #4caf50; font-weight: bold; padding: 3px 10px; border-radius: 4px;">Yes</span>'
                        elif state == "No":
                            badge = '<span style="background: rgba(198, 40, 40, 0.35); color: #ef9a9a; border: 1px solid #e57373; font-weight: bold; padding: 3px 10px; border-radius: 4px;">No</span>'
                        else:
                            badge = '<span style="color: #9e9e9e; padding: 3px 10px;">—</span>'

                        clean_ev = ev.replace("<", "&lt;").replace(">", "&gt;")
                        row = f'<tr style="border-bottom: 1px solid rgba(76, 175, 80, 0.15);"><td style="padding: 12px; text-align: center; font-weight: bold; color: #a5d6a7;">{num}</td><td style="padding: 12px; font-weight: 600; color: #ffffff;">{name}</td><td style="padding: 12px; text-align: center;">{badge}</td><td style="padding: 12px; font-size: 13.5px; color: #c8e6c9; line-height: 1.45;">{clean_ev}</td></tr>'
                        rows_list.append(row)

                    rows_html = "".join(rows_list)
                    full_table_html = f'<div style="background: rgba(14, 34, 20, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(76, 175, 80, 0.3); border-radius: 10px; overflow: hidden; margin-top: 14px; box-shadow: 0 4px 18px rgba(0,0,0,0.25);"><table style="width: 100%; border-collapse: collapse; font-size: 14px; text-align: left;"><thead><tr style="background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%); color: #ffffff;"><th style="padding: 12px; width: 45px; text-align: center;">#</th><th style="padding: 12px; width: 280px;">Feature</th><th style="padding: 12px; width: 90px; text-align: center;">State</th><th style="padding: 12px;">Evidence Quote / Observation</th></tr></thead><tbody>{rows_html}</tbody></table></div>'
                    
                    if hasattr(st, "html"):
                        st.html(full_table_html)
                    else:
                        st.markdown(full_table_html, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error auditing `{file_name}`: {e}")