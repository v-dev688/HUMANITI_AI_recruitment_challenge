import os
import re
import cv2
import base64
import tempfile
from typing import List, Literal, Tuple, Optional
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
    page_title="Dual-Modal AV Audit Engine",
    page_icon="💻",
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
    background-color: rgba(9, 21, 12, 0.85) !important;
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
    score: Literal[1, 3, 4] = Field(description="Deterministic final transcript score: strictly 1, 3, or 4.")
    tier: Literal["Low Value", "Medium Value", "High Value"] = Field(description="Evaluation tier matching the score.")
    matched_rule: str = Field(description="The exact matched combinatorial rule from the rubric.")
    reasoning: str = Field(description="Brief explanation of why this specific score was awarded based on the feature states.")

QualityClass = Literal[
    "Excellent", 
    "Good", 
    "Average", 
    "Weak", 
    "Known Quality Issue"
]

class VideoAudioAuditResult(BaseModel):
    predicted_class: QualityClass = Field(
        description="One of: Excellent, Good, Average, Weak, Known Quality Issue"
    )
    score: int = Field(
        description="4 for Excellent, 3 for Good, 2 for Average, 1 for Weak, 0 for Known Quality Issue"
    )
    audio_feature_matched: str = Field(
        description="Exact matched audio rubric condition from the specified rubric."
    )
    audio_evidence: str = Field(
        description="Observations regarding vocal delivery, teacher tone, energy, and engagement cadence."
    )
    video_feature_matched: str = Field(
        description="Exact matched video rubric condition from the specified rubric."
    )
    video_evidence: str = Field(
        description="Observations from keyframes regarding lighting, 8x8 background, framing, and face visibility."
    )

# =====================================================================
# 3. System Prompts
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

AV_RUBRIC_SYSTEM_PROMPT = """You are an expert pedagogical and technical audio-visual quality auditor evaluating children's online chess coaching video recordings (.mp4).
Evaluate the provided audio-visual summary and keyframe telemetry strictly against these 5 tiered classes:

### CLASS 1: Excellent (Score: 4)
- Audio: "Good interaction tone maintained throughout" OR "Tone was adequate for class engagement"
- Video: "Tone was adequate for class engagement." AND "Background and general setup were acceptable."

### CLASS 2: Good (Score: 3)
- Audio: "Needed more pleasant verbal interaction." OR "Delivery was balanced with reasonable interaction."
- Video: "Branded 8*8 background was properly fixed." OR "Acceptable camera setup and framing."

### CLASS 3: Average (Score: 2)
- Audio: "Standard delivery without major vocal issues." OR "Needed more energetic and pleasant interaction."
- Video: "Clarity problems noted with the virtual background." OR "Insufficient room lighting affected the image."

### CLASS 4: Weak (Score: 1)
- Audio: "Needed more energetic and pleasant delivery." OR "Subdued tone without warm opening engagement."
- Video: "Virtual background was maintained." OR "Camera was turned off while teaching."

### CLASS 5: Known Quality Issue (Score: 0)
- Audio: "Needed a happier, warmer communication style." OR "Needed happier communication with learners."
- Video: "Virtual background was missing entirely." OR "Tutor's face was not visible on camera."

CRITICAL VETO RULES:
1. If the tutor's face is not visible on camera or virtual background is missing entirely -> Classify strictly as 'Known Quality Issue'.
2. If room lighting is insufficient or virtual background has noticeable clarity/clipping issues -> Classify strictly as 'Average'.
3. If camera was turned off during teaching -> Classify strictly as 'Weak'.
4. To qualify as 'Excellent' or 'Good', camera framing must be centered, room sufficiently illuminated, and the branded 8*8 background properly fixed.

Output strictly a single valid JSON object matching this schema:
{
  "predicted_class": "Excellent" | "Good" | "Average" | "Weak" | "Known Quality Issue",
  "score": 4 | 3 | 2 | 1 | 0,
  "audio_feature_matched": "Exact matched condition string",
  "audio_evidence": "Concrete observations regarding audio interaction and tone",
  "video_feature_matched": "Exact matched condition string",
  "video_evidence": "Concrete observations regarding lighting, 8x8 background, and framing"
}
"""

# =====================================================================
# 4. Helper Utilities (Large File Streaming, Extraction, Parsing)
# =====================================================================
def extract_compact_transcript_payload(raw_vtt: str) -> str:
    lines = [
        l.strip() for l in raw_vtt.splitlines()
        if l.strip() and not l.startswith("WEBVTT") and not l.isdigit() and "-->" not in l
    ]
    full_text = "\n".join(lines)
    opening = full_text[:4000]
    closing = full_text[-3500:]
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

def extract_uniform_keyframes(video_path: str, num_frames: int = 10) -> List[Tuple[str, str, dict]]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total_frames <= 0:
        cap.release()
        return []

    duration_secs = total_frames / fps
    start_frame = int(fps * 120) if duration_secs > 300 else int(fps * 15)
    end_frame = max(start_frame + 1, int(total_frames - (fps * 120)))
    step = (end_frame - start_frame) // max(1, num_frames - 1)

    keyframes = []
    for i in range(num_frames):
        target_frame = min(start_frame + (i * step), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret:
            cur_sec = int(target_frame / fps)
            ts_str = f"{cur_sec // 60:02d}:{cur_sec % 60:02d}"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            contrast = float(gray.std())
            h, w = frame.shape[:2]
            scaled_w = int(w * (360 / h))
            frame_resized = cv2.resize(frame, (scaled_w, 360))
            _, buf = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buf).decode("utf-8")
            metrics = {
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "is_dark": brightness < 65.0
            }
            keyframes.append((ts_str, b64_str, metrics))
    cap.release()
    return keyframes

def save_uploaded_video_chunked(uploaded_file) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    chunk_size = 8 * 1024 * 1024
    while True:
        chunk = uploaded_file.read(chunk_size)
        if not chunk:
            break
        temp_file.write(chunk)
    temp_file.flush()
    temp_file.close()
    return temp_file.name

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

def evaluate_transcript(compact_text: str, llm: ChatGroq) -> Tuple[FeatureAuditResult, RubricScoreResult]:
    extraction_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("Class Transcript Dialogue:\n\n{transcript_text}")
    ])
    chain = extraction_prompt | llm
    raw_res = chain.invoke({"transcript_text": compact_text})
    content = extract_raw_text(raw_res)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse transcript JSON: {content[:200]}")
    audit = FeatureAuditResult.model_validate_json(match.group(0))

    scoring_prompt = ChatPromptTemplate.from_messages([
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

Evaluate against combinatorial rules and output JSON.""")
    ])
    score_chain = scoring_prompt | llm
    raw_score = score_chain.invoke({
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
    s_content = extract_raw_text(raw_score)
    s_match = re.search(r"\{.*\}", s_content, re.DOTALL)
    if not s_match:
        raise ValueError(f"Could not parse scoring JSON: {s_content[:200]}")
    score_res = RubricScoreResult.model_validate_json(s_match.group(0))
    return audit, score_res

def evaluate_video(file_name: str, keyframes: List[Tuple[str, str, dict]], llm: ChatGroq) -> VideoAudioAuditResult:
    dark_count = sum(1 for _, _, m in keyframes if m["is_dark"])
    avg_brightness = sum(m["brightness"] for _, _, m in keyframes) / len(keyframes)
    telemetry = f"""Duration: ~60 minutes
Sampled Frames: {len(keyframes)}
Average Luminance: {avg_brightness:.1f} / 255
Dark Frames: {dark_count} / {len(keyframes)}
Timestamps: {', '.join([ts for ts, _, _ in keyframes])}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=AV_RUBRIC_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("""Class Video Recording: {file_name}
Visual Telemetry:
{telemetry}

Evaluate the recording against the 5 standardized classes and return evaluation strictly in JSON format.""")
    ])
    chain = prompt | llm
    raw_res = chain.invoke({"file_name": file_name, "telemetry": telemetry})
    content = extract_raw_text(raw_res)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse video JSON: {content[:200]}")
    return VideoAudioAuditResult.model_validate_json(match.group(0))

def compute_overall_pool(total_score: int) -> Tuple[str, str]:
    """
    Pools total score according to specified thresholds:
    - >= 7: Excellent
    - 5-6: Good
    - 3-4: Average
    - < 3: Poor
    """
    if total_score >= 7:
        return "Excellent", "#4ade80"
    elif total_score >= 5:
        return "Good", "#2ecc71"
    elif total_score >= 3:
        return "Average", "#fb923c"
    else:
        return "Poor", "#ef4444"

# =====================================================================
# 5. Streamlit App Layout
# =====================================================================
st.title(" 💻 Dual-Modal Class Quality Audit Engine")
st.markdown("Automated joint evaluation of chess coaching recordings: **Pedagogical Transcripts (.vtt)** + **Audio/Video Keyframes (.mp4, up to 500MB)**.")

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
    num_frames = st.slider("Video Keyframes Sampled", min_value=6, max_value=12, value=10, step=2)
    st.markdown("---")
    st.markdown("**Overall Score Pooling Rubric (out of 8):**")
    st.markdown("- 🟢 **≥ 7:** Excellent")
    st.markdown("- 🟢 **5 – 6:** Good")
    st.markdown("- 🟠 **3 – 4:** Average")
    st.markdown("- 🔴 **< 3:** Poor")

col_left, col_right = st.columns(2)
with col_left:
    uploaded_vtt = st.file_uploader("1. Upload Transcript (.vtt)", type=["vtt"])
with col_right:
    uploaded_mp4 = st.file_uploader("2. Upload Video (.mp4, up to 500MB)", type=["mp4"])

run_btn = st.button("🚀 Run Dual-Modal Quality Audit", use_container_width=True)

if run_btn:
    if not groq_api_key:
        st.error("Please enter a valid Groq API Key in the sidebar.")
    elif not uploaded_vtt and not uploaded_mp4:
        st.warning("Please upload at least a transcript (.vtt) or video (.mp4) file to audit.")
    else:
        llm = ChatGroq(model_name=selected_model, groq_api_key=groq_api_key, temperature=0.0)
        
        transcript_audit: Optional[FeatureAuditResult] = None
        transcript_score: Optional[RubricScoreResult] = None
        video_audit: Optional[VideoAudioAuditResult] = None
        temp_video_path = None

        # -------------------------------------------------------------
        # Phase 1: Transcript Audit
        # -------------------------------------------------------------
        if uploaded_vtt:
            with st.spinner("Analyzing transcript features and combinatorial rules..."):
                vtt_text = uploaded_vtt.read().decode("utf-8", errors="ignore")
                compact_text = extract_compact_transcript_payload(vtt_text)
                transcript_audit, transcript_score = evaluate_transcript(compact_text, llm)

        # -------------------------------------------------------------
        # Phase 2: Video & Audio Audit
        # -------------------------------------------------------------
        if uploaded_mp4:
            with st.spinner("Streaming video cache and extracting keyframes across 60 mins..."):
                temp_video_path = save_uploaded_video_chunked(uploaded_mp4)
                sampled_frames = extract_uniform_keyframes(temp_video_path, num_frames=num_frames)

            if sampled_frames:
                with st.spinner("Evaluating AV quality against 5 tiers..."):
                    video_audit = evaluate_video(uploaded_mp4.name, sampled_frames, llm)

        # -------------------------------------------------------------
        # Phase 3: Overall Pooling Calculation
        # -------------------------------------------------------------
        t_pts = transcript_score.score if transcript_score else 0
        v_pts = video_audit.score if video_audit else 0
        total_pts = t_pts + v_pts
        max_possible = (4 if transcript_score else 0) + (4 if video_audit else 0)

        pool_class, pool_color = compute_overall_pool(total_pts)

        st.markdown("---")
        
        # Master Pooling Header Banner
        overall_banner_html = f"""
        <div style="background: rgba(18, 43, 25, 0.85); backdrop-filter: blur(14px); border: 2px solid {pool_color}; border-radius: 12px; padding: 22px 28px; margin-bottom: 24px; box-shadow: 0 6px 30px rgba(0,0,0,0.45);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                    <span style="background: rgba(255,255,255,0.08); padding: 4px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; color: #a5d6a7; text-transform: uppercase; letter-spacing: 0.5px;">Master Evaluation Summary</span>
                    <h2 style="margin: 8px 0 0 0; color: #ffffff; font-size: 28px;">Overall Class Rating: <span style="color: {pool_color}; font-weight: 800;">{pool_class}</span></h2>
                    <p style="margin: 6px 0 0 0; color: #c8e6c9; font-size: 15px;">
                        Combined Score: <b>{total_pts} / {max_possible}</b> &nbsp;|&nbsp; 
                        Pedagogical Transcript: <b>{t_pts}/4</b> &nbsp;|&nbsp; 
                        AV/Keyframe Setup: <b>{v_pts}/4</b>
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="background: rgba(0, 0, 0, 0.35); border: 1.5px solid {pool_color}; padding: 12px 24px; border-radius: 10px; text-align: center;">
                        <span style="display: block; font-size: 12px; color: #a5d6a7; text-transform: uppercase;">Final Pool</span>
                        <span style="font-size: 32px; font-weight: 800; color: {pool_color}; line-height: 1.1;">{pool_class}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        st.html(overall_banner_html)

        # -------------------------------------------------------------
        # Phase 4: Detailed Breakdowns (Tabs for Clean Viewing)
        # -------------------------------------------------------------
        tab_pedagogy, tab_av, tab_frames = st.tabs(["📋 Pedagogical Transcript Audit", "🎥 Audio/Video Quality Audit", "🎞️ Sampled Keyframe Gallery"])

        with tab_pedagogy:
            if transcript_audit and transcript_score:
                t_color = "#4ade80" if transcript_score.score == 4 else ("#fb923c" if transcript_score.score == 3 else "#f87171")
                
                st.markdown(f"""
                <div style="background: rgba(18, 43, 25, 0.65); border-left: 5px solid {t_color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;">
                    <div style="font-size: 18px; font-weight: bold; color: #ffffff;">📄 Transcript: {uploaded_vtt.name}</div>
                    <div style="font-size: 14.5px; color: #c8e6c9; margin-top: 4px;">
                        Score: <span style="color: {t_color}; font-weight: bold;">{transcript_score.score} / 4</span> ({transcript_score.tier}) &nbsp;|&nbsp;
                        Rule: <i>"{transcript_score.matched_rule}"</i>
                    </div>
                    <div style="font-size: 13.5px; color: #a5d6a7; margin-top: 3px;">
                        Learners Called: <b>{', '.join(transcript_audit.students_called) if transcript_audit.students_called else 'None'}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                table_data = [
                    ("1", "Individual Questioning (1-on-1 / Name)", transcript_audit.f1_individual_questions, transcript_audit.f1_evidence),
                    ("2", "Homework Assignment / Tasks", transcript_audit.f2_homework_given, transcript_audit.f2_evidence),
                    ("3", "Rapport Building Check-in", transcript_audit.f_rapport_building, transcript_audit.f_rapport_evidence),
                    ("4", "Solid Session Introduction", transcript_audit.f3_solid_intro, transcript_audit.f3_evidence),
                    ("5", "Setup & Puzzles Engagement", transcript_audit.f4_setup_and_puzzles, transcript_audit.f4_evidence),
                    ("6", "General Interactive Flow", transcript_audit.f5_interactive_flow, transcript_audit.f5_evidence),
                    ("7", "Session Conclusion Quality", transcript_audit.f6_conclusion_quality, transcript_audit.f6_evidence),
                    ("8", "Joiner Introductions", transcript_audit.f7_joiner_intros, transcript_audit.f7_evidence),
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

                full_t_html = f'<div style="background: rgba(14, 34, 20, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(76, 175, 80, 0.3); border-radius: 10px; overflow: hidden; margin-top: 10px;"><table style="width: 100%; border-collapse: collapse; font-size: 14px; text-align: left;"><thead><tr style="background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%); color: #ffffff;"><th style="padding: 12px; width: 45px; text-align: center;">#</th><th style="padding: 12px; width: 280px;">Feature</th><th style="padding: 12px; width: 90px; text-align: center;">State</th><th style="padding: 12px;">Evidence Quote / Observation</th></tr></thead><tbody>{"".join(rows_list)}</tbody></table></div>'
                st.html(full_t_html)
            else:
                st.info("No transcript uploaded for evaluation.")

        with tab_av:
            if video_audit:
                av_color_map = {
                    "Excellent": "#4ade80",
                    "Good": "#2ecc71",
                    "Average": "#fb923c",
                    "Weak": "#f87171",
                    "Known Quality Issue": "#ef4444"
                }
                v_col = av_color_map.get(video_audit.predicted_class, "#4ade80")
                
                st.markdown(f"""
                <div style="background: rgba(18, 43, 25, 0.65); border-left: 5px solid {v_col}; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;">
                    <div style="font-size: 18px; font-weight: bold; color: #ffffff;">🎥 Video Recording: {uploaded_mp4.name}</div>
                    <div style="font-size: 14.5px; color: #c8e6c9; margin-top: 4px;">
                        Evaluation Tier: <span style="color: {v_col}; font-weight: bold;">{video_audit.predicted_class}</span> (Score: {video_audit.score} / 4)
                    </div>
                </div>
                """, unsafe_allow_html=True)

                av_table_html = f"""
                <div style="background: rgba(14, 34, 20, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(76, 175, 80, 0.3); border-radius: 10px; overflow: hidden; margin-top: 10px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px; text-align: left;">
                        <thead>
                            <tr style="background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%); color: #ffffff;">
                                <th style="padding: 12px; width: 120px; text-align: center;">Modality</th>
                                <th style="padding: 12px; width: 340px;">Matched Rubric Condition</th>
                                <th style="padding: 12px;">Auditor Observations & Evidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid rgba(76, 175, 80, 0.15);">
                                <td style="padding: 12px; text-align: center; font-weight: bold; color: #a5d6a7;">🎙️ Audio Tone</td>
                                <td style="padding: 12px; font-weight: 600; color: #ffffff;">{video_audit.audio_feature_matched}</td>
                                <td style="padding: 12px; font-size: 13.5px; color: #c8e6c9; line-height: 1.45;">{video_audit.audio_evidence}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(76, 175, 80, 0.15);">
                                <td style="padding: 12px; text-align: center; font-weight: bold; color: #a5d6a7;">📹 Video & Setup</td>
                                <td style="padding: 12px; font-weight: 600; color: #ffffff;">{video_audit.video_feature_matched}</td>
                                <td style="padding: 12px; font-size: 13.5px; color: #c8e6c9; line-height: 1.45;">{video_audit.video_evidence}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
                st.html(av_table_html)
            else:
                st.info("No video uploaded for AV quality evaluation.")

        with tab_frames:
            if uploaded_mp4 and 'sampled_frames' in locals() and sampled_frames:
                st.markdown(f"#### Sampled Keyframes ({len(sampled_frames)} frames across ~60 mins)")
                cols_per_row = 5 if len(sampled_frames) >= 10 else 4
                for row_idx in range(0, len(sampled_frames), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for col_idx, (ts, b64_img, metrics) in enumerate(sampled_frames[row_idx:row_idx+cols_per_row]):
                        dark_flag = "⚠️ Low Light" if metrics["is_dark"] else "✅ Light OK"
                        cols[col_idx].image(
                            f"data:image/jpeg;base64,{b64_img}", 
                            caption=f"⏱️ {ts} | {dark_flag}"
                        )
            else:
                st.info("Upload a video (.mp4) to inspect sampled keyframes.")

        # Cleanup temporary video file
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)