import os
from typing import List, Literal, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Allowed states per feature
FeatureState = Literal["Yes", "No", "Blank"]

# =====================================================================
# 1. Pydantic LLM Output Schema
# =====================================================================
class FeatureAuditResult(BaseModel):
    # Feature 1: Individual Questioning
    f1_individual_questions: FeatureState = Field(
        description="Yes: asked questions individually / Questions asked individually by name / included everyone / Maintained one to one queries. No: Generic group questions only. Blank: Neutral."
    )
    f1_evidence: str = Field(description="Direct transcript quote or observation.")

    # Feature 2: Homework Assignment
    f2_homework_given: FeatureState = Field(
        description="Yes: HW was given / assigned homework / gave homework. No: missed HW / class ended without homework. Blank: Not evaluated."
    )
    f2_evidence: str = Field(description="Direct transcript quote or observation.")

    # Rapport Building (Embedded in your criteria 2 & Low rules)
    f_rapport_building: FeatureState = Field(
        description="Yes: Good rapport building (informal check-ins, personal banter). No: Skipped rapport / Directly entered class without building rapport. Blank: Minimal/standard greeting."
    )
    f_rapport_evidence: str = Field(description="Direct transcript quote or observation.")

    # Feature 3: Introduction Quality
    f3_solid_intro: FeatureState = Field(
        description="Yes: Included a solid introduction / Good intro. No: Lacked good intro. Blank: Neutral."
    )
    f3_evidence: str = Field(description="Direct transcript quote or observation.")

    # Feature 4: Setup & Puzzles Engagement
    f4_setup_and_puzzles: FeatureState = Field(
        description="Yes: Addressed learners during setup and puzzles. No: Silent or left learners unaddressed during exercises. Blank: No setup/puzzle phase observed."
    )
    f4_evidence: str = Field(description="Direct transcript quote or observation.")

    # Feature 5: General Interactive Flow
    f5_interactive_flow: FeatureState = Field(
        description="Yes: General interactive flow was present (continuous 2-way dialogue). No: Long tutor monologues / passive silence. Blank: Neutral."
    )
    f5_evidence: str = Field(description="Direct transcript quote or observation.")

    # Feature 6: Conclusion Quality
    f6_conclusion_quality: FeatureState = Field(
        description="No: Lacked conclusion (abrupt end, no wrap-up). Yes: Clear wrap-up provided. Blank: Standard end."
    )
    f6_evidence: str = Field(description="Direct transcript quote or observation.")

    # Feature 7: Joiner Introductions
    f7_joiner_intros: FeatureState = Field(
        description="Yes: joiner introductions (allowed time for new student to introduce themselves). No: missed joiner intro / lacked joiner intros. Blank: No new joiners in session."
    )
    f7_evidence: str = Field(description="Direct transcript quote or observation.")

    students_called: List[str] = Field(
        description="Names of individual students called upon during the class."
    )


# =====================================================================
# 2. System Prompt Defining Observable Features
# =====================================================================
EXTRACTION_SYSTEM_PROMPT = """
You are an expert pedagogical auditor for online children's chess classes.
Analyze the provided transcript and extract the state for each of the 7 features strictly as 'Yes', 'No', or 'Blank', providing exact transcript evidence quotes:

1. Feature 1 (Individual Questions):
   - 'Yes': asked questions individually / Questions asked individually by name / included everyone / Maintained one to one queries.
   - 'No': Failed to query individually, addressed only the whole room, or left learners out.
   - 'Blank': Neutral / not evaluated.

2. Feature 2 (Homework):
   - 'Yes': HW was given / assigned homework / gave homework.
   - 'No': missed HW / should give HW (session ended without assigning homework tasks).
   - 'Blank': Not addressed.

3. Rapport Building:
   - 'Yes': Good rapport building (warm casual check-ins, non-chess icebreakers, friendly banter).
   - 'No': Skipped rapport / Directly entered class without building rapport (dived straight into revision/moves).
   - 'Blank': Standard neutral greeting.

4. Feature 3 (Introduction):
   - 'Yes': Included a solid introduction / Good intro (clear orientation of topics/goals).
   - 'No': Lacked good intro.
   - 'Blank': Neutral.

5. Feature 4 (Setup & Puzzles):
   - 'Yes': Addressed learners during setup and puzzles (actively coached learners while setting up or solving).
   - 'No': Passive, left students waiting in silence without coaching.
   - 'Blank': No setup or puzzle phase.

6. Feature 5 (Interactive Flow):
   - 'Yes': General interactive flow was present (active back-and-forth dialogue throughout).
   - 'No': Long monologue by tutor, one-sided interaction.
   - 'Blank': Neutral.

7. Feature 6 (Conclusion):
   - 'No': Lacked conclusion (abrupt drop-off without summarizing or closing).
   - 'Yes': Proper session wrap-up given.
   - 'Blank': Standard exit.

8. Feature 7 (Joiner Introductions):
   - 'Yes': joiner introductions (facilitated dedicated time for new students to introduce themselves).
   - 'No': missed joiner intro / lacked joiner introductions (new student was present but not invited to introduce themselves).
   - 'Blank': No new joiners attended.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.0
)

structured_extractor = llm.with_structured_output(FeatureAuditResult)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", EXTRACTION_SYSTEM_PROMPT),
    ("human", "Class Transcript:\n\n{transcript_text}")
])

extraction_chain = extraction_prompt | structured_extractor


# =====================================================================
# 3. Deterministic Scoring Logic (Your Exact Combinations)
# =====================================================================
def compute_score_and_rule(audit: FeatureAuditResult) -> Tuple[int, str, str]:
    f1 = audit.f1_individual_questions
    f2 = audit.f2_homework_given
    rap = audit.f_rapport_building
    f3 = audit.f3_solid_intro
    f4 = audit.f4_setup_and_puzzles
    f5 = audit.f5_interactive_flow
    f6 = audit.f6_conclusion_quality
    f7 = audit.f7_joiner_intros

    # -----------------------------------------------------------------
    # High Value Feature Combination = 4
    # (Questions asked individually by name + HW was given) OR
    # (Good rapport building + asked questions individually)
    # -----------------------------------------------------------------
    if (f1 == "Yes" and f2 == "Yes") or (rap == "Yes" and f1 == "Yes"):
        if f1 == "Yes" and f2 == "Yes":
            rule = "Questions asked individually by name + HW was given"
        else:
            rule = "Good rapport building + asked questions individually"
        return 4, "High Value", rule

    # -----------------------------------------------------------------
    # Low Value Features Combination = 1
    # (Lacked good intro + lacked conclusion + joiner introductions) OR
    # (Skipped rapport + missed joiner intro) OR
    # (Directly entered class without building rapport.)
    # -----------------------------------------------------------------
    if (
        (f3 == "No" and f6 == "No" and f7 == "Yes")
        or (rap == "No" and f7 == "No")
        or (rap == "No")
    ):
        if f3 == "No" and f6 == "No" and f7 == "Yes":
            rule = "Lacked good intro + lacked conclusion + joiner introductions"
        elif rap == "No" and f7 == "No":
            rule = "Skipped rapport + missed joiner intro"
        else:
            rule = "Directly entered class without building rapport"
        return 1, "Low Value", rule

    # -----------------------------------------------------------------
    # Medium Value Feature Combination = 3
    # (Included a solid introduction + assigned homework) OR
    # (Good intro + included everyone + missed HW) OR
    # (Addressed learners during setup and puzzles + gave homework) OR
    # (Maintained one to one queries + missed Homeworks) OR
    # (General interactive flow was present)
    # -----------------------------------------------------------------
    if f3 == "Yes" and f2 == "Yes":
        rule = "Included a solid introduction + assigned homework"
    elif f3 == "Yes" and f1 == "Yes" and f2 == "No":
        rule = "Good intro + included everyone + missed HW"
    elif f4 == "Yes" and f2 == "Yes":
        rule = "Addressed learners during setup and puzzles + gave homework"
    elif f1 == "Yes" and f2 == "No":
        rule = "Maintained one to one queries + missed Homeworks"
    else:
        rule = "General interactive flow was present"

    return 3, "Medium Value", rule


# =====================================================================
# 4. Helper & Formatter
# =====================================================================
def clean_vtt(raw_text: str) -> str:
    lines = []
    for line in raw_text.splitlines():
        t = line.strip()
        if not t or t.startswith("WEBVTT") or t.isdigit() or "-->" in t:
            continue
        lines.append(t)
    return "\n".join(lines)

def run_pedagogical_audit(vtt_file_path: str):
    with open(vtt_file_path, "r", encoding="utf-8", errors="ignore") as f:
        cleaned = clean_vtt(f.read())

    # Step 1: LLM extracts features
    extracted: FeatureAuditResult = extraction_chain.invoke({"transcript_text": cleaned})

    # Step 2: Deterministic scoring
    score, tier, matched_rule = compute_score_and_rule(extracted)

    # Step 3: Print Table & Score
    print("\n" + "=" * 88)
    print("                      CLASS PEDAGOGICAL FEATURE EXTRACTION TABLE")
    print("=" * 88)
    print(f"{'#':<3} | {'Feature Metric':<32} | {'State':<7} | {'Observation / Quoted Evidence'}")
    print("-" * 88)

    rows = [
        ("1", "Questions Asked Individually", extracted.f1_individual_questions, extracted.f1_evidence),
        ("2", "Homework Assignment", extracted.f2_homework_given, extracted.f2_evidence),
        ("—", "Rapport Building", extracted.f_rapport_building, extracted.f_rapport_evidence),
        ("3", "Introduction Quality", extracted.f3_solid_intro, extracted.f3_evidence),
        ("4", "Setup & Puzzles Engagement", extracted.f4_setup_and_puzzles, extracted.f4_evidence),
        ("5", "General Interactive Flow", extracted.f5_interactive_flow, extracted.f5_evidence),
        ("6", "Session Conclusion", extracted.f6_conclusion_quality, extracted.f6_evidence),
        ("7", "Joiner Introductions", extracted.f7_joiner_intros, extracted.f7_evidence),
    ]

    for num, name, state, ev in rows:
        print(f"{num:<3} | {name:<32} | {state:<7} | {ev}")

    print("=" * 88)
    print(f"Students Addressed Individually : {', '.join(extracted.students_called)}")
    print("-" * 88)
    print(f"OUTPUT SCORE                    : {score}  ({tier})")
    print(f"MATCHED COMBINATION RULE        : \"({matched_rule})\"")
    print("=" * 88 + "\n")

    return extracted, score, matched_rule