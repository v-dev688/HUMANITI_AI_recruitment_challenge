# Audio-Visual Quality Audit Engine

An automated multimodal evaluation engine for online children's chess coaching sessions. The system evaluates both dialogue transcripts (`.vtt`) and full session recordings (`.mp4`, up to 500 MB+) using Groq LLMs and computer vision keyframe extraction, outputting standardized scores, diagnostics, and tiered classification pools.

## 🚀 Key Features

* **Pedagogical Transcript Audit (`.vtt`):**

  * Extracts 7 atomic pedagogical features + Rapport building check-ins.

  * LLM-as-a-Judge architecture applying strict combinatorial scoring rules (Scores: **1, 3, or 4**).

  * Veto gate enforcement (e.g., missing homework strictly blocks Score 4; missing rapport triggers Score 1).

* **Audio-Visual Quality Audit (`.mp4`):**

  * Memory-safe chunked streaming for recordings up to 500 MB+.

  * Uniform keyframe extraction across \~60-minute timelines using OpenCV frame-seek indexing.

  * Automated visual telemetry (luminance, contrast, dark-frame detection).

  * Evaluates 5 quality tiers: *Excellent (4)*, *Good (3)*, *Average (2)*, *Weak (1)*, and *Known Quality Issue (0)*.

* **Master Quality Pooling:**

  * Combines multimodal signals into an aggregated 8-point scale:

    * $\ge 7$: 🟢 Excellent

    * $5 - 6$: 🟢 Good

    * $3 - 4$: 🟠 Average

    * $< 3$: 🔴 Poor

* **Modern Glassmorphic UI:** Built with Streamlit featuring an ambient 2D blurred green background, interactive preview galleries, and formatted tables.

## 🛠️ Tech Stack & Requirements

* **Python**: 3.10+

* **Frontend / Dashboard**: [Streamlit](https://streamlit.io/)

* **Inference & LLMs**: [Groq](https://groq.com/) via `langchain-groq` (Default: `qwen/qwen3.8-27b`)

* **Schema Validation**: Pydantic v2

* **Computer Vision**: OpenCV (`opencv-python`)

## 📦 Installation & Setup

### 1. Clone the Repository & Navigate to Directory

```
git clone <your-repository-url>
cd HUMANITI_AI_recruitment_challenge

```

### 2. Create and Activate Virtual Environment

**Using Python `venv` (Windows PowerShell):**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

**Using `uv` (Fast alternative):**

```
uv venv
.\.venv\Scripts\Activate.ps1

```

### 3. Install Dependencies

```
python -m pip install -r requirements.txt

```

*(Or with uv: `uv pip install -r requirements.txt`)*

### 4. Configure Environment Variables

Create a `.env` file in the root project folder:

```
GROQ_API_KEY=gsk_your_groq_api_key_here

```

*(You can obtain a free API key from [Groq Console](https://console.groq.com/keys). You can also enter the key directly in the Streamlit sidebar at runtime).*

## ▶️ Running the Application

### 1. Launch the Unified Dual-Modal App (Recommended)

Because video uploads can reach 500 MB, start Streamlit with an elevated upload buffer (1,000 MB):

```
python -m streamlit run unified_app.py --server.maxUploadSize=1000

```

### 2. Launch Specific Sub-Modules (Optional)

* **Transcript-Only Auditor:**

  ```
  python -m streamlit run app.py
  
  ```

* **Video/Audio-Only Auditor:**

  ```
  python -m streamlit run streamlit_app_video.py --server.maxUploadSize=1000
  
  ```

Once launched, your browser will open to:

```
http://localhost:8501

```

## 📋 Evaluation Rubrics

### 1. Pedagogical Transcript Rubric (Score 1, 3, or 4)

| Tier | Score | Triggers / Combinations | 
| ----- | ----- | ----- | 
| **High Value** | **4** | • *(Questions asked individually by name + HW was given)*  • *(Good rapport building + asked questions individually)*  *(Missing HW strictly vetoes and blocks Score 4)* | 
| **Medium Value** | **3** | • *(Included a solid introduction + assigned homework)*  • *(Good intro + included everyone + missed HW)*  • *(Addressed learners during setup and puzzles + gave homework)*  • *(Maintained one-to-one queries + missed Homeworks)*  • *(General interactive flow was present)* | 
| **Low Value** | **1** | • *(Lacked good intro + lacked conclusion + joiner introductions)*  • *(Skipped rapport + missed joiner intro)*  • *(Directly entered class without building rapport)* | 

### 2. Audio-Visual Quality Rubric (Score 0 to 4)

| Quality Class | Score | Audio & Video Criteria | 
| ----- | ----- | ----- | 
| **Excellent** | **4** | **Audio:** Good interaction tone maintained throughout.  **Video:** Adequate tone, acceptable background, framing, and lighting. | 
| **Good** | **3** | **Audio:** Balanced delivery with reasonable interaction.  **Video:** Branded 8x8 background properly fixed, centered framing. | 
| **Average** | **2** | **Audio:** Standard delivery without major vocal issues.  **Video:** Virtual background clarity problems or insufficient room lighting. | 
| **Weak** | **1** | **Audio:** Subdued tone or lacks warm opening engagement.  **Video:** Virtual background maintained, but camera was turned off during teaching. | 
| **Known Quality Issue** | **0** | **Audio:** Needed happier, warmer communication style.  **Video:** Virtual background missing entirely or tutor's face not visible on camera. | 

## 📂 Project Structure

```
├── .env                         # API keys (GROQ_API_KEY)
├── requirements.txt             # Project dependencies
├── unified_app.py               # Combined dual-modal Streamlit application
├── app.py                       # Transcript-only evaluation dashboard
├── streamlit_app_video.py       # Audio-Visual video evaluation dashboard
├── README.md                    # Project documentation

```

## 💡 Troubleshooting

* **`ModuleNotFoundError: No module named 'streamlit'`**:
  Ensure your terminal points to the activated virtual environment and launch using `python -m streamlit run unified_app.py`.

* **`AxiosError: Request failed with status code 413` (File too large)**:
  Remember to pass `--server.maxUploadSize=1000` when executing the `streamlit run` command.

* **Groq Rate Limits (`429 Rate Limit`)**:
  The app samples compact payloads (\~3,000 tokens) to minimize token consumption. Allow 5–10 seconds between large batch uploads.
