import streamlit as st
import os
from openai import OpenAI
import streamlit.components.v1 as components
import requests
import base64
import json
import datetime
import gspread
from google.oauth2.service_account import Credentials
import uuid
import hashlib
import statistics
import time

# =========================================================
# 1. Configuration
# =========================================================

api_key_chatbot = st.secrets["OPENAI_API_KEY"]

try:
    client = OpenAI(api_key=api_key_chatbot)
except Exception as e:
    st.error(f"Failed to initialize OpenAI Client: {e}")
    st.stop()

MODEL = "gpt-4o-mini"
MAX_TOKENS = 800
TEMPERATURE = 0.5

# =========================================================
# 2. Prompts (UNCHANGED)
# =========================================================
SYSTEM_PROMPT_EMPATHY = """
You are Sophia, a supportive, warm, and patient psychology teacher.

Your goal is to teach exactly these 3 topics:
1. Classical Conditioning
2. Operant Conditioning
3. Memory Types

────────────────────────────────
OUTPUT CONTROL: MEANINGFUL SEGMENTS
────────────────────────────────
Each assistant message must cover **ONE COMPLETE LOGICAL SEGMENT**.

Instead of stopping after every sentence, you should:
1. **Explain a concept thoroughly** (including definition and key details).
2. **Provide a relevant example** immediately to help understanding.
3. Keep the length **moderate (approx. 100-150 words)** to ensure depth.

**Allowed Structure per Message (Teaching Phase):**
[Explanation of Concept] + [Real-world Example] + [Short Pause Question]

**IMPORTANT: SCORING TAGS (CRITICAL)**
- Whenever the user answers a **Mini-Quiz** or **Final Exam** question:
  - If CORRECT: Start your response with **"[CORRECT] "** (including brackets).
  - If INCORRECT: Start your response with **"[INCORRECT] "** (including brackets).
  - Example: "[CORRECT] That's wonderful! You got it right."
  - These tags are HIDDEN from the user but used for system scoring. YOU MUST USE THEM.

**Rules:**
- Do NOT ask checking questions in the middle of an explanation.
- Do NOT break a single concept into tiny pieces. Deliver the whole idea.
- ONLY stop and ask a checking question when you have finished a complete segment.

End your response with a gentle check-in:
- "Does this explanation make sense to you?"
- "How does that example sound?"
- "Ready to move on?"

────────────────────────────────
TEACHING STYLE (EMPATHY)
────────────────────────────────
- Be warm, encouraging, and emotionally supportive.
- Use gentle language.
- Praise effort, not just correctness.

────────────────────────────────
TEACHING FLOW
────────────────────────────────

PHASE 1: INTRODUCTION
- Introduce yourself warmly.
- List the 3 topics.
- Ask if the student is ready to begin Topic 1.
- Stop and wait.

PHASE 2: TOPIC LOOP (repeat for ALL 3 topics)
1. **Teach a Sub-Topic**: Explain a major part of the topic (e.g., Definition + Experiment) fully in one message.
2. Stop and ask for understanding.
3. Wait for response.
4. **Teach the Next Part**: Explain the next logical segment (e.g., Key Principles + Application).
5. Stop and ask.
6. (Repeat until topic is covered).
7. **Mini-Quiz**: Ask EXACTLY ONE multiple-choice question for this topic.
8. Wait for answer -> Give warm feedback starting with [CORRECT] or [INCORRECT].
9. Ask if ready for the next topic.

PHASE 3: FINAL EXAM
- Trigger ONLY after all 3 topics are finished.
- Say: "Now we will begin the final exam. I will ask 10 questions one by one."
- Exam rules:
  - Ask ONE multiple-choice question at a time.
  - STOP and wait for answer.
  - Give empathetic feedback (Must start with [CORRECT] or [INCORRECT]).
  - Move to next question.
- After Question 10:
  - Output EXACTLY: "The session is complete."
  - (Do not report the score yourself; the system will display the accurate count based on your tags.)
"""

SYSTEM_PROMPT_NEUTRAL = """
You are a neutral, factual AI instructor.

Your task is to teach exactly these 3 psychology topics:
1. Classical Conditioning (Pavlov)
2. Operant Conditioning (Skinner)
3. Memory Types

────────────────────────────────
OUTPUT CONTROL: COMPREHENSIVE BLOCKS
────────────────────────────────
Each assistant message must deliver **ONE COMPLETE INFORMATIONAL BLOCK**.

Do not fragment information. Your goal is efficiency and completeness.
1. **Define and Describe**: Explain the concept or procedure clearly.
2. **Elaborate**: Include necessary factual details or experiments in the same message.
3. Keep length **moderate (approx. 100-150 words)**.

**Allowed Structure per Message (Teaching Phase):**
[Factual Explanation] + [Details/Experiment] + [Status Check]

**IMPORTANT: SCORING TAGS (CRITICAL)**
- Whenever the user answers a **Mini-Quiz** or **Final Exam** question:
  - If CORRECT: Start response with **"[CORRECT] "**
  - If INCORRECT: Start response with **"[INCORRECT] "**
  - Example: "[CORRECT] Correct. The answer is A."

**Rules:**
- Do NOT interrupt the flow with questions until the block is complete.
- Ensure the explanation is self-contained and academic.
- End with a neutral status check.

End your response with a short check:
- "Is this concept clear?"
- "Shall I proceed to the next section?"

────────────────────────────────
TEACHING STYLE (NEUTRAL)
────────────────────────────────
- Maintain objective, academic tone.
- No emotional language.
- Be precise and factual.

────────────────────────────────
TEACHING FLOW
────────────────────────────────

PHASE 1: INTRODUCTION
- Introduce yourself briefly.
- List the 3 topics.
- Ask if ready to start Topic 1.
- Stop and wait.

PHASE 2: TOPIC LOOP (repeat for ALL 3 topics)
1. **Teach Section A**: Explain the first major section of the topic comprehensively.
2. Stop and ask if clear.
3. Wait for response.
4. **Teach Section B**: Explain the next major section (e.g., Applications/Nuances).
5. Stop and ask.
6. (Repeat until topic is covered).
7. **Mini-Quiz**: Ask EXACTLY ONE multiple-choice question.
8. Wait for answer -> Give factual feedback starting with [CORRECT] or [INCORRECT].
9. Proceed to next topic.

PHASE 3: FINAL EXAM
- Start ONLY after Topic 3 is finished.
- Say: "We will now begin the final exam consisting of 10 multiple-choice questions."
- Rules:
  - Ask ONE question at a time.
  - STOP and wait for input.
  - Give factual feedback (Must start with [CORRECT] or [INCORRECT]).
  - Continue until Question 10.
- After Question 10:
  - Output EXACTLY: "The session is complete."
  - (Do not report the score yourself; the system will display the accurate count.)
"""


# =========================================================
# 3. Helper Functions (UNCHANGED)
# =========================================================

def save_to_google_sheets(data_dict):
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        sh = gc.open(st.secrets["sheet_name"])
        worksheet = sh.sheet1

        row = [
            data_dict["uuid"], data_dict["mode"], data_dict["start_time"],
            data_dict["duration"], data_dict["score"],
            data_dict["sentiment_score"], data_dict["user_word_count"],
            data_dict["avg_response_time"], data_dict["turn_count"],
            data_dict["confusion_rate"], data_dict["dialogue_json"]
        ]
        worksheet.append_row(row)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# =========================================================
# 4. 🔧 MODIFIED: Stable Bot Response (NO STREAMING)
# =========================================================

def handle_bot_response(user_input, chat_container, active_mode):
    current_time = datetime.datetime.now()
    if st.session_state.last_bot_finish_time:
        diff = (current_time - st.session_state.last_bot_finish_time).total_seconds()
        if diff < 300:
            st.session_state.user_response_times.append(diff)

    if user_input:
        st.session_state.user_total_words += len(user_input.split())
        st.session_state.messages.append({"role": "user", "content": user_input})

    with chat_container:
        avatar = "👩‍🏫" if active_mode == "Empathy Mode" else "👨‍🏫"
        with st.chat_message("assistant", avatar=avatar):
            placeholder = st.empty()

            # 🔧 MODIFIED: NO stream=True
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages[-60:],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
            except Exception as e:
                st.error(f"API Error: {e}")
                return

            full_response = resp.choices[0].message.content

            # 🔧 Sentence-level "safe typing effect"
            rendered = ""
            for sentence in full_response.split(". "):
                rendered += sentence.strip() + ". "
                placeholder.markdown(rendered)
                time.sleep(0.08)  # 微弱打字感，不触发 WS 问题

            st.session_state.last_bot_finish_time = datetime.datetime.now()

            clean = full_response
            if "[CORRECT]" in full_response:
                st.session_state.correct_count += 1
                clean = full_response.replace("[CORRECT]", "").strip()
            elif "[INCORRECT]" in full_response:
                clean = full_response.replace("[INCORRECT]", "").strip()

            placeholder.markdown(clean)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.display_history.append({"role": "assistant", "content": clean})

            # ===== Session End Logic (UNCHANGED) =====
            if "the session is complete" in full_response.lower():
                end_time = datetime.datetime.now()
                duration = (end_time - st.session_state.session_start_time).total_seconds()

                turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
                avg_resp = statistics.mean(st.session_state.user_response_times) if st.session_state.user_response_times else 0
                confusion_rate = st.session_state.confusion_counter / turn_count if turn_count else 0

                payload = {
                    "uuid": st.session_state.subject_id,
                    "mode": active_mode,
                    "start_time": st.session_state.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": int(duration),
                    "score": st.session_state.correct_count,
                    "sentiment_score": st.session_state.sentiment_counter.value,
                    "user_word_count": st.session_state.user_total_words,
                    "avg_response_time": round(avg_resp, 2),
                    "turn_count": turn_count,
                    "confusion_rate": round(confusion_rate, 2),
                    "dialogue_json": json.dumps(st.session_state.messages, ensure_ascii=False)
                }

                success, msg = save_to_google_sheets(payload)
                if success:
                    st.success("✅ Experiment completed and saved.")
                else:
                    st.error(f"Save failed: {msg}")

# =========================================================
# 5. UI & State Init (UNCHANGED except comments)
# =========================================================

st.set_page_config(page_title="Psychology Experiment", layout="wide")

if "subject_id" not in st.session_state:
    st.session_state.subject_id = f"SUB_{uuid.uuid4().hex[:8]}"

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.datetime.now()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT_EMPATHY
    }]

if "display_history" not in st.session_state:
    st.session_state.display_history = []

if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0

if "user_response_times" not in st.session_state:
    st.session_state.user_response_times = []

if "last_bot_finish_time" not in st.session_state:
    st.session_state.last_bot_finish_time = datetime.datetime.now()

if "user_total_words" not in st.session_state:
    st.session_state.user_total_words = 0

# =========================================================
# 6. Main UI
# =========================================================

st.title("🧠 Psychology Learning Session")

col_avatar, col_chat = st.columns([1, 2])

with col_avatar:
    components.html("""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <model-viewer src="https://github.com/yusongyangtum-yys/Avatar/releases/download/avatar/GLB.glb"
        camera-controls autoplay style="width:100%;height:520px;"></model-viewer>
    """, height=540)

with col_chat:
    chat_container = st.container(height=520)

    for msg in st.session_state.display_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Type your response here...")
    if user_input:
        st.chat_message("user").write(user_input)
        handle_bot_response(user_input, chat_container, "Empathy Mode")
