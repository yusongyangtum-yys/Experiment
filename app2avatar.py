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

# --- 1. Configuration ---

# Ensure secrets exist, otherwise show an error prompt
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing 'OPENAI_API_KEY' in st.secrets")
    st.stop()

api_key_chatbot = st.secrets["OPENAI_API_KEY"]

try:
    client = OpenAI(api_key=api_key_chatbot)
except Exception as e:
    st.error(f"Failed to initialize OpenAI Client: {e}")
    st.stop()

MODEL = "gpt-4o-mini"
MAX_TOKENS = 800 
TEMPERATURE = 0.5   

# --- Prompt Definitions (UNCHANGED) ---

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

# --- 2. Helper Functions ---

def save_to_google_sheets(data_dict):
    """
    Save detailed data to Google Sheets
    """
    try:
        if "gcp_service_account" not in st.secrets:
            return False, "Error: 'gcp_service_account' not found in st.secrets."
        
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        sheet_name = st.secrets.get("sheet_name", "Experiment_Data")
        try:
            sh = gc.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            return False, f"Error: Spreadsheet '{sheet_name}' not found."

        worksheet = sh.sheet1
        
        # Construct the complete data row
        row = [
            str(data_dict.get("uuid")),
            str(data_dict.get("mode")),
            str(data_dict.get("start_time")),
            str(data_dict.get("duration")),
            str(data_dict.get("score")),
            str(data_dict.get("sentiment_score")),
            str(data_dict.get("user_word_count")),
            str(data_dict.get("avg_response_time")),
            str(data_dict.get("turn_count")),
            str(data_dict.get("confusion_rate")),
            str(data_dict.get("dialogue_json"))
        ]
        
        worksheet.append_row(row)
        return True, "Success"
    except Exception as e:
        return False, str(e)

POSITIVE_WORDS = ["good", "great", "excellent", "ready", "yes", "understand", "clear"]
NEGATIVE_WORDS = ["bad", "hard", "don't understand", "no", "confused", "wait", "what?", "difficult"]

class SafeCounter:
    def __init__(self, min_val=-10, max_val=10):
        self.value = 0
        self.min_val = min_val
        self.max_val = max_val
    def increment(self): self.value = min(self.max_val, self.value + 1)
    def decrement(self): self.value = max(self.min_val, self.value - 1)
    def reset(self): self.value = 0

if "sentiment_counter" not in st.session_state: st.session_state.sentiment_counter = SafeCounter()
if "confusion_counter" not in st.session_state: st.session_state.confusion_counter = 0 

def detect_sentiment(user_message):
    """
    Detect sentiment and count the number of times confused
    """
    msg = user_message.lower()
    
    # Sentiment scoring
    for w in POSITIVE_WORDS: 
        if w in msg: st.session_state.sentiment_counter.increment()
    
    # Confusion and negative scoring
    is_confused = False
    for w in NEGATIVE_WORDS: 
        if w in msg: 
            st.session_state.sentiment_counter.decrement()
            is_confused = True
            
    if is_confused:
        st.session_state.confusion_counter += 1

def enforce_token_budget(messages):
    if len(messages) > 60:
        return [messages[0]] + messages[-58:]
    return messages

# --- 3. Logic ---

def handle_bot_response(user_input, chat_container, active_mode):
    # --- Metric: User Response Time Logic ---
    current_time = datetime.datetime.now()
    if st.session_state.last_bot_finish_time:
        time_diff = (current_time - st.session_state.last_bot_finish_time).total_seconds()
        # Filter abnormally long response times (e.g. > 5 mins)
        if time_diff < 300: 
            st.session_state.user_response_times.append(time_diff)

    # --- Metric: User Word Count ---
    if user_input:
        word_count = len(user_input.split())
        st.session_state.user_total_words += word_count
        st.session_state.messages.append({"role": "user", "content": user_input})
    
    with chat_container:
        bot_avatar = "👨‍🏫" if active_mode == "Neutral Mode" else "👩‍🏫"
        
        with st.chat_message("assistant", avatar=bot_avatar):
            chat_placeholder = st.empty()
            
            full_response = ""
            
            # [DEV FEATURE]: Developer skip mechanism
            if user_input.strip() == "/dev_skip":
                full_response = "The session is complete. Score: 10/10."
                chat_placeholder.markdown(full_response)
                # Simulate the correct count for testing
                st.session_state.correct_count = 10
            else:
                try:
                    stream = client.chat.completions.create(
                        model=MODEL,
                        messages=enforce_token_budget(st.session_state.messages),
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=True,
                    )
                    for chunk in stream:
                        txt = chunk.choices[0].delta.content
                        if txt:
                            full_response += txt
                            chat_placeholder.markdown(full_response + "▌")
                except Exception as e:
                    st.error(f"API Error: {e}")
                    return

            # --- Metric: Update Last Bot Finish Time ---
            st.session_state.last_bot_finish_time = datetime.datetime.now()

            if "begin the final exam" in full_response.lower():
                st.session_state.correct_count = 0

            clean_display_response = full_response
            
            if "[CORRECT]" in full_response:
                st.session_state.correct_count += 1
                clean_display_response = full_response.replace("[CORRECT]", "").strip()
            elif "[INCORRECT]" in full_response:
                clean_display_response = full_response.replace("[INCORRECT]", "").strip()
            
            # If not in skip mode, re-render the content without the cursor
            if user_input.strip() != "/dev_skip":
                chat_placeholder.markdown(clean_display_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response}) 
            st.session_state.display_history.append({"role": "assistant", "content": clean_display_response})
            
            # --- Finalization Logic ---
            response_lower = full_response.lower()
            # Compatible with "session is complete" or the Score text forced by "/dev_skip"
            if ("session" in response_lower and "complete" in response_lower) or ("score" in response_lower and "10" in response_lower):
                
                # 1. Calculate all metrics
                final_score = st.session_state.correct_count
                end_time = datetime.datetime.now()
                start_time = st.session_state.session_start_time
                duration_seconds = (end_time - start_time).total_seconds()
                
                # Sentiment score
                sentiment_val = st.session_state.sentiment_counter.value
                
                # Average response time
                if len(st.session_state.user_response_times) > 0:
                    avg_resp_time = statistics.mean(st.session_state.user_response_times)
                else:
                    avg_resp_time = 0
                
                # Number of turns
                turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
                
                # Confusion rate
                confusion_rate = 0
                if turn_count > 0:
                    confusion_rate = st.session_state.confusion_counter / turn_count
                
                # Full dialogue JSON
                dialogue_dump = json.dumps(st.session_state.messages, ensure_ascii=False)

                st.info(f"📊 Final Score: {final_score}/10 | Time: {int(duration_seconds)}s")
                
                # 2. Prepare data dictionary
                data_payload = {
                    "uuid": st.session_state.subject_id,
                    "mode": active_mode,
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": int(duration_seconds),
                    "score": final_score,
                    "sentiment_score": sentiment_val,
                    "user_word_count": st.session_state.user_total_words,
                    "avg_response_time": round(avg_resp_time, 2),
                    "turn_count": turn_count,
                    "confusion_rate": round(confusion_rate, 2),
                    "dialogue_json": dialogue_dump
                }
                
                # 3. Save
                success, msg = save_to_google_sheets(data_payload)
                
                if success:
                    st.success("✅ Experiment Complete. All metrics saved successfully.")
                    st.balloons()
                    
                    # [Backend Fix Focus]: Get the locked URL directly from session_state
                    target_url = st.session_state.get("post_survey_url", "#")
                    st.markdown("---")
                    st.markdown(f"""
                    <a href="{target_url}" target="_blank" style="text-decoration:none;">
                        <div style="
                            background-color: #FF5722; color: white; padding: 16px; text-align: center;
                            border-radius: 8px; font-size: 18px; margin: 10px 0; font-weight: bold;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        ">
                            📋 Click Here to Start Post-Survey
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Save Failed: {msg}")

# --- 4. Initialization & Setup ---

st.set_page_config(page_title="Psychology Experiment", layout="wide", initial_sidebar_state="collapsed")

# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# --- ID Generation and Mode Assignment ---

# [Backend Fix Focus]: ID generation and forced atomic locking of URLs
if "subject_id" not in st.session_state:
    auto_id = str(uuid.uuid4())[:8]
    st.session_state.subject_id = f"SUB_{auto_id}"
    
    # Updated Entry ID and Base URL
    PRE_SURVEY_ENTRY_ID = "entry.538559089"   
    POST_SURVEY_ENTRY_ID = "entry.596968100"  # <--- Updated to the latest ID
    
    PRE_SURVEY_BASE = "https://docs.google.com/forms/d/e/1FAIpQLSdqNQ8oRvM-kxVTitRXCtGRuQg_oopmegL-koixLQxJVVjayA/viewform"
    POST_SURVEY_BASE = "https://docs.google.com/forms/d/e/1FAIpQLSckI_yCbL5gQu6P7aP-9vRn5BKp7fX8NrBA_z3FmEegIggCTg/viewform"

    # Permanently store the generated URL in session_state
    st.session_state.pre_survey_url = f"{PRE_SURVEY_BASE}?usp=pp_url&{PRE_SURVEY_ENTRY_ID}={st.session_state.subject_id}"
    st.session_state.post_survey_url = f"{POST_SURVEY_BASE}?usp=pp_url&{POST_SURVEY_ENTRY_ID}={st.session_state.subject_id}"

# --- State Control ---
if "pre_survey_completed" not in st.session_state:
    st.session_state.pre_survey_completed = False
if "auto_start_triggered" not in st.session_state:
    st.session_state.auto_start_triggered = False

if "active_mode" not in st.session_state:
    hash_object = hashlib.md5(st.session_state.subject_id.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    if hash_int % 2 == 0:
        st.session_state.active_mode = "Empathy Mode"
    else:
        st.session_state.active_mode = "Neutral Mode"

# --- Metrics Initialization ---
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.datetime.now()

if "user_response_times" not in st.session_state:
    st.session_state.user_response_times = []

if "last_bot_finish_time" not in st.session_state:
    st.session_state.last_bot_finish_time = datetime.datetime.now()

if "user_total_words" not in st.session_state:
    st.session_state.user_total_words = 0

# --- System Prompt Init ---
if "messages" not in st.session_state:
    prompt = SYSTEM_PROMPT_EMPATHY if st.session_state.active_mode == "Empathy Mode" else SYSTEM_PROMPT_NEUTRAL
    st.session_state.messages = [{"role": "system", "content": prompt}]

if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0

# --- 5. Main UI Logic ---

# [Logic Branch 1: If Pre-Survey is not completed, show the landing page]
if not st.session_state.pre_survey_completed:
    st.container().markdown("<br><br>", unsafe_allow_html=True) # Spacer
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🎓 Psychology Learning Experiment")
        st.info("👋 Welcome! Before we begin the session with the AI teacher, please complete a short survey.")
        st.write(f"**Your Participant ID:** `{st.session_state.subject_id}` (Auto-filled)")
        
        # Fix: Get the locked pre-survey link directly from session_state
        pre_url = st.session_state.pre_survey_url
        st.markdown(f"""
        <a href="{pre_url}" target="_blank" style="text-decoration:none;">
            <div style="
                background-color: #4CAF50; color: white; padding: 20px; text-align: center;
                border-radius: 8px; font-size: 18px; margin: 20px 0; font-weight: bold;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                👉 Click Here to Start Pre-Survey
            </div>
        </a>
        """, unsafe_allow_html=True)
        
        st.warning("⚠️ Please keep this tab open. After submitting the Google Form, return here and click the button below. And please keep all the pages open during the whole experiment.")
        
        st.write("---")
        
        # Confirmation button
        if st.button("I have submitted the Pre-Survey"):
            st.session_state.pre_survey_completed = True
            st.rerun()

# [Logic Branch 2: Avatar Interaction Phase]
else:
    st.title("🧠 Psychology Learning Session")

    col_avatar, col_chat = st.columns([1, 2])

    # -------------------------------------------------------------
    # [FIX] Fix WebSocket crash: Load 3D model directly using URL
    # -------------------------------------------------------------
    NEW_MODEL_URL = "https://huggingface.co/Giillm/Avatar/resolve/main/GLB.glb?download=true" 

    html = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <div style="
        display: flex; 
        justify-content: center; 
        align-items: center; 
        height: 540px; 
        background-color: #f0f2f6; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
    ">
        <model-viewer 
            src="{NEW_MODEL_URL}" 
            camera-controls 
            autoplay 
            animation-name="*" 
            shadow-intensity="1" 
            style="width:100%; height:100%;" 
            interaction-prompt="none"
            loading="eager" 
            alt="AI Teacher Avatar"
        >
            <div slot="poster" style="
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 100%; 
                color: #555; 
                font-family: sans-serif;
                flex-direction: column;
            ">
                <div style="font-size: 40px;">⏳</div>
                <div style="margin-top: 10px; font-weight: bold;">Loading AI Teacher...</div>
                <div style="font-size: 12px; color: #888; margin-top: 5px;">(Large file downloading, please wait...)</div>
            </div>
        </model-viewer>
    </div>
    """
    with col_avatar: 
        components.html(html, height=540)

    with col_chat:
        chat_container = st.container(height=520)
        locked_mode = st.session_state.active_mode

        # Display chat history
        with chat_container:
            for msg in st.session_state.display_history:
                avatar = "👩‍🏫" if msg["role"] == "assistant" and locked_mode == "Empathy Mode" else ("👨‍🏫" if msg["role"] == "assistant" else "👤")
                st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

        # Auto-trigger logic
        if len(st.session_state.display_history) == 0:
            trigger_msg = "The student has logged in. Please start Phase 1: Introduction now."
            has_assistant_reply = any(m["role"] == "assistant" for m in st.session_state.messages)
            
            if not has_assistant_reply and not st.session_state.auto_start_triggered:
                st.session_state.auto_start_triggered = True 
                st.session_state.messages.append({"role": "system", "content": trigger_msg})
                st.session_state.last_bot_finish_time = datetime.datetime.now() 
                handle_bot_response("", chat_container, locked_mode)
                st.rerun() 

        # User input
        user_input = st.chat_input("Type your response here...")
        
        if user_input:
            with chat_container:
                st.chat_message("user", avatar="👤").write(user_input)
                st.session_state.display_history.append({"role": "user", "content": user_input})
                
                # Analysis Logic
                detect_sentiment(user_input)
                
                sentiment_val = st.session_state.sentiment_counter.value
                system_instruction = ""
                if locked_mode == "Empathy Mode":
                    if sentiment_val <= -2:
                        system_instruction = f"(System: User discouraged (Score {sentiment_val}). Be extra encouraging!) "
                    elif sentiment_val >= 2:
                        system_instruction = f"(System: User confident. Keep going.) "
                
                final_prompt = system_instruction + user_input
                
                handle_bot_response(final_prompt, chat_container, locked_mode)
