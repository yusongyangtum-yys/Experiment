import streamlit as st
import os
from openai import OpenAI
import streamlit.components.v1 as components 
import requests 
import base64
import json
import datetime
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import asyncio
import edge_tts
import uuid 
import re # 新增：用于正则提取标记

# --- 1. Configuration ---

api_key_chatbot = st.secrets["OPENAI_API_KEY"]

try:
    client = OpenAI(api_key=api_key_chatbot)
except Exception as e:
    st.error(f"Failed to initialize OpenAI Client: {e}")
    st.stop()

MODEL = "gpt-4o-mini"
MAX_TOKENS = 800 
TEMPERATURE = 0.5   

# --- TTS Voice Configuration ---
VOICE_EMPATHY = "en-US-AnaNeural" 
VOICE_NEUTRAL = "en-US-ChristopherNeural" 

# --- Prompt Definitions (Updated for Python Scoring) ---

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

Allowed Structure (Teaching Phase):
[Explanation] + [Example] + [Check Question]

**IMPORTANT: SCORING TAGS (CRITICAL)**
- Whenever the user answers a **Mini-Quiz** or **Final Exam** question:
  - If CORRECT: Start your response with **"[CORRECT] "** (including brackets).
  - If INCORRECT: Start your response with **"[INCORRECT] "** (including brackets).
  - Example: "[CORRECT] That's wonderful! You got it right."
  - Example: "[INCORRECT] Not quite. The correct answer is..."
- These tags are HIDDEN from the user but used for scoring. YOU MUST USE THEM.

────────────────────────────────
TEACHING FLOW
────────────────────────────────

PHASE 1: INTRODUCTION
- Introduce yourself warmly.
- List 3 topics.
- Ask if ready for Topic 1.

PHASE 2: TOPIC LOOP (repeat for ALL 3 topics)
1. **Teach**: Explain concept + example. Stop and ask.
2. (Continue teaching parts...)
3. **Mini-Quiz**: Ask 1 multiple-choice question.
4. Wait for answer.
5. **Feedback**: MUST start with [CORRECT] or [INCORRECT]. Give warm feedback.
6. Ask if ready for next topic.

PHASE 3: FINAL EXAM
- Say: "Now we will begin the final exam. I will ask 10 questions one by one."
- Loop 10 times:
  - Ask ONE question.
  - Wait for answer.
  - **Feedback**: MUST start with [CORRECT] or [INCORRECT].
- After Question 10:
  - Output ONLY: "The session is complete." (Python will handle the score display).
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

Allowed Structure (Teaching Phase):
[Explanation] + [Details] + [Status Check]

**IMPORTANT: SCORING TAGS (CRITICAL)**
- Whenever the user answers a **Mini-Quiz** or **Final Exam** question:
  - If CORRECT: Start response with **"[CORRECT] "**
  - If INCORRECT: Start response with **"[INCORRECT] "**
  - Example: "[CORRECT] Correct. The answer is A."

────────────────────────────────
TEACHING FLOW
────────────────────────────────

PHASE 1: INTRODUCTION
- Introduce yourself.
- List 3 topics.
- Ask if ready for Topic 1.

PHASE 2: TOPIC LOOP (repeat for ALL 3 topics)
1. **Teach**: Explain concept. Stop and ask.
2. (Continue teaching...)
3. **Mini-Quiz**: Ask 1 multiple-choice question.
4. Wait for answer.
5. **Feedback**: MUST start with [CORRECT] or [INCORRECT].
6. Proceed to next topic.

PHASE 3: FINAL EXAM
- Say: "We will now begin the final exam consisting of 10 multiple-choice questions."
- Loop 10 times:
  - Ask ONE question.
  - Wait for answer.
  - **Feedback**: MUST start with [CORRECT] or [INCORRECT].
- After Question 10:
  - Output ONLY: "The session is complete."
"""

# --- 2. Javascript Hack ---
def stop_previous_audio():
    js_code = """
        <script>
            var audios = document.getElementsByTagName('audio');
            for(var i = 0; i < audios.length; i++){
                audios[i].pause();
                audios[i].currentTime = 0;
                audios[i].remove(); 
            }
        </script>
    """
    components.html(js_code, height=0, width=0)

stop_previous_audio()

# --- 3. Helper Functions ---

def save_to_google_sheets(subject_id, chat_history, mode, audio_enabled, score_summary="N/A"):
    """保存数据到 Google Sheets"""
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
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        audio_status = "On" if audio_enabled else "Off"
        
        row = [subject_id, timestamp, mode, audio_status, score_summary]
        worksheet.append_row(row)
        
        return True, "Success"
    except Exception as e:
        return False, str(e)

POSITIVE_WORDS = ["good", "great", "excellent", "ready", "yes", "understand", "clear"]
NEGATIVE_WORDS = ["bad", "hard", "don't understand", "no", "confused", "wait"]

class SafeCounter:
    def __init__(self, min_val=-10, max_val=10):
        self.value = 0
        self.min_val = min_val
        self.max_val = max_val
    def increment(self): self.value = min(self.max_val, self.value + 1)
    def decrement(self): self.value = max(self.min_val, self.value - 1)
    def reset(self): self.value = 0
    # 新增：答对计数器
    def add_correct(self):
        if "correct_count" not in st.session_state:
            st.session_state.correct_count = 0
        st.session_state.correct_count += 1
    def get_correct(self):
        return st.session_state.get("correct_count", 0)

if "sentiment_counter" not in st.session_state: st.session_state.sentiment_counter = SafeCounter()
if "correct_count" not in st.session_state: st.session_state.correct_count = 0

def detect_sentiment(user_message):
    msg = user_message.lower()
    for w in POSITIVE_WORDS: 
        if w in msg: st.session_state.sentiment_counter.increment()
    for w in NEGATIVE_WORDS: 
        if w in msg: st.session_state.sentiment_counter.decrement()

def enforce_token_budget(messages):
    # 保持较大的上下文窗口
    if len(messages) > 60:
        return [messages[0]] + messages[-58:]
    return messages

# --- 4. TTS Logic ---

async def edge_tts_generate(text, voice, rate):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def play_audio_full(text, active_mode, enable_audio):
    if not text.strip() or not enable_audio:
        return
        
    if active_mode == "Neutral Mode":
        voice = VOICE_NEUTRAL
        current_rate = "+10%" 
    else:
        voice = VOICE_EMPATHY
        current_rate = "+25%" 
    
    clean_text = text.replace("*", "").replace("#", "").replace("`", "")

    if "audio_container" in st.session_state:
        st.session_state.audio_container.empty()

    try:
        with st.spinner(f"🔊 Generating audio..."):
            audio_bytes = asyncio.run(edge_tts_generate(clean_text, voice, current_rate))
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return

    if not audio_bytes:
        return

    b64 = base64.b64encode(audio_bytes).decode()
    unique_id = f"audio_{uuid.uuid4()}" 
    
    md = f"""
        <audio autoplay="true" style="display:none;" id="{unique_id}">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    
    st.session_state.audio_container = st.empty()
    st.session_state.audio_container.markdown(md, unsafe_allow_html=True)

# --- 5. Logic ---

def handle_bot_response(user_input, chat_container, active_mode, enable_audio):
    if user_input: 
        st.session_state.messages.append({"role": "user", "content": user_input})
    
    with chat_container:
        bot_avatar = "👨‍🏫" if active_mode == "Neutral Mode" else "👩‍🏫"
        
        with st.chat_message("assistant", avatar=bot_avatar):
            chat_placeholder = st.empty()
            
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=enforce_token_budget(st.session_state.messages),
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
            except Exception as e:
                st.error(f"API Error: {e}")
                return

            full_response = ""
            for chunk in stream:
                txt = chunk.choices[0].delta.content
                if txt:
                    full_response += txt
                    chat_placeholder.markdown(full_response + "▌")
            
            # --- 关键修改：处理隐形标签 ---
            # 1. 检查是否有 [CORRECT] 或 [INCORRECT]
            # 2. 如果有，更新 Python 计数器
            # 3. 将标签从显示文本中移除，让用户看不到，但逻辑能捕捉到
            
            clean_display_response = full_response
            
            if "[CORRECT]" in full_response:
                st.session_state.correct_count += 1
                clean_display_response = full_response.replace("[CORRECT]", "").strip()
            elif "[INCORRECT]" in full_response:
                clean_display_response = full_response.replace("[INCORRECT]", "").strip()
            
            # 重新渲染不带标签的干净文本
            chat_placeholder.markdown(clean_display_response)
            
            # 保存到 history (保存干净文本)
            st.session_state.messages.append({"role": "assistant", "content": full_response}) # 存原始带标签的给 LLM 保持上下文
            st.session_state.display_history.append({"role": "assistant", "content": clean_display_response})
            
            # --- 自动保存与结算逻辑 ---
            if "session is complete" in full_response.lower():
                # 使用 Python 统计的分数
                final_score = st.session_state.correct_count
                
                # 总题数 (3个Topic Quiz + 10个Final Exam = 13题) 
                # 或者你可以只算 Final Exam，但因为代码是从头跑的，correct_count 会包含 Topic Quiz。
                # 建议：在 Final Exam 开始时重置计数器？
                # 简化方案：只显示 "Total Correct Answers: X"
                
                # 修正显示：在最后追加 Python 算出的真实分数
                st.info(f"📊 Final Score Calculation: You answered {final_score} questions correctly.")
                summary_text = f"Completed - Score: {final_score}"
                
                success, msg = save_to_google_sheets(
                    st.session_state.subject_id, 
                    st.session_state.display_history, 
                    active_mode, 
                    enable_audio, 
                    summary_text
                )
                if success:
                    st.success("✅ Session Data Saved!")
                    st.balloons()
                else:
                    st.error(f"❌ Save Failed: {msg}")

            # 播放音频 (播放干净文本)
            play_audio_full(clean_display_response, active_mode, enable_audio)

def reset_experiment_logic():
    st.session_state.display_history = []
    st.session_state.sentiment_counter.reset()
    st.session_state.correct_count = 0 # 重置分数
    st.session_state.experiment_started = False
    st.session_state.audio_container = st.empty()
    if "active_mode" in st.session_state:
        del st.session_state.active_mode
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT_EMPATHY}]

# --- 6. Streamlit UI ---

YOUR_GLB_URL = "https://github.com/yusongyangtum-yys/Avatar/releases/download/avatar/GLB.glb"
LOCAL_GLB_PATH = "cached_model.glb"

@st.cache_resource
def get_glb_base64(url, local_path):
    if not os.path.exists(local_path):
        try:
            r = requests.get(url, stream=True)
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except: return None
    with open(local_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

st.set_page_config(page_title="Sophia Experiment", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT_EMPATHY}]
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "subject_id" not in st.session_state:
    st.session_state.subject_id = ""
if "experiment_started" not in st.session_state:
    st.session_state.experiment_started = False
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Empathy Mode" 
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0

# --- Sidebar ---
with st.sidebar:
    st.title("🔧 Settings")
    input_id = st.text_input("Enter Subject ID:", value=st.session_state.subject_id)
    if input_id != st.session_state.subject_id:
        st.session_state.subject_id = input_id
    
    # 语音开关
    enable_audio = st.checkbox("🔊 Enable Audio", value=True)
    
    if not st.session_state.experiment_started:
        if st.button("🚀 Start Experiment", type="primary"):
            if st.session_state.subject_id.strip():
                st.session_state.experiment_started = True
                selected = st.session_state.get("mode_selection", "Empathy Mode")
                st.session_state.active_mode = selected
                st.session_state.correct_count = 0 # 重置分数
                prompt = SYSTEM_PROMPT_EMPATHY if selected == "Empathy Mode" else SYSTEM_PROMPT_NEUTRAL
                st.session_state.messages = [{"role": "system", "content": prompt}]
                st.rerun()
            else:
                st.error("⚠️ Enter Subject ID first.")
    else:
        st.success(f"Running: {st.session_state.subject_id}")
        st.info(f"Mode: {st.session_state.active_mode}") 
    
    st.markdown("---")
    
    mode_disabled = st.session_state.experiment_started 
    st.radio(
        "Select Teacher Style:",
        ["Empathy Mode", "Neutral Mode"],
        key="mode_selection",
        disabled=mode_disabled
    )
    
    st.markdown("---")
    
    csv_data = pd.DataFrame({
        "SubjectID": [st.session_state.subject_id] * len(st.session_state.display_history),
        "AudioEnabled": ["On" if enable_audio else "Off"] * len(st.session_state.display_history),
        "Role": [m["role"] for m in st.session_state.display_history],
        "Content": [m["content"] for m in st.session_state.display_history],
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(st.session_state.display_history)
    }).to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download CSV (Backup)",
        data=csv_data,
        file_name=f"data_{st.session_state.subject_id}.csv",
        mime="text/csv"
    )

    if st.button("☁️ Force Save to Sheets"):
        success, msg = save_to_google_sheets(
            st.session_state.subject_id, 
            st.session_state.display_history, 
            st.session_state.active_mode, 
            enable_audio,
            f"Manual Save (Score: {st.session_state.correct_count})"
        )
        if success:
            st.success("Saved!")
        else:
            st.error(f"Failed: {msg}")

    if st.button("🔴 Reset Experiment"):
        reset_experiment_logic()
        st.rerun()

# --- Main UI ---
st.title("🧠 Sophia: Psychology Learning Experiment")
col_avatar, col_chat = st.columns([1, 2])

glb_data = get_glb_base64(YOUR_GLB_URL, LOCAL_GLB_PATH)
if glb_data:
    src = f"data:model/gltf-binary;base64,{glb_data}"
    html = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <model-viewer 
        src="{src}" camera-controls autoplay animation-name="*" shadow-intensity="1" 
        style="width:100%;height:520px;" interaction-prompt="none"
    ></model-viewer>
    """
    with col_avatar: components.html(html, height=540)

with col_chat:
    chat_container = st.container(height=520)
    locked_mode = st.session_state.active_mode

    with chat_container:
        for msg in st.session_state.display_history:
            avatar = "👩‍🏫" if msg["role"] == "assistant" and locked_mode == "Empathy Mode" else ("👨‍🏫" if msg["role"] == "assistant" else "👤")
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    if st.session_state.experiment_started:
        
        if len(st.session_state.display_history) == 0:
            trigger_msg = "The student has logged in. Please start Phase 1: Introduction now."
            st.session_state.messages.append({"role": "system", "content": trigger_msg})
            handle_bot_response("", chat_container, locked_mode, enable_audio)

        user_input = st.chat_input("Type your response here...")
        
        if user_input:
            with chat_container:
                st.chat_message("user", avatar="👤").write(user_input)
                st.session_state.display_history.append({"role": "user", "content": user_input})
                
                detect_sentiment(user_input)
                sentiment_val = st.session_state.sentiment_counter.value
                
                system_instruction = ""
                if locked_mode == "Empathy Mode":
                    if sentiment_val <= -2:
                        system_instruction = f"(System: User discouraged (Score {sentiment_val}). Be extra encouraging!) "
                    elif sentiment_val >= 2:
                        system_instruction = f"(System: User confident. Keep going.) "
                
                final_prompt = system_instruction + user_input
                handle_bot_response(final_prompt, chat_container, locked_mode, enable_audio)
    else:
        with chat_container:
            st.info("👈 Please enter your Subject ID in the sidebar and click 'Start Experiment' to begin.")
