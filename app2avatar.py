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

# --- Prompt Definitions (Updated to Strict Control Version) ---

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

**Allowed Structure per Message:**
[Explanation of Concept] + [Real-world Example] + [Short Pause Question]

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
8. Wait for answer -> Give warm feedback.
9. Ask if ready for the next topic.

PHASE 3: FINAL EXAM
- Trigger ONLY after all 3 topics are finished.
- Say: "Now we will begin the final exam. I will ask 10 questions one by one."
- Exam rules:
  - Ask ONE multiple-choice question at a time.
  - STOP and wait for answer.
  - Give empathetic feedback.
  - Move to next question.
- After Question 10, calculate score and say:
  "You got X out of 10. The session is complete."
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

**Allowed Structure per Message:**
[Factual Explanation] + [Details/Experiment] + [Status Check]

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
8. Wait for answer -> Give factual feedback ("Correct"/"Incorrect").
9. Proceed to next topic.

PHASE 3: FINAL EXAM
- Start ONLY after Topic 3 is finished.
- Say: "We will now begin the final exam consisting of 10 multiple-choice questions."
- Rules:
  - Ask ONE question at a time.
  - STOP and wait for input.
  - Give factual feedback only.
  - Continue until Question 10.
- After Question 10, calculate and report:
  "Score: X/10. The session is complete."
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

# 修改点：加入 audio_enabled 参数
def save_to_google_sheets(subject_id, chat_history, mode, audio_enabled, score_summary="N/A"):
    """保存数据到 Google Sheets"""
    try:
        # 1. 检查 Secrets
        if "gcp_service_account" not in st.secrets:
            return False, "Error: 'gcp_service_account' not found in st.secrets."
        
        # 2. 连接
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        # 3. 打开工作表
        sheet_name = st.secrets.get("sheet_name", "Experiment_Data")
        try:
            sh = gc.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            return False, f"Error: Spreadsheet '{sheet_name}' not found."

        worksheet = sh.sheet1
        
        # 4. 准备数据
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        audio_status = "On" if audio_enabled else "Off"
        
        # 5. 写入行 [ID, Time, Mode, Audio, Score]
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

if "sentiment_counter" not in st.session_state: st.session_state.sentiment_counter = SafeCounter()

def detect_sentiment(user_message):
    msg = user_message.lower()
    for w in POSITIVE_WORDS: 
        if w in msg: st.session_state.sentiment_counter.increment()
    for w in NEGATIVE_WORDS: 
        if w in msg: st.session_state.sentiment_counter.decrement()

def enforce_token_budget(messages):
    if len(messages) > 12:
        return [messages[0]] + messages[-10:]
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
    # 如果文本为空或语音功能被关闭，直接返回
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
            
            chat_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.display_history.append({"role": "assistant", "content": full_response})
            
            # --- 自动保存触发逻辑 ---
            if "session is complete" in full_response.lower():
                summary_text = "Completed"
                if "score" in full_response.lower():
                    # 尝试提取分数信息
                    summary_text = f"Completed ({full_response[-30:].strip()})" 
                
                # 传入 enable_audio 状态
                success, msg = save_to_google_sheets(
                    st.session_state.subject_id, 
                    st.session_state.display_history, 
                    active_mode, 
                    enable_audio, 
                    summary_text
                )
                if success:
                    st.success("✅ Session Data Successfully Saved to Google Sheets!")
                    st.balloons()
                else:
                    st.error(f"❌ Save Failed: {msg}")

            play_audio_full(full_response, active_mode, enable_audio)

def reset_experiment_logic():
    st.session_state.display_history = []
    st.session_state.sentiment_counter.reset()
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
    
    # CSV 下载 (包含 Audio Status 列)
    csv_data = pd.DataFrame({
        "SubjectID": [st.session_state.subject_id] * len(st.session_state.display_history),
        "AudioEnabled": ["On" if enable_audio else "Off"] * len(st.session_state.display_history), # 新增列
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
        # 强制保存时也带上 audio_enabled
        success, msg = save_to_google_sheets(
            st.session_state.subject_id, 
            st.session_state.display_history, 
            st.session_state.active_mode, 
            enable_audio,
            "Manual Save"
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
