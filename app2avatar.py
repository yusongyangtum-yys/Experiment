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
from mutagen.mp3 import MP3
import time
import io

# --- 1. Configuration ---

api_key_chatbot = st.secrets["OPENAI_API_KEY"]

try:
    client = OpenAI(api_key=api_key_chatbot)
except Exception as e:
    st.error(f"Failed to initialize OpenAI Client: {e}")
    st.stop()

MODEL = "gpt-4o-mini"
# 限制 Token 数量，配合 Prompt 控制长度
MAX_TOKENS = 400 
TEMPERATURE = 0.5   

# --- TTS Voice Configuration ---
VOICE_EMPATHY = "en-US-AnaNeural" 
VOICE_NEUTRAL = "en-US-GuyNeural"

# --- Prompt Definitions ---
SYSTEM_PROMPT_EMPATHY = (
    "You are Sophia, a supportive psychology teacher. Your goal is to teach 6 topics step-by-step: "
    "1. Classical Conditioning, 2. Operant Conditioning, 3. Memory Types, "
    "4. Cognitive Biases, 5. Social Conformity, 6. Motivation Theory."
    "\n\n"
    "### IMPORTANT: LENGTH CONTROL"
    "\n- **Keep every response SHORT (under 80 words).**"
    "\n- If a concept is complex, explain the first half, then STOP and ask: 'Are you following so far?'"
    "\n- Do NOT explain everything in one big chunk."
    "\n\n"
    "### INSTRUCTION FLOW:"
    "\n\n"
    "**PHASE 1: INTRODUCTION**\n"
    "- Briefly introduce yourself and list the 6 topics.\n"
    "- Ask if the student is ready to begin Topic 1."
    "\n\n"
    "**PHASE 2: TEACHING LOOP (Repeat for ALL 6 topics)**\n"
    "1. **Teach ONE Concept**: Explain the concept strictly following the LENGTH CONTROL rule.\n"
    "2. **Stop & Ask**: Check understanding.\n"
    "3. **Formative Test**: Present 1 multiple-choice question about the *current* concept.\n"
    "4. **Feedback**: Wait for answer. If correct, praise. If incorrect, correct gently.\n"
    "5. **Transition**: Ask if ready for the NEXT topic. Repeat until all 6 are done."
    "\n\n"
    "**PHASE 3: SUMMATIVE EXAM (Final Phase)**\n"
    "- Trigger this ONLY after all 6 topics are taught.\n"
    "- Say: 'Now, let's take the final exam. I will ask 6 questions one by one.'\n"
    "- Ask questions one by one. After the last one, show total score and say 'The session is complete.'"
)

SYSTEM_PROMPT_NEUTRAL = (
    "You are a neutral, factual AI instructor teaching 6 Psychology topics. "
    "Do not use emotional language. Do not praise. Be concise."
    "\n\n"
    "### LENGTH CONTROL"
    "\n- **Keep responses SHORT (under 150 words).**"
    "\n- If complex, split the explanation. Explain part 1, then ask 'Shall I continue?' before part 2."
    "\n\n"
    "Follow the teaching flow: Intro -> 6 Topics (Teach-Test-Feedback) -> Final Exam."
)

# --- 2. Helper Functions ---

def save_to_google_sheets(subject_id, chat_history, score_summary="N/A"):
    try:
        if "gcp_service_account" not in st.secrets:
            return False, "Secrets not configured"
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        sheet_name = st.secrets.get("sheet_name", "Psychology_Experiment_Data")
        sh = gc.open(sheet_name)
        worksheet = sh.sheet1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_json = json.dumps(chat_history, ensure_ascii=False)
        row = [subject_id, timestamp, score_summary, history_json]
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

# --- 3. TTS Logic (Fixed) ---

async def edge_tts_generate(text, voice):
    """异步生成音频"""
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def play_audio_full(text, mode_selection):
    """
    修正后的播放逻辑：
    1. 接收完整文本
    2. 生成完整音频
    3. 计算音频准确时长
    4. 阻塞等待直至播放结束，防止音频被截断
    """
    if not text.strip():
        return
        
    voice = VOICE_EMPATHY if mode_selection == "Empathy Mode" else VOICE_NEUTRAL
    
    # 简单的文本清理
    clean_text = text.replace("*", "").replace("#", "").replace("`", "")

    try:
        with st.spinner("🔊 Generating voice..."):
            audio_bytes = asyncio.run(edge_tts_generate(clean_text, voice))
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return

    if not audio_bytes:
        return

    # --- 关键修正：准确计算时长 ---
    try:
        audio_stream = io.BytesIO(audio_bytes)
        audio = MP3(audio_stream)
        duration = audio.info.length
    except Exception as e:
        # 如果获取失败，估算一个较长的时长以防截断 (每秒约 2-3 个词)
        duration = len(clean_text.split()) / 2.5 
    
    # 增加一点缓冲时间 (0.5秒) 确保尾音播完
    sleep_time = duration + 0.5

    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    
    sound_placeholder = st.empty()
    sound_placeholder.markdown(md, unsafe_allow_html=True)
    
    # --- 关键修正：必须等待播放完成 ---
    # 如果不等待，sound_placeholder.empty() 会立即执行，导致浏览器移除音频标签，声音停止。
    time.sleep(sleep_time) 
    
    # 播放完成后清理 HTML，以免下次重叠
    sound_placeholder.empty()

# --- 4. Logic: Text First, Then Audio ---

def generate_text_and_speak(messages, chat_placeholder, mode_selection):
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=enforce_token_budget(messages),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
    except Exception as e:
        return f"API Error: {e}"

    full_response = ""
    
    # 1. 文本生成阶段
    for chunk in stream:
        txt = chunk.choices[0].delta.content
        if txt:
            full_response += txt
            chat_placeholder.markdown(full_response + "▌")
    
    # 完成文本显示
    chat_placeholder.markdown(full_response)
    
    # 2. 音频生成与播放阶段
    play_audio_full(full_response, mode_selection)

    return full_response

def reset_experiment():
    st.session_state.display_history = []
    st.session_state.sentiment_counter.reset()
    st.session_state.experiment_started = False
    mode = st.session_state.get("mode_selection", "Empathy Mode")
    prompt = SYSTEM_PROMPT_EMPATHY if mode == "Empathy Mode" else SYSTEM_PROMPT_NEUTRAL
    st.session_state.messages = [{"role": "system", "content": prompt}]

# --- 5. Streamlit UI ---

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

# --- Sidebar ---
with st.sidebar:
    st.title("🔧 Settings")
    input_id = st.text_input("Enter Subject ID:", value=st.session_state.subject_id)
    if input_id != st.session_state.subject_id:
        st.session_state.subject_id = input_id
    
    if not st.session_state.experiment_started:
        if st.button("🚀 Start Experiment", type="primary"):
            if st.session_state.subject_id.strip():
                st.session_state.experiment_started = True
                st.rerun()
            else:
                st.error("⚠️ Enter Subject ID first.")
    else:
        st.success(f"Running: {st.session_state.subject_id}")
    
    st.markdown("---")
    
    mode_disabled = st.session_state.experiment_started 
    mode_selection = st.radio(
        "Select Teacher Style:",
        ["Empathy Mode", "Neutral Mode"],
        key="mode_selection",
        on_change=reset_experiment,
        disabled=mode_disabled
    )
    
    st.markdown("---")
    if st.button("Download CSV (Backup)"):
        data = {
            "SubjectID": [st.session_state.subject_id] * len(st.session_state.display_history),
            "Role": [m["role"] for m in st.session_state.display_history],
            "Content": [m["content"] for m in st.session_state.display_history],
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(st.session_state.display_history)
        }
        pd.DataFrame(data).to_csv("data.csv", index=False)
        st.write("CSV Saved locally (Simulation).")

    if st.button("🔴 Reset Experiment"):
        reset_experiment()
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
    
    with chat_container:
        for msg in st.session_state.display_history:
            avatar = "👩‍🏫" if msg["role"] == "assistant" else "👤"
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    if st.session_state.experiment_started:
        
        # A. Auto-Start Logic
        if len(st.session_state.display_history) == 0:
            with chat_container:
                with st.chat_message("assistant", avatar="👩‍🏫"):
                    chat_placeholder = st.empty()
                    
                    trigger_msg = {"role": "system", "content": "The student has logged in. Please start Phase 1: Introduction now."}
                    st.session_state.messages.append(trigger_msg)
                    
                    full_response = generate_text_and_speak(
                        st.session_state.messages, 
                        chat_placeholder, 
                        mode_selection
                    )
                    
                    st.session_state.display_history.append({"role": "assistant", "content": full_response})
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

        # B. User Input Logic
        user_input = st.chat_input("Type your response here...")
        
        if user_input:
            with chat_container:
                st.chat_message("user", avatar="👤").write(user_input)
                st.session_state.display_history.append({"role": "user", "content": user_input})
                
                detect_sentiment(user_input)
                sentiment_val = st.session_state.sentiment_counter.value
                
                system_instruction = ""
                if mode_selection == "Empathy Mode":
                    if sentiment_val <= -2:
                        system_instruction = f"(System: User discouraged (Score {sentiment_val}). Be extra encouraging!) "
                    elif sentiment_val >= 2:
                        system_instruction = f"(System: User confident. Keep going.) "
                
                st.session_state.messages.append({"role": "user", "content": system_instruction + user_input})
                
                with st.chat_message("assistant", avatar="👩‍🏫"):
                    chat_placeholder = st.empty()
                    
                    full_response = generate_text_and_speak(
                        st.session_state.messages, 
                        chat_placeholder, 
                        mode_selection
                    )
                    
                    if "session is complete" in full_response.lower():
                        save_to_google_sheets(st.session_state.subject_id, st.session_state.display_history, "Completed")
                        st.success("Session Data Saved.")
                
                st.session_state.display_history.append({"role": "assistant", "content": full_response})
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    else:
        with chat_container:
            st.info("👈 Please enter your Subject ID in the sidebar and click 'Start Experiment' to begin.")