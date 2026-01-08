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
import time
import uuid # 新增：用于生成唯一ID

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
VOICE_NEUTRAL = "en-US-GuyNeural"
TTS_RATE = "+25%" 

# --- Prompt Definitions ---
SYSTEM_PROMPT_EMPATHY = (
    "You are Sophia, a supportive psychology teacher. Your goal is to teach 6 topics step-by-step: "
    "1. Classical Conditioning, 2. Operant Conditioning, 3. Memory Types, "
    "4. Cognitive Biases, 5. Social Conformity, 6. Motivation Theory."
    "\n\n"
    "### IMPORTANT: LENGTH CONTROL"
    "\n- **Keep every response Moderate (around 150-200 words).**"
    "\n- Explain concepts clearly with examples."
    "\n- If a concept is VERY complex, you may ask 'Are you following?' in the middle, but generally try to explain one concept fully."
    "\n\n"
    "### INSTRUCTION FLOW:"
    "\n\n"
    "**PHASE 1: INTRODUCTION**\n"
    "- Briefly introduce yourself and list the 6 topics.\n"
    "- Ask if the student is ready to begin Topic 1."
    "\n\n"
    "**PHASE 2: TEACHING LOOP (Repeat for ALL 6 topics)**\n"
    "1. **Teach ONE Concept**: Explain the concept fully (150-200 words).\n"
    "2. **Check Understanding**: Ask 'Do you have any questions about this topic, or shall we move to the next one?'\n"
    "3. **Transition**: If user says yes/ready, move to the NEXT topic. (NO INTERMEDIATE QUIZZES)."
    "\n\n"
    "**PHASE 3: SUMMATIVE EXAM (Final Phase)**\n"
    "- Trigger this ONLY after all 6 topics are taught.\n"
    "- Say: 'Now that we have finished all topics, let's take the final exam. I will ask 15 questions one by one.'\n"
    "- **Action**: Present 15 multiple-choice questions one by one. Wait for the answer after each question.\n"
    "- After the 15th question, show the total score and say 'Thanks for the effort. The session is complete.'"
)

SYSTEM_PROMPT_NEUTRAL = (
    "You are a neutral, factual AI instructor teaching 6 Psychology topics. "
    "Do not use emotional language. Do not praise. Be concise but thorough."
    "\n\n"
    "### LENGTH CONTROL"
    "\n- **Keep responses Moderate (around 150-200 words).**"
    "\n\n"
    "Follow the teaching flow: Intro -> 6 Topics (Teach -> Check Understanding -> Next) -> Final Exam (15 Questions)."
)

# --- 2. Javascript Hack (Fixing Issue 1: Interruption) ---

def stop_previous_audio():
    # 这个脚本不仅暂停，还移除所有的 audio 元素，确保没有残留
    js_code = """
        <script>
            var audios = document.getElementsByTagName('audio');
            for(var i = 0; i < audios.length; i++){
                audios[i].pause();
                audios[i].currentTime = 0;
                audios[i].remove(); // 暴力移除，防止占位
            }
        </script>
    """
    components.html(js_code, height=0, width=0)

# 每次 Rerun 最开始就执行清理
stop_previous_audio()

# --- 3. Helper Functions ---

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

# --- 4. TTS Logic (Fixing Issue 1: Audio Silence) ---

async def edge_tts_generate(text, voice):
    """异步生成音频，应用语速参数"""
    communicate = edge_tts.Communicate(text, voice, rate=TTS_RATE)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def play_audio_full(text, mode_selection):
    """
    修改逻辑：使用唯一 ID 强制浏览器刷新音频
    """
    if not text.strip():
        return
        
    voice = VOICE_EMPATHY if mode_selection == "Empathy Mode" else VOICE_NEUTRAL
    clean_text = text.replace("*", "").replace("#", "").replace("`", "")

    # 1. 立即清除上一个音频容器（如果有）
    if "audio_container" in st.session_state:
        st.session_state.audio_container.empty()

    try:
        # 即使这里还在转圈，如果用户打断，下面的代码就不会执行，状态在外面已经保存了
        with st.spinner("🔊 Generating audio..."):
            audio_bytes = asyncio.run(edge_tts_generate(clean_text, voice))
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return

    if not audio_bytes:
        return

    b64 = base64.b64encode(audio_bytes).decode()
    
    # IMPORTANT FIX: 生成唯一的 div ID，强制浏览器认为是新内容
    unique_id = f"audio_{uuid.uuid4()}"
    
    md = f"""
        <audio autoplay="true" style="display:none;" id="{unique_id}">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    
    # 重新创建一个容器
    st.session_state.audio_container = st.empty()
    st.session_state.audio_container.markdown(md, unsafe_allow_html=True)

# --- 5. Logic: Text and State Management (Fixing Issue 2: Repetition) ---

def handle_bot_response(user_input, chat_container, mode_selection):
    """
    核心逻辑重构：
    1. 显示用户输入
    2. 生成 LLM 文本
    3. 【立刻】保存 LLM 文本到 Session State (防止打断后丢失导致重复)
    4. 最后才生成音频
    """
    
    # 1. Append User Input to internal messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Start Generating Bot Response
    with chat_container:
        with st.chat_message("assistant", avatar="👩‍🏫"):
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
            
            # --- IMPORTANT FIX: SAVE STATE HERE ---
            # 文本生成一旦完成，立刻保存到历史记录。
            # 这样，即使紧接着的音频生成被打断，LLM 也“记住”了它已经说过这句话。
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.display_history.append({"role": "assistant", "content": full_response})
            
            # Check for completion to save data
            if "session is complete" in full_response.lower():
                save_to_google_sheets(st.session_state.subject_id, st.session_state.display_history, "Completed")
                st.success("Session Data Saved.")

            # 3. Generate & Play Audio (Last Step)
            # 如果用户在这期间输入，Script 停止，但上面的 append 已经执行，所以不会重复。
            play_audio_full(full_response, mode_selection)


def reset_experiment():
    st.session_state.display_history = []
    st.session_state.sentiment_counter.reset()
    st.session_state.experiment_started = False
    st.session_state.audio_container = st.empty()
    mode = st.session_state.get("mode_selection", "Empathy Mode")
    prompt = SYSTEM_PROMPT_EMPATHY if mode == "Empathy Mode" else SYSTEM_PROMPT_NEUTRAL
    st.session_state.messages = [{"role": "system", "content": prompt}]

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
    
    # 渲染历史记录
    with chat_container:
        for msg in st.session_state.display_history:
            avatar = "👩‍🏫" if msg["role"] == "assistant" else "👤"
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    if st.session_state.experiment_started:
        
        # A. Auto-Start Logic
        if len(st.session_state.display_history) == 0:
            # 触发开场白
            trigger_msg = "The student has logged in. Please start Phase 1: Introduction now."
            # 手动注入 context 到 messages
            st.session_state.messages.append({"role": "system", "content": trigger_msg})
            # 调用处理函数（不作为 User Input，而是系统触发）
            handle_bot_response("", chat_container, mode_selection)

        # B. User Input Logic
        user_input = st.chat_input("Type your response here...")
        
        if user_input:
            with chat_container:
                st.chat_message("user", avatar="👤").write(user_input)
                st.session_state.display_history.append({"role": "user", "content": user_input})
                
                # 情感分析 & Prompt 调整
                detect_sentiment(user_input)
                sentiment_val = st.session_state.sentiment_counter.value
                
                system_instruction = ""
                if mode_selection == "Empathy Mode":
                    if sentiment_val <= -2:
                        system_instruction = f"(System: User discouraged (Score {sentiment_val}). Be extra encouraging!) "
                    elif sentiment_val >= 2:
                        system_instruction = f"(System: User confident. Keep going.) "
                
                # 组合最终输入
                final_prompt = system_instruction + user_input
                
                # 调用处理核心
                handle_bot_response(final_prompt, chat_container, mode_selection)

    else:
        with chat_container:
            st.info("👈 Please enter your Subject ID in the sidebar and click 'Start Experiment' to begin.")
