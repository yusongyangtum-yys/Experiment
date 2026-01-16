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

# --- Prompt Definitions ---

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

def save_to_google_sheets(subject_id, mode, final_score):
    """
    保存数据到 Google Sheets
    Schema: [UUID, Mode, FinishTime, Score]
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
        
        # 获取当前时间
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建简化的数据行
        row = [str(subject_id), str(mode), str(timestamp), str(final_score)]
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
    if len(messages) > 60:
        return [messages[0]] + messages[-58:]
    return messages

# --- 3. Logic ---

def handle_bot_response(user_input, chat_container, active_mode):
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
            
            if "begin the final exam" in full_response.lower():
                st.session_state.correct_count = 0

            clean_display_response = full_response
            
            if "[CORRECT]" in full_response:
                st.session_state.correct_count += 1
                clean_display_response = full_response.replace("[CORRECT]", "").strip()
            elif "[INCORRECT]" in full_response:
                clean_display_response = full_response.replace("[INCORRECT]", "").strip()
            
            chat_placeholder.markdown(clean_display_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response}) 
            st.session_state.display_history.append({"role": "assistant", "content": clean_display_response})
            
            # --- 结算逻辑 ---
            response_lower = full_response.lower()
            if ("session" in response_lower and "complete" in response_lower) or ("score" in response_lower and "10" in response_lower):
                final_score = st.session_state.correct_count
                
                st.info(f"📊 Final Score: {final_score}/10")
                
                # 保存简化版数据: ID, Mode, Time, Score
                success, msg = save_to_google_sheets(
                    st.session_state.subject_id, 
                    active_mode, 
                    final_score
                )
                if success:
                    st.success("✅ Experiment Complete. Data Saved.")
                    st.balloons()
                else:
                    st.error(f"Save Failed: {msg}")

# --- 4. Initialization & Setup ---

st.set_page_config(page_title="Psychology Experiment", layout="wide", initial_sidebar_state="collapsed")

# 隐藏侧边栏
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# --- 核心修改：ID生成与自动平衡 Mode 分配 ---

if "subject_id" not in st.session_state:
    # 1. 生成随机 8位 ID
    auto_id = str(uuid.uuid4())[:8]
    st.session_state.subject_id = f"SUB_{auto_id}"

if "active_mode" not in st.session_state:
    # 2. 确定性哈希分配 (Deterministic Hashing)
    # 将 ID 转为哈希整数，对 2 取余。
    # 结果 0 -> Empathy, 1 -> Neutral
    # 这确保了同一个 ID 永远对应同一个模式，且在大样本下概率为 50/50。
    
    hash_object = hashlib.md5(st.session_state.subject_id.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    
    if hash_int % 2 == 0:
        st.session_state.active_mode = "Empathy Mode"
    else:
        st.session_state.active_mode = "Neutral Mode"
        
    # 可选：打印日志用于后台调试
    # print(f"Assigned {st.session_state.subject_id} to {st.session_state.active_mode}")

# 3. 初始化 System Prompt
if "messages" not in st.session_state:
    prompt = SYSTEM_PROMPT_EMPATHY if st.session_state.active_mode == "Empathy Mode" else SYSTEM_PROMPT_NEUTRAL
    st.session_state.messages = [{"role": "system", "content": prompt}]

if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0

# --- 5. Main UI ---

st.title("🧠 Psychology Learning Session")

col_avatar, col_chat = st.columns([1, 2])

# 3D Model
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
    with col_avatar: 
        components.html(html, height=540)

with col_chat:
    chat_container = st.container(height=520)
    locked_mode = st.session_state.active_mode

    # 显示历史记录
    with chat_container:
        for msg in st.session_state.display_history:
            avatar = "👩‍🏫" if msg["role"] == "assistant" and locked_mode == "Empathy Mode" else ("👨‍🏫" if msg["role"] == "assistant" else "👤")
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # 自动触发第一句话 (Auto Start)
    if len(st.session_state.display_history) == 0:
        trigger_msg = "The student has logged in. Please start Phase 1: Introduction now."
        has_assistant_reply = any(m["role"] == "assistant" for m in st.session_state.messages)
        
        if not has_assistant_reply:
            st.session_state.messages.append({"role": "system", "content": trigger_msg})
            handle_bot_response("", chat_container, locked_mode)
            st.rerun() 

    # 用户输入
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
            handle_bot_response(final_prompt, chat_container, locked_mode)
