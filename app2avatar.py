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
# ... (此处省略 SYSTEM_PROMPT_EMPATHY 和 SYSTEM_PROMPT_NEUTRAL 的内容，保持原样)

# --- 2. Helper Functions ---

def save_to_google_sheets(data_dict):
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

# ... (此处省略 detect_sentiment 和 enforce_token_budget 等辅助函数，保持原样)

# --- 3. Logic ---

def handle_bot_response(user_input, chat_container, active_mode):
    current_time = datetime.datetime.now()
    if st.session_state.last_bot_finish_time:
        time_diff = (current_time - st.session_state.last_bot_finish_time).total_seconds()
        if time_diff < 300: 
            st.session_state.user_response_times.append(time_diff)

    if user_input:
        word_count = len(user_input.split())
        st.session_state.user_total_words += word_count
        st.session_state.messages.append({"role": "user", "content": user_input})
    
    with chat_container:
        bot_avatar = "👨‍🏫" if active_mode == "Neutral Mode" else "👩‍🏫"
        
        with st.chat_message("assistant", avatar=bot_avatar):
            chat_placeholder = st.empty()
            full_response = ""
            
            if user_input.strip() == "/dev_skip":
                full_response = "The session is complete. Score: 10/10."
                chat_placeholder.markdown(full_response)
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

            st.session_state.last_bot_finish_time = datetime.datetime.now()

            if "begin the final exam" in full_response.lower():
                st.session_state.correct_count = 0

            clean_display_response = full_response
            
            if "[CORRECT]" in full_response:
                st.session_state.correct_count += 1
                clean_display_response = full_response.replace("[CORRECT]", "").strip()
            elif "[INCORRECT]" in full_response:
                clean_display_response = full_response.replace("[INCORRECT]", "").strip()
            
            if user_input.strip() != "/dev_skip":
                chat_placeholder.markdown(clean_display_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response}) 
            st.session_state.display_history.append({"role": "assistant", "content": clean_display_response})
            
            response_lower = full_response.lower()
            if ("session" in response_lower and "complete" in response_lower) or ("score" in response_lower and "10" in response_lower):
                
                # 计算指标
                final_score = st.session_state.correct_count
                end_time = datetime.datetime.now()
                start_time = st.session_state.session_start_time
                duration_seconds = (end_time - start_time).total_seconds()
                sentiment_val = st.session_state.sentiment_counter.value
                
                if len(st.session_state.user_response_times) > 0:
                    avg_resp_time = statistics.mean(st.session_state.user_response_times)
                else:
                    avg_resp_time = 0
                
                turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
                confusion_rate = st.session_state.confusion_counter / turn_count if turn_count > 0 else 0
                dialogue_dump = json.dumps(st.session_state.messages, ensure_ascii=False)

                st.info(f"📊 Final Score: {final_score}/10 | Time: {int(duration_seconds)}s")
                
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
                
                success, msg = save_to_google_sheets(data_payload)
                
                if success:
                    st.success("✅ Experiment Complete. All metrics saved successfully.")
                    st.balloons()
                    
                    # --- [后端修复重点]：直接从 session_state 获取锁定的 URL ---
                    post_survey_url_final = st.session_state.get("post_survey_url", "#")
                    
                    st.markdown("---")
                    st.markdown(f"""
                    <a href="{post_survey_url_final}" target="_blank" style="text-decoration:none;">
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

# 隐藏侧边栏 (原代码保持)
st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)

# --- [后端修复重点]：ID生成与 URL 强制绑定 ---
if "subject_id" not in st.session_state:
    # 1. 生成 ID
    auto_id = str(uuid.uuid4())[:8]
    st.session_state.subject_id = f"SUB_{auto_id}"
    
    # 2. 立即定义 Entry ID 和 Base URL (后端常量)
    PRE_SURVEY_ENTRY_ID = "entry.538559089"
    POST_SURVEY_ENTRY_ID = "entry.596968110"
    PRE_SURVEY_BASE = "https://docs.google.com/forms/d/e/1FAIpQLSdqNQ8oRvM-kxVTitRXCtGRuQg_oopmegL-koixLQxJVVjayA/viewform"
    POST_SURVEY_BASE = "https://docs.google.com/forms/d/e/1FAIpQLSckI_yCbL5gQu6P7aP-9vRn5BKp7fX8NrBA_z3FmEegIggCTg/viewform"
    
    # 3. 构造完整链接并永久存入 session_state
    st.session_state.pre_survey_url = f"{PRE_SURVEY_BASE}?usp=pp_url&{PRE_SURVEY_ENTRY_ID}={st.session_state.subject_id}"
    st.session_state.post_survey_url = f"{POST_SURVEY_BASE}?usp=pp_url&{POST_SURVEY_ENTRY_ID}={st.session_state.subject_id}"

# --- 状态控制 ---
if "pre_survey_completed" not in st.session_state:
    st.session_state.pre_survey_completed = False
if "auto_start_triggered" not in st.session_state:
    st.session_state.auto_start_triggered = False

# ... (后续 Mode 分配、Metrics 初始化等逻辑保持原样，但 URL 改为直接调用 session_state)

if "active_mode" not in st.session_state:
    hash_object = hashlib.md5(st.session_state.subject_id.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    st.session_state.active_mode = "Empathy Mode" if hash_int % 2 == 0 else "Neutral Mode"

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.datetime.now()
if "user_response_times" not in st.session_state:
    st.session_state.user_response_times = []
if "last_bot_finish_time" not in st.session_state:
    st.session_state.last_bot_finish_time = datetime.datetime.now()
if "user_total_words" not in st.session_state:
    st.session_state.user_total_words = 0
if "messages" not in st.session_state:
    prompt = SYSTEM_PROMPT_EMPATHY if st.session_state.active_mode == "Empathy Mode" else SYSTEM_PROMPT_NEUTRAL
    st.session_state.messages = [{"role": "system", "content": prompt}]
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0

# --- 5. Main UI Logic ---

if not st.session_state.pre_survey_completed:
    st.container().markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🎓 Psychology Learning Experiment")
        st.info("👋 Welcome! Before we begin the session with the AI teacher, please complete a short survey.")
        st.write(f"**Your Participant ID:** `{st.session_state.subject_id}` (Auto-filled)")
        
        # 修正：直接从 session_state 获取
        pre_url = st.session_state.pre_survey_url
        st.markdown(f"""
        <a href="{pre_url}" target="_blank" style="text-decoration:none;">
            <div style="background-color: #4CAF50; color: white; padding: 20px; text-align: center; border-radius: 8px; font-size: 18px; margin: 20px 0; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                👉 Click Here to Start Pre-Survey
            </div>
        </a>
        """, unsafe_allow_html=True)
        
        st.warning("⚠️ Please keep this tab open...")
        st.write("---")
        if st.button("I have submitted the Pre-Survey"):
            st.session_state.pre_survey_completed = True
            st.rerun()

else:
    # ... (Avatar 渲染和 Chat 逻辑部分保持 100% 原样)
    # 唯一确保的是 handle_bot_response 内部引用的 post_survey_url 现在来自 session_state
    # (此处省略 avatar 渲染代码，引用你原文件中的 HTML 即可)
    # ...
