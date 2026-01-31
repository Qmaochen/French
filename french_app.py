import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import edge_tts
import asyncio
from streamlit_mic_recorder import mic_recorder
from io import BytesIO
from rapidfuzz import fuzz
from groq import Groq
import os
import re
import random
from streamlit_gsheets import GSheetsConnection  # 新增引用

# --- 🎨 1. UI 設定 ---
st.set_page_config(
    page_title="French SRS Master", 
    layout="centered", 
    page_icon="🇫🇷",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 隱藏預設元件，但保留 Header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* 全局樣式 */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }

    .question-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        margin-bottom: 25px;
        border: 1px solid #edf2f7;
        text-align: center;
        transition: all 0.3s ease;
    }

    .tag-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        background-color: #e0e7ff;
        color: #4338ca;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .big-font {
        font-size: 20px !important;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.5;
        margin: 20px 0;
    }
    
    .hint-text {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 15px;
        font-weight: 500;
    }

    .stAudio {
        margin-top: 15px;
        margin-bottom: 25px;
        width: 100%;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    div.stTextInput > div > div > input {
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
        font-size: 1.1rem;
    }

    iframe {
        border: none !important;
        margin-bottom: 10px;
    }
    
    .result-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        text-align: left;
    }
    
    .correct-answer {
        color: #059669;
        font-weight: 600;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 檔案與資料庫 (Google Sheets 版本) ---

# 建立連線物件
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    try:
        # 使用 ttl=0 確保每次都讀取最新資料，不使用快取
        df = conn.read(worksheet="Sheet1", ttl=0)
    except Exception as e:
        st.error(f"無法讀取 Google Sheet: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    required = ['Sentences', 'Tags', 'Answers', 'Captions', 'Date', 'Times', 'Next']
    for col in required:
        if col not in df.columns:
            df[col] = "" if col not in ['Times', 'Next', 'Date'] else (0 if col=='Times' else datetime.now().date())

    df['Times'] = pd.to_numeric(df['Times'], errors='coerce').fillna(0).astype(int)
    # 確保日期格式正確
    df['Next'] = pd.to_datetime(df['Next'], errors='coerce').fillna(pd.Timestamp.now()).dt.date
    return df.dropna(subset=['Sentences'])

def save_data(df):
    try:
        # 將日期轉為字串格式存入 Google Sheet，避免格式錯亂
        save_df = df.copy()
        save_df['Next'] = pd.to_datetime(save_df['Next']).dt.strftime('%Y-%m-%d')
        if 'Date' in save_df.columns:
             save_df['Date'] = pd.to_datetime(save_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

        conn.update(worksheet="Sheet1", data=save_df)
        st.cache_data.clear() # 清除快取以防萬一
    except Exception as e:
        st.error(f"⚠️ 無法存檔至 Google Sheet：{e}")

# --- 3. 核心邏輯工具 ---

def go_next_question():
    st.session_state.current_q_idx = None

def get_target_answer(row):
    if pd.notna(row['Answers']) and str(row['Answers']).strip() != "":
        return str(row['Answers']).strip()
    return str(row['Sentences']).replace("[", "").replace("]", "").strip()

async def play_audio(text):
    clean_text = text.replace("[", "").replace("]", "")
    communicate = edge_tts.Communicate(clean_text, "fr-FR-HenriNeural", rate="-20%")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def transcribe_with_groq(api_key, audio_bytes):
    if not api_key: return "Error: No API Key"
    client = Groq(api_key=api_key)
    try:
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.webm"
        return client.audio.transcriptions.create(
            file=audio_file, model="whisper-large-v3", language="fr", response_format="text"
        )
    except Exception as e:
        return f"Error: {str(e)}"

def llm_grade_answer(api_key, user_text, context_text, correct_answer):
    if not api_key: return 0.0, "請輸入 API Key"
    client = Groq(api_key=api_key)
    prompt = f"""
    Context/Scenario: "{context_text}"
    Reference Ideal Answer: "{correct_answer}"
    User's Answer: "{user_text}"
    
    There is no reference ideal answer if the problem is short conversation.
    So, please grade it based on daily conversational standards.
    Task: Grade the User's Answer from 0.00 to 100.00.
    Criteria: 
    1. Does the user convey the meaning of the Reference Ideal Answer?
    2. Is the grammar correct?
    
    Return ONLY the number (e.g. 85.50).
    """
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        match = re.search(r'\d+(\.\d+)?', chat.choices[0].message.content)
        return float(match.group()) if match else 0.0, "AI Graded"
    except:
        return 0.0, "AI Error"

# --- 4. 初始化 Session State ---

if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'current_q_idx' not in st.session_state:
    st.session_state.current_q_idx = None
if 'last_q_idx' not in st.session_state:
    st.session_state.last_q_idx = -1
if 'q_processed' not in st.session_state:
    st.session_state.q_processed = False
    st.session_state.q_user_text = ""
    st.session_state.q_grade = 0.0
    st.session_state.q_ai_msg = ""
    st.session_state.q_saved_idx = -1
if 'api_key_input' not in st.session_state:
    st.session_state.api_key_input = ""

# --- 5. 主程式 ---

with st.sidebar:
    st.markdown("### ⚙️ 設定 (Settings)")
    
    groq_api_key = st.text_input(
        "Groq API Key", 
        type="password", 
        value=st.session_state.api_key_input, 
        help="輸入 Key 才能使用 AI 語音功能",
        placeholder="請貼上您的 Key"
    )
    st.session_state.api_key_input = groq_api_key

    st.markdown("---")
    if st.button("🔄 重新載入題庫"):
        st.session_state.df = load_data()
        st.session_state.current_q_idx = None
        st.rerun()

st.title("🇫🇷 French SRS Master")

if not groq_api_key:
    st.info("💡 提示：請在左側選單輸入 API Key 以啟用語音功能。")

today = datetime.now().date()
df = st.session_state.df
due_indices = df[df['Next'] <= today].index.tolist()

if not due_indices:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 60px;">
        <h1 style="color:#10b981; font-size: 3rem;">Bravo! 🎉</h1>
        <p style="font-size: 1.2rem; color: #64748b;">今天的複習進度已全部完成。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔄 強制複習全部 (Demo Mode)", use_container_width=True):
            due_indices = df.index.tolist()
            st.session_state.current_q_idx = random.choice(due_indices)
            st.rerun()

else:
    # 選題邏輯
    if st.session_state.current_q_idx is None or st.session_state.current_q_idx not in due_indices:
        st.session_state.current_q_idx = random.choice(due_indices)
    
    current_idx = st.session_state.current_q_idx
    row = df.loc[current_idx]

    # 換題檢測
    if current_idx != st.session_state.last_q_idx:
        st.session_state.q_processed = False
        st.session_state.q_user_text = ""
        st.session_state.q_grade = 0.0
        st.session_state.q_ai_msg = ""
        st.session_state.last_q_idx = current_idx

    target_answer = get_target_answer(row)

    # 進度條
    progress_val = 1.0 - (len(due_indices) / len(df))
    st.progress(progress_val)
    c1, c2 = st.columns([1, 1])
    with c1: st.caption(f"📅 待複習: {len(due_indices)} 題")
    with c2: st.caption(f"🔥 連續答對: {row['Times']} 次")

    # -----------------------------------------------------------
    # [修改] 音檔生成邏輯
    # 預設唸 Sentences，但如果是 Question/Phrases，我們希望聽到 Answer
    audio_source_text = row['Sentences']
    
    if row['Tags'] in ['Question', 'Phrases']:
        # 檢查 Answers 是否有值，有才用，沒有則退回用 Sentences
        if pd.notna(row['Answers']) and str(row['Answers']).strip() != "":
            audio_source_text = str(row['Answers'])

    audio_bytes = asyncio.run(play_audio(audio_source_text))
    # -----------------------------------------------------------

    # --- 🃏 題目卡片區域 ---
    # st.markdown('<div class="question-card">', unsafe_allow_html=True)
    st.markdown(f'<span class="tag-badge">{row["Tags"]}</span>', unsafe_allow_html=True)

    # === [Visibility Logic] 決定顯示什麼 ===
    show_text_initially = False
    text_content = ""
    
    # Writing, Conversation -> 隱藏文字 (考聽力)
    # Speaking, Phrases, Question -> 顯示文字
    
    if row['Tags'] in ['Speaking', 'Question', 'Phrases']:
        show_text_initially = True
        
        if row['Tags'] == 'Question':
            # Question 顯示 Captions (題目)
            if pd.notna(row['Captions']) and str(row['Captions']).strip() != "":
                text_content = str(row['Sentences']) + " " + "\n" + str(row['Captions'])
            else:
                text_content = str(row['Sentences'])
                
        elif row['Tags'] == 'Phrases':
            # Phrases 挖空
            match = re.search(r'\[(.*?)\]', row['Sentences'])
            if match:
                text_content = row['Sentences'].replace(f"[{match.group(1)}]", " <span style='border-bottom: 2px solid #4f46e5; color: #4f46e5; font-weight:bold;'>______</span> ")
            else:
                text_content = row['Sentences'].replace(target_answer, " ______ ") if target_answer in row['Sentences'] else row['Sentences'] + " ______"
        else:
            # Speaking
            text_content = row['Sentences'].replace('\n', '<br>')

    # Audio 顯示邏輯
    show_audio_initially = False
    if row['Tags'] in ['Writing', 'Conversation', 'Speaking']:
        show_audio_initially = True

    # === [Render] 渲染 ===
    
    # 1. 顯示文字
    if show_text_initially:
        st.markdown(f'<p class="big-font">{text_content}</p>', unsafe_allow_html=True)
    
    # 2. 顯示提示
    elif not st.session_state.q_processed:
        st.markdown('<p class="hint-text" style="font-size:1.2rem;">🎧 請仔細聆聽音檔回答問題...</p>', unsafe_allow_html=True)

    # 3. 顯示音檔 (如果需要一開始就播)
    if show_audio_initially:
        st.audio(audio_bytes, format='audio/mp3')

    # === [Input Phase] 輸入階段 ===
    if not st.session_state.q_processed:
        
        # A. 錄音模式
        if row['Tags'] in ['Speaking', 'Conversation']:
            if not groq_api_key:
                st.error("⚠️ 請在左側選單輸入 API Key 才能使用")
            else:
                st.markdown('<p class="hint-text">🎙️ 點擊下方麥克風錄音</p>', unsafe_allow_html=True)
                audio_data = mic_recorder(start_prompt="開始錄音", stop_prompt="⏹️ 完成送出", key=f'mic_{current_idx}')
                
                if audio_data:
                    user_text = transcribe_with_groq(groq_api_key, audio_data['bytes'])
                    st.session_state.q_user_text = user_text
                    
                    if row['Tags'] == 'Conversation':
                        grade, msg = llm_grade_answer(groq_api_key, user_text, row['Sentences'], target_answer)
                        st.session_state.q_grade = grade
                        st.session_state.q_ai_msg = msg
                    else:
                        st.session_state.q_grade = float(fuzz.ratio(user_text.lower(), target_answer.lower()))
                    
                    st.session_state.q_processed = True
                    st.rerun()

        # B. 文字輸入模式
        else: 
            user_input = st.text_input("✍️ 請輸入答案:", key=f"input_{current_idx}")
            if st.button("送出檢查", key=f"btn_{current_idx}", type="primary", use_container_width=True):
                st.session_state.q_user_text = user_input
                
                final_target = target_answer
                # Phrases 特殊處理
                if row['Tags'] == 'Phrases' and pd.isna(row['Answers']):
                     match = re.search(r'\[(.*?)\]', row['Sentences'])
                     if match: final_target = match.group(1)
                
                if user_input.strip().lower() == final_target.strip().lower():
                    st.session_state.q_grade = 100.0
                else:
                    st.session_state.q_grade = float(fuzz.ratio(user_input.lower(), final_target.lower()))
                
                st.session_state.q_processed = True
                st.rerun()

    # === [Result Phase] 結果與檢討階段 ===
    else:
        score_color = '#10b981' if st.session_state.q_grade >= 80 else '#f59e0b'
        st.markdown(f"""
        <div class="result-box">
            <h3 style="margin:0; color: {score_color}">
                得分: {st.session_state.q_grade:.2f}
            </h3>
            <p style="margin-top:10px; font-size:1.1rem;">你的回答: <b>{st.session_state.q_user_text}</b></p>
            {f'<p style="color:#64748b; font-size:0.9em;">AI 評語: {st.session_state.q_ai_msg}</p>' if st.session_state.q_ai_msg else ''}
            <p style="margin-top:10px; font-size:1.1rem;">✅ {target_answer}</b></p>
            ✅ {target_answer}
        </div>
        """, unsafe_allow_html=True)

        # 補救顯示 - 音檔 (如果是 Question/Phrases，現在才給聽)
        # 注意：這裡的音檔已經改成唸 Answer 了
        if not show_audio_initially:
            st.caption("🔊 音檔 (答案):")
            st.audio(audio_bytes, format='audio/mp3')

        # 補救顯示 - 原文
        if not show_text_initially:
            st.markdown(f"<div style='margin-top:15px; text-align:left; padding:15px; background:#f1f5f9; border-radius:10px;'><b>📖 原文參考:</b><br>{row['Sentences']}</div>", unsafe_allow_html=True)

        # 存檔邏輯
        if st.session_state.q_saved_idx != current_idx:
            if st.session_state.q_grade >= 80:
                df.at[current_idx, 'Times'] += 1
                days_to_add = int(df.at[current_idx, 'Times'])
                df.at[current_idx, 'Next'] = today + timedelta(days=days_to_add)
                st.toast("🎉 Level Up! 下次複習時間延後")
            else:
                df.at[current_idx, 'Times'] = 0 
                df.at[current_idx, 'Next'] = today 
                st.toast("💪 繼續加油！保持在今日進度")
            save_data(df)
            st.session_state.q_saved_idx = current_idx

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("➡️ 下一題 (Next Card)", type="primary", use_container_width=True, on_click=go_next_question)

    st.markdown('</div>', unsafe_allow_html=True)