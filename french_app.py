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
import base64
import json
from streamlit_gsheets import GSheetsConnection

# --- 🎨 1. UI 設定 ---
st.set_page_config(
    page_title="French SRS Master", 
    layout="centered", 
    page_icon="🇫🇷",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 隱藏預設元件 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* 全局樣式 - 加深文字顏色 */
    .stApp {
        background-color: #f8f9fa;
        color: #111827;
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
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 25px;
        border: 1px solid #e5e7eb;
        text-align: center;
        transition: all 0.3s ease;
    }

    .tag-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .big-font {
        font-size: 22px !important;
        font-weight: 800;
        color: #000000;
        line-height: 1.5;
        margin: 20px 0;
    }
    
    .hint-text {
        color: #374151;
        font-size: 1rem;
        margin-bottom: 15px;
        font-weight: 600;
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000;
    }
    
    div.stTextInput > div > div > input {
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #cbd5e1;
        text-align: center;
        font-size: 1.1rem;
        color: #00008B;
        font-weight: 600;
    }

    iframe {
        border: none !important;
        margin-bottom: 10px;
    }
    
    .result-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 檔案與資料庫 (Google Sheets 版本) ---

conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
    except Exception as e:
        st.error(f"無法讀取 Google Sheet: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    required = ['Sentences', 'Tags', 'Answers', 'Captions', 'Date', 'Times', 'Next']
    
    for col in required:
        if col not in df.columns:
            if col == 'Times':
                df[col] = 0
            elif col in ['Next', 'Date']:
                df[col] = pd.Timestamp.now().strftime('%Y-%m-%d')
            else:
                df[col] = ""

    # 強制轉型 Times
    df['Times'] = pd.to_numeric(df['Times'], errors='coerce').fillna(0).astype(int)
    
    # Next 先轉字串，後續由主程式處理
    df['Next'] = df['Next'].astype(str)
    
    return df.dropna(subset=['Sentences'])

def save_data(df):
    try:
        save_df = df.copy()
        # 確保 Next 是標準字串格式 YYYY-MM-DD
        if pd.api.types.is_datetime64_any_dtype(save_df['Next']):
             save_df['Next'] = save_df['Next'].dt.strftime('%Y-%m-%d')
        
        if 'Date' in save_df.columns:
             if pd.api.types.is_datetime64_any_dtype(save_df['Date']):
                save_df['Date'] = save_df['Date'].dt.strftime('%Y-%m-%d')

        conn.update(worksheet="Sheet1", data=save_df)
        st.cache_data.clear()
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
    You are a strictly helpful French language tutor.
    
    Scenario/Context: "{context_text}"
    Reference Answer: "{correct_answer}"
    User's Input: "{user_text}"
    
    Task:
    1. Compare the User's Input with the Reference Answer.
    2. Check for grammar errors, wrong vocabulary, or unnatural phrasing.
    3. Grade from 0 to 100.
    4. Provide a "feedback" string:
       - If perfect: say "Parfait !"
       - If there are errors: Provide the CORRECTED sentence and a very brief explanation (in Traditional Chinese or English).
    
    IMPORTANT: You must return ONLY a valid JSON object.
    Format:
    {{
        "score": 85.5,
        "feedback": "Your corrected sentence here. (Explanation)"
    }}
    """
    
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"} 
        )
        content = chat.choices[0].message.content
        data = json.loads(content)
        return float(data.get("score", 0.0)), data.get("feedback", "No feedback.")
    except Exception as e:
        return 0.0, f"AI Error: {str(e)}"
    
def play_hidden_sound(text):
    try:
        audio_bytes = asyncio.run(play_audio(text))
        b64 = base64.b64encode(audio_bytes).decode()
        md = f"""<audio autoplay="true" style="display:none;"><source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>"""
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        print(f"Sound Error: {e}")

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
    groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state.api_key_input)
    st.session_state.api_key_input = groq_api_key
    st.markdown("---")
    if st.button("🔄 重新載入題庫"):
        st.session_state.df = load_data()
        st.session_state.current_q_idx = None
        st.rerun()

st.title("🇫🇷 French SRS Master")

if not groq_api_key:
    st.info("💡 提示：請在左側選單輸入 API Key 以啟用語音功能。")

# === [Fix] 日期處理核心修正 (針對 2/5 格式) ===

df = st.session_state.df
today = pd.Timestamp.now().normalize()
current_year = today.year

# 定義智慧解析函數
def smart_parse_date(val):
    if pd.isna(val) or str(val).strip() in ['', 'NaT', 'None', 'nan']:
        return pd.NaT
    
    s = str(val).strip()
    
    # 針對 Google Sheets 常見的 "2/5" 這種只有月/日的格式
    # 正規表達式：1或2位數字 + 斜線 + 1或2位數字 (例如 2/5, 12/31)
    if re.match(r'^\d{1,2}/\d{1,2}$', s):
        s = f"{current_year}/{s}" # 自動補上今年
        
    try:
        # 轉成 datetime 物件
        return pd.to_datetime(s, dayfirst=False)
    except:
        return pd.NaT

try:
    # 應用解析函數
    df['Next'] = df['Next'].apply(smart_parse_date)
    
    # 只有真的讀不懂的 (NaT)，才設定為昨天 (強迫複習)
    df['Next'] = df['Next'].fillna(today - pd.Timedelta(days=1))
    
    # 正規化 (去掉時間)
    df['Next'] = df['Next'].dt.normalize()

except Exception as e:
    st.error(f"日期轉換發生嚴重錯誤: {e}")

# === 2. 初始化 Demo 模式 ===
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# === 3. 篩選邏輯 ===

if st.session_state.demo_mode:
    due_indices = df.index.tolist()
else:
    # 現在 Next 已經是乾淨的 Timestamp，可以直接比較
    mask = df['Next'] <= today
    due_indices = df[mask].index.tolist()

# === 4. 除錯顯示 (Debug) ===
with st.sidebar.expander("🕵️‍♀️ 日期格式檢查", expanded=False):
    st.write(f"系統日期 (Today): {today.date()}")
    st.write(f"待複習題數: {len(due_indices)}")
    debug_view = df[['Sentences', 'Next']].copy()
    debug_view['Is_Due'] = debug_view['Next'] <= today
    st.write("轉換後的資料預覽：")
    st.dataframe(debug_view.head(5))

# === 5. 顯示邏輯 (Bravo 畫面) ===

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
            st.session_state.demo_mode = True
            st.rerun()

else:
    if st.session_state.demo_mode:
        st.info(f"💡 目前為 Demo 模式 (共 {len(due_indices)} 題)")
        if st.button("❌ 退出 Demo 模式", use_container_width=True):
            st.session_state.demo_mode = False
            st.session_state.current_q_idx = None
            st.rerun()

    # === 6. 選題與變數重置 ===
    
    if st.session_state.current_q_idx is None or st.session_state.current_q_idx not in due_indices:
        st.session_state.current_q_idx = random.choice(due_indices)
    
    current_idx = st.session_state.current_q_idx
    row = df.loc[current_idx]

    if current_idx != st.session_state.last_q_idx:
        st.session_state.q_processed = False
        st.session_state.q_user_text = ""
        st.session_state.q_grade = 0.0
        st.session_state.q_ai_msg = ""
        st.session_state.last_q_idx = current_idx

    target_answer = get_target_answer(row)

    # 進度條
    total_count = len(df)
    remaining = len(due_indices)
    progress_val = 1.0 - (remaining / total_count) if total_count > 0 else 0.0
    
    st.progress(progress_val)
    c1, c2 = st.columns([1, 1])
    with c1: st.caption(f"📅 待複習: {len(due_indices)} 題")
    with c2: st.caption(f"🔥 連續答對: {row['Times']} 次")
    
    # 音檔生成
    audio_source_text = row['Sentences']
    if row['Tags'] in ['Question', 'Phrases']:
        if pd.notna(row['Answers']) and str(row['Answers']).strip() != "":
            audio_source_text = str(row['Answers'])

    audio_bytes = asyncio.run(play_audio(audio_source_text))

    # --- 🃏 題目卡片區域 ---
    st.markdown(f'<span class="tag-badge">{row["Tags"]}</span>', unsafe_allow_html=True)

    show_text_initially = False
    text_content = ""
    
    if row['Tags'] in ['Speaking', 'Question', 'Phrases']:
        show_text_initially = True
        if row['Tags'] == 'Question':
            if pd.notna(row['Captions']) and str(row['Captions']).strip() != "":
                text_content = str(row['Sentences']) + " " + "\n" + str(row['Captions'])
            else:
                text_content = str(row['Sentences'])
        elif row['Tags'] == 'Phrases':
            match = re.search(r'\[(.*?)\]', row['Sentences'])
            if match:
                text_content = row['Sentences'].replace(f"[{match.group(1)}]", " <span style='border-bottom: 2px solid #4f46e5; color: #4f46e5; font-weight:bold;'>______</span> ")
            else:
                text_content = row['Sentences'].replace(target_answer, " ______ ") if target_answer in row['Sentences'] else row['Sentences'] + " ______"
        else:
            text_content = row['Sentences'].replace('\n', '<br>')

    show_audio_initially = False
    if row['Tags'] in ['Writing', 'Conversation', 'Speaking']:
        show_audio_initially = True

    # === [Render] 渲染 ===
    if show_text_initially:
        st.markdown(f'<p class="big-font">{text_content}</p>', unsafe_allow_html=True)
    elif not st.session_state.q_processed:
        st.markdown('<p class="hint-text" style="font-size:1.2rem;">🎧 請仔細聆聽音檔回答問題...</p>', unsafe_allow_html=True)

    if show_audio_initially:
        st.audio(BytesIO(audio_bytes), format='audio/mpeg')

    # === [Input Phase] 輸入階段 ===
    if not st.session_state.q_processed:
        if row['Tags'] in ['Speaking', 'Conversation']:
            if not groq_api_key:
                st.error("⚠️ 請在左側選單輸入 API Key 才能使用")
            else:
                st.markdown('<p class="hint-text">🎙️ 點擊下方麥克風錄音</p>', unsafe_allow_html=True)
                audio_data = mic_recorder(start_prompt="開始錄音", stop_prompt="⏹️ 完成送出", key=f'mic_{current_idx}')
                
                if audio_data:
                    user_text = transcribe_with_groq(groq_api_key, audio_data['bytes'])
                    st.session_state.q_user_text = user_text
                    
                    if row['Tags'] in ['Conversation', 'Speaking'] and groq_api_key:
                        grade, msg = llm_grade_answer(groq_api_key, user_text, row['Sentences'], target_answer)
                        st.session_state.q_grade = grade
                        st.session_state.q_ai_msg = msg
                    else:
                        st.session_state.q_grade = float(fuzz.ratio(user_text.lower(), target_answer.lower()))
                        st.session_state.q_ai_msg = ""
                    
                    st.session_state.q_processed = True
                    st.rerun()
        else: 
            user_input = st.text_input("✍️ 請輸入答案:", key=f"input_{current_idx}")
            if st.button("送出檢查", key=f"btn_{current_idx}", type="primary", use_container_width=True):
                st.session_state.q_user_text = user_input
                final_target = target_answer
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
            ✅ {target_answer}
        </div>
        """, unsafe_allow_html=True)

        if not show_audio_initially:
            st.caption("🔊 音檔 (答案):")
            st.audio(BytesIO(audio_bytes), format='audio/mpeg')

        if not show_text_initially:
            st.markdown(f"<div style='margin-top:15px; text-align:left; padding:15px; background:#f1f5f9; border-radius:10px;'><b>📖 原文參考:</b><br>{row['Sentences']}</div>", unsafe_allow_html=True)

        if st.session_state.q_saved_idx != current_idx:
            if st.session_state.q_grade >= 80:
                praises = ["Très bien !", "Excellent !", "Bravo !", "Magnifique !", "C'est super !"]
                play_hidden_sound(random.choice(praises))
                
                df.at[current_idx, 'Times'] += 1
                days_to_add = int(df.at[current_idx, 'Times'])
                # 更新 Next (Timestamp 運算)
                new_date = today + pd.Timedelta(days=days_to_add)
                df.at[current_idx, 'Next'] = new_date
                
                st.toast(f"🎉 Level Up! 下次複習: {new_date.strftime('%Y-%m-%d')}")
            else:
                df.at[current_idx, 'Times'] = 0 
                df.at[current_idx, 'Next'] = today 
                st.toast("💪 繼續加油！保持在今日進度")
            
            save_data(df)
            st.session_state.q_saved_idx = current_idx

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("➡️ 下一題 (Next Card)", type="primary", use_container_width=True, on_click=go_next_question)

    st.markdown('</div>', unsafe_allow_html=True)