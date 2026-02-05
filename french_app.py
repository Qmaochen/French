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
    /* 隱藏預設元件 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* 全局樣式 - 加深文字顏色 */
    .stApp {
        background-color: #f8f9fa;
        color: #111827; /* 全局深色字 */
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
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); /* 陰影稍微加深 */
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
        color: #3730a3; /* 標籤文字加深 */
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    /* 主要題目：純黑、大字 */
    .big-font {
        font-size: 22px !important;
        font-weight: 800; /* 加粗 */
        color: #000000;   /* 純黑 */
        line-height: 1.5;
        margin: 20px 0;
    }
    
    /* 提示文字：深灰 */
    .hint-text {
        color: #374151; /* 深灰 */
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
        color: #000000; /* 按鈕文字強制黑色 */
    }
    
    div.stTextInput > div > div > input {
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #cbd5e1; /* 邊框加深 */
        text-align: center;
        font-size: 1.1rem;
        color: #00008B; /* 輸入文字強制黑色 */
        font-weight: 600;
    }

    iframe {
        border: none !important;
        margin-bottom: 10px;
    }
    
    .result-box {
        background-color: #f1f5f9; /* 背景稍微加深一點點 */
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        text-align: left;
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

import json 

def llm_grade_answer(api_key, user_text, context_text, correct_answer):
    if not api_key: return 0.0, "請輸入 API Key"
    
    client = Groq(api_key=api_key)
    
    # 修改 Prompt：要求 JSON 格式，並強調給予修正建議
    prompt = f"""
    You are a strictly helpful French language tutor.
    
    Scenario/Context: "{context_text}"
    Reference Answer: "{correct_answer}"
    User's Input: "{user_text}"
    
    Task:
    1. Compare the User's Input with the Reference Answer (if provided) or judge based on natural French conversation standards.
    2. Check for grammar errors, wrong vocabulary, or unnatural phrasing.
    3. Grade from 0 to 100.
    4. Provide a "feedback" string:
       - If perfect: say "Parfait !"
       - If there are errors: Provide the CORRECTED sentence and a very brief explanation (in Traditional Chinese or English).
    
    IMPORTANT: You must return ONLY a valid JSON object. Do not add markdown code blocks.
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
            # 加上這個參數強制讓模型輸出 JSON，降低格式錯誤機率
            response_format={"type": "json_object"} 
        )
        
        content = chat.choices[0].message.content
        
        # 解析 JSON
        data = json.loads(content)
        score = float(data.get("score", 0.0))
        feedback = data.get("feedback", "No feedback provided.")
        
        return score, feedback

    except json.JSONDecodeError:
        # 如果 JSON 解析失敗，嘗試用舊方法的 regex 抓分數做為備案
        match = re.search(r'\d+(\.\d+)?', content)
        fallback_score = float(match.group()) if match else 0.0
        return fallback_score, "格式解析錯誤，但已記錄分數。"
        
    except Exception as e:
        return 0.0, f"AI Error: {str(e)}"
    
def play_hidden_sound(text):
    """生成語音並隱藏播放，完全不顯示播放器"""
    try:
        # 1. 生成語音 (法文)
        audio_bytes = asyncio.run(play_audio(text))
        
        # 2. 轉成 Base64
        b64 = base64.b64encode(audio_bytes).decode()
        
        # 3. 嵌入隱藏的 HTML (沒有 controls 屬性，且 style 設為 none)
        md = f"""
            <audio autoplay="true" style="display:none;">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
            </audio>
        """
        # 4. 寫入網頁
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

df = st.session_state.df

# --- 修正後的邏輯 ---

# 1. 確保日期格式一致 (先把 Next 轉成純日期，避免時間戳導致比較錯誤)
df['Next'] = pd.to_datetime(df['Next']).dt.date
today = datetime.now().date()

# 2. 初始化 Demo 模式狀態 (如果還沒設定過)
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# 3. 根據狀態決定 due_indices (這是關鍵修正)
if st.session_state.demo_mode:
    # 如果在 Demo 模式，取出全部題目
    due_indices = df.index.tolist()
else:
    # 正常模式，只取到期題目
    due_indices = df[df['Next'] <= today].index.tolist()

# === 顯示邏輯 ===

if not due_indices:
    # 這裡代表：正常模式下沒有到期題目，且沒有開啟 Demo 模式
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 60px;">
        <h1 style="color:#10b981; font-size: 3rem;">Bravo! 🎉</h1>
        <p style="font-size: 1.2rem; color: #64748b;">今天的複習進度已全部完成。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # 按下按鈕只負責改變 session_state，然後 rerun
        if st.button("🔄 強制複習全部 (Demo Mode)", use_container_width=True):
            st.session_state.demo_mode = True  # 設定旗標
            st.rerun() # 重跑後會進入下方的 else 區塊

else:
    # --- 進入複習流程 ---

    # (選用) 在 Demo 模式下顯示一個退出的按鈕
    if st.session_state.demo_mode:
        st.info("💡 目前為 Demo 模式 (複習全部題目)")
        if st.button("❌ 退出 Demo 模式"):
            st.session_state.demo_mode = False
            st.session_state.current_q_idx = None # 重置題目指標
            st.rerun()

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

    # 進度條 (計算方式：總題數 - 剩餘待複習數)
    total_len = len(df) if st.session_state.demo_mode else len(df) # 簡化邏輯，分母通常用總題庫數較直觀
    # 如果你希望進度條在 Demo 模式下針對「這次複習」顯示，可以調整分母，但這裡維持原邏輯
    progress_val = 1.0 - (len(due_indices) / len(df)) if len(df) > 0 else 0
    
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
    # 記得要在開頭 import BytesIO (你原本的程式碼第 6 行已經有 import 了，所以直接用)
        st.audio(BytesIO(audio_bytes), format='audio/mpeg')

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
                    # 1. 先轉錄語音
                    user_text = transcribe_with_groq(groq_api_key, audio_data['bytes'])
                    st.session_state.q_user_text = user_text
                    
                    # 2. 判斷評分邏輯
                    # 如果是 Conversation 或 Speaking 且有 API Key，就用 AI 評分並給建議
                    if row['Tags'] in ['Conversation', 'Speaking'] and groq_api_key:
                        grade, msg = llm_grade_answer(groq_api_key, user_text, row['Sentences'], target_answer)
                        st.session_state.q_grade = grade
                        st.session_state.q_ai_msg = msg  # 儲存 AI 的修正建議
                    
                    # 其他情況 (如單字題、或是沒有 API Key)，使用單純的文字比對
                    else:
                        st.session_state.q_grade = float(fuzz.ratio(user_text.lower(), target_answer.lower()))
                        st.session_state.q_ai_msg = ""   # ⚠️ 重要：必須清空建議，避免上一題的 AI 訊息殘留
                    
                    # 3. 完成處理並重整頁面
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
            ✅ {target_answer}
        </div>
        """, unsafe_allow_html=True)

        # 補救顯示 - 音檔 (如果是 Question/Phrases，現在才給聽)
        # 注意：這裡的音檔已經改成唸 Answer 了
        if not show_audio_initially:
            st.caption("🔊 音檔 (答案):")
            st.audio(BytesIO(audio_bytes), format='audio/mpeg')

        # 補救顯示 - 原文
        if not show_text_initially:
            st.markdown(f"<div style='margin-top:15px; text-align:left; padding:15px; background:#f1f5f9; border-radius:10px;'><b>📖 原文參考:</b><br>{row['Sentences']}</div>", unsafe_allow_html=True)

        # 存檔邏輯
        if st.session_state.q_saved_idx != current_idx:
            if st.session_state.q_grade >= 80:
                praises = ["Très bien !", "Excellent !", "Bravo !", "Magnifique !", "C'est super !"]
                praise_text = random.choice(praises)
                
                # 直接呼叫我們剛寫好的隱形函數
                play_hidden_sound(praise_text)
                
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