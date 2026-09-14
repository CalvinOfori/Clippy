import streamlit as st
import json
import re
import base64
from pathlib import Path
from datetime import datetime

# 1. BULLETPROOF ABSOLUTE PATHS
BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "chat_histories"
HISTORY_DIR.mkdir(exist_ok=True)

STAGING_DIR = BASE_DIR / "staged_assets"
STAGING_DIR.mkdir(exist_ok=True)

ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
LOGO_PATH = ASSETS_DIR / "clippy_logo.svg"


def _load_logo_b64() -> str | None:
    """Returns base64-encoded SVG data if the logo file exists, else None
    (so the sidebar can fall back to text-only if it's missing)."""
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return None

# 2. IMPORT YOUR ACTUAL AGENT
try:
    from post_production_agent import agent
except ImportError:
    agent = None

st.set_page_config(page_title="Clippy — AI Post-Production", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# 3. GEMINI-INSPIRED EMERALD THEME STYLING
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
html, body { background-color: #0d1110 !important; min-height: 100vh; }
.stApp { background: radial-gradient(circle at 50% 38%, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.05) 30%, #0d1110 75%); background-color: #0d1110; background-attachment: fixed; min-height: 100vh; color: #f1f5f9; }
/* FIX: only hide Streamlit's footer branding, not the header — the header
   contains the sidebar collapse/expand toggle button. Hiding the whole
   header (as the previous version did with "header, footer { visibility:
   hidden; }") removes that toggle entirely, so if the sidebar ever
   collapses there's no way to bring it back. Making the header transparent
   keeps the clean look without losing the control. */
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1110; }
::-webkit-scrollbar-thumb { background: #1e2925; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #10b981; }

.hero-container { text-align: center; margin-top: 5vh; margin-bottom: 2.5rem; animation: fadeIn 0.8s ease-in-out; }
.hero-sparkle { font-size: 2.2rem; display: inline-block; filter: drop-shadow(0 0 16px rgba(16, 185, 129, 0.6)); margin-bottom: 0.5rem; }
.hero-title { font-size: 2.75rem; font-weight: 600; letter-spacing: -0.03em; background: linear-gradient(135deg, #ffffff 40%, #10b981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
.hero-subtitle { color: #6ee7b7; font-size: 1.05rem; font-weight: 400; opacity: 0.85; }

.suggestion-card { background: rgba(19, 26, 23, 0.7); border: 1px solid rgba(16, 185, 129, 0.18); border-radius: 16px; padding: 1.1rem; transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25); backdrop-filter: blur(8px); height: 100%; }
.suggestion-card:hover { border-color: rgba(16, 185, 129, 0.45); background: rgba(16, 185, 129, 0.06); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12); }
.card-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.card-title { font-weight: 600; font-size: 0.95rem; color: #f3f4f6; margin-bottom: 0.2rem; }
.card-desc { font-size: 0.8rem; color: #9ca3af; line-height: 1.35; }

div[data-testid="stBottomBlockContainer"] { background: transparent !important; }
div[data-testid="stChatInput"] { background-color: #141b17 !important; border: 1px solid rgba(16, 185, 129, 0.35) !important; border-radius: 32px !important; box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45), 0 0 16px rgba(16, 185, 129, 0.08) !important; padding: 6px 14px !important; }
div[data-testid="stChatInput"]:focus-within { border-color: #10b981 !important; box-shadow: 0 0 22px rgba(16, 185, 129, 0.28) !important; }
div[data-testid="stChatInput"] textarea { color: #f1f5f9 !important; background-color: transparent !important; font-size: 0.95rem !important; }
div[data-testid="stChatInput"] textarea::placeholder { color: #6ee7b7 !important; opacity: 0.55; }
div[data-testid="stSidebar"] { background-color: #0b0f0e !important; border-right: 1px solid rgba(16, 185, 129, 0.15) !important; }
.stChatMessage { background-color: rgba(19, 26, 23, 0.45) !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; border-radius: 18px !important; margin-bottom: 0.75rem !important; backdrop-filter: blur(6px); }

/* Claude-style sidebar chat list: plain text rows, subtle hover, active
   chat gets a soft highlight. We use Streamlit's own primary/secondary
   button types to distinguish active vs inactive rows, then restyle both
   so they look like flat list rows instead of buttons. */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    color: #cbd5e1 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 400 !important;
    padding: 0.4rem 0.6rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover {
    background: rgba(16, 185, 129, 0.10) !important;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="primary"] {
    background: rgba(16, 185, 129, 0.16) !important;
    border: none !important;
    color: #f1f5f9 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.6rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


def _chat_title_from_file(path: Path) -> str:
    """Claude-style title: derive from the first user message instead of
    showing a raw timestamp filename. Falls back to 'New chat' if empty
    or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for msg in data:
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    title = msg["content"].strip().replace("\n", " ")
                    return title[:40] + ("..." if len(title) > 40 else "")
    except Exception:
        pass
    return "New chat"

# 4. STATE INITIALIZATION & FILE PRE-CREATION
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_draft" not in st.session_state:
    st.session_state.active_draft = "Nothing_Phone_Short"

# Ensure the active session always has a file on disk to populate the sidebar
active_file_path = HISTORY_DIR / f"{st.session_state.session_id}.json"
if not active_file_path.exists():
    with open(active_file_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f)

# 5. SIDEBAR NAVIGATION
with st.sidebar:
    logo_b64 = _load_logo_b64()
    if logo_b64:
        st.markdown(
            f'''
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                <img src="data:image/svg+xml;base64,{logo_b64}" width="32" height="32">
                <span style="font-size:1.3rem; font-weight:600; color:#f1f5f9;">Clippy</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### ✨ **Clippy**")

    if st.button("➕ New chat", use_container_width=True, type="primary"):
        st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.messages = []
        with open(HISTORY_DIR / f"{st.session_state.session_id}.json", "w", encoding="utf-8") as f:
            json.dump([], f)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Recent**")

    # Load chat list, newest first
    history_files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)

    if history_files:
        for f in history_files:
            session_id = f.stem
            title = _chat_title_from_file(f)
            is_active = session_id == st.session_state.session_id

            row_col, del_col = st.columns([5, 1])
            with row_col:
                if st.button(
                    title,
                    key=f"chat_row_{session_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if not is_active:
                        st.session_state.session_id = session_id
                        try:
                            with open(f, "r", encoding="utf-8") as fh:
                                data = json.load(fh)
                                if isinstance(data, list):
                                    st.session_state.messages = data
                                elif isinstance(data, dict) and "role" in data:
                                    st.session_state.messages = [data]
                                else:
                                    st.session_state.messages = []
                        except Exception:
                            st.session_state.messages = []
                        st.rerun()
            with del_col:
                if st.button("🗑", key=f"del_{session_id}", help="Delete this chat"):
                    f.unlink(missing_ok=True)
                    if is_active:
                        st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("No recent chats.")

    st.markdown("---")

    with st.expander("⚙️ Agent Settings"):
        st.markdown("**Active CapCut Draft**")
        st.session_state.active_draft = st.text_input("Project Name", value=st.session_state.active_draft, label_visibility="collapsed")

        st.markdown("**Media Staging**")
        uploaded_images = st.file_uploader("Reference Keyframes", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        staged_image_paths = []
        if uploaded_images:
            for uploaded_file in uploaded_images:
                file_path = STAGING_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                staged_image_paths.append(str(file_path.resolve()))
            st.success(f"{len(staged_image_paths)} image(s) staged.")

        st.markdown("**Engine Config**")
        st.selectbox("Bedrock Model", ["amazon.nova-reel-v1:0"], index=0)
        st.selectbox("Reasoning Engine", ["amazon.nova-pro-v1:0"], index=0)

# 6. MAIN CHAT & HERO DISPLAY
if not st.session_state.messages:
    hero_logo_b64 = _load_logo_b64()
    if hero_logo_b64:
        hero_icon_html = (
            f'<img src="data:image/svg+xml;base64,{hero_logo_b64}" '
            f'width="56" height="56" style="filter: drop-shadow(0 0 16px rgba(16, 185, 129, 0.6));">'
        )
    else:
        hero_icon_html = '<div class="hero-sparkle">✨</div>'

    st.markdown(f"""
        <div class="hero-container">
            {hero_icon_html}
            <div class="hero-title">What should Clippy cut today</div>
            <div class="hero-subtitle">Clippy automates generative b-roll, semantic search, and CapCut timelines.</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="suggestion-card"><div class="card-icon">🎬</div><div class="card-title">Generate B-Roll Intro</div><div class="card-desc">6s cinematic render of the Nothing Phone transparent glass and glyphs.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="suggestion-card"><div class="card-icon">⚡</div><div class="card-title">Extract Viral Hooks</div><div class="card-desc">Transcribe via Whisper and extract 3 high-retention 15s clips with subtitles.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="suggestion-card"><div class="card-icon">🔍</div><div class="card-title">Semantic Scene Search</div><div class="card-desc">Multimodal embeddings search to locate when the host talks about Snapdragon.</div></div>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            avatar = "👤" if msg["role"] == "user" else "✨"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

# 7. AGENT INVOCATION & PIPELINE EXECUTION
prompt = st.chat_input("Ask Clippy to generate b-roll, slice hooks, or search scenes...")

if prompt:
    if not isinstance(st.session_state.messages, list):
        st.session_state.messages = []

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        if agent is None:
            err_msg = "⚠️ `agent` could not be imported from `post_production_agent.py`. Please check your agent instance name."
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
        else:
            with st.spinner("Clippy is working on your timeline..."):
                try:
                    augmented_prompt = prompt
                    if staged_image_paths:
                        augmented_prompt += f"\n[Staged Images: {staged_image_paths}]"

                    raw_response = agent(augmented_prompt)

                    if isinstance(raw_response, dict):
                        response_text = raw_response.get("output", json.dumps(raw_response, indent=2))
                    else:
                        response_text = str(raw_response)

                    response_text = re.sub(r"<thinking>.*?</thinking>\s*", "", response_text, flags=re.DOTALL)
                    response_text = re.sub(r"</?response>\s*", "", response_text).strip()

                except Exception as e:
                    response_text = f"❌ Error executing agent action: {str(e)}"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Save the file and instantly refresh the UI so the sidebar updates
                with open(HISTORY_DIR / f"{st.session_state.session_id}.json", "w", encoding="utf-8") as f:
                    json.dump(st.session_state.messages, f)
                st.rerun()