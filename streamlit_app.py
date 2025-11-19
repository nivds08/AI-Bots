"""
Enhanced Streamlit UI for Audio Customer Support Agent (Updated for mid-session)
Handles:
    ✔ New JSON audio response format
    ✔ Base64 decoding
    ✔ Transcript display
    ✔ Processing-time display
"""

import streamlit as st
import requests
import json
import time
import io
import wave
import numpy as np
from typing import Dict, Any, Optional
import base64
from datetime import datetime

# Audio recording
try:
    import sounddevice as sd
    AUDIO_RECORDING_AVAILABLE = True
except ImportError:
    AUDIO_RECORDING_AVAILABLE = False
    st.warning("Audio recording not available. Install: pip install sounddevice")

DEFAULT_SERVER_URL = "http://localhost:8000"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1


# -------------------------------------------------------------
# Session State Init
# -------------------------------------------------------------
def init_session_state():
    if "server_url" not in st.session_state:
        st.session_state.server_url = DEFAULT_SERVER_URL
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "server_status" not in st.session_state:
        st.session_state.server_status = "Unknown"
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None


# -------------------------------------------------------------
# API Helpers
# -------------------------------------------------------------
def check_server_status(server_url: str) -> Dict[str, Any]:
    """Hit / and /health"""
    try:
        root = requests.get(f"{server_url}/", timeout=5)
        health = requests.get(f"{server_url}/health", timeout=5)

        return {
            "server_running": True,
            "root_info": root.json() if root.status_code == 200 else {},
            "health_info": health.json() if health.status_code == 200 else {}
        }
    except:
        return {"server_running": False}


def send_text_message(server_url: str, text: str) -> Dict[str, Any]:
    payload = {"text": text, "parameters": {}}

    try:
        res = requests.post(
            f"{server_url}/chat/text",
            json=payload,
            timeout=30
        )
        if res.status_code == 200:
            return {"success": True, "data": res.json()}
        else:
            return {"success": False, "error": res.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_audio_message(server_url: str, audio_bytes: bytes) -> Dict[str, Any]:
    """
    Updated for NEW JSON audio response:
    Returns:
        success
        audio_response (decoded bytes)
        transcript: { user_input, agent_response }
        processing_time_ms
    """
    try:
        files = {
            "audio": ("audio.wav", audio_bytes, "audio/wav")
        }

        res = requests.post(f"{server_url}/chat/audio", files=files, timeout=60)

        if res.status_code != 200:
            return {"success": False, "error": res.text}

        data = res.json()

        if not data.get("success"):
            return {"success": False, "error": "Pipeline error"}

        # decode base64 audio
        b64_audio = data.get("audio_response", "")
        audio_decoded = base64.b64decode(b64_audio)

        return {
            "success": True,
            "audio_bytes": audio_decoded,
            "transcript": data.get("transcript"),
            "processing_time_ms": data.get("processing_time_ms")
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# -------------------------------------------------------------
# Audio Recording
# -------------------------------------------------------------
def record_audio():
    if not AUDIO_RECORDING_AVAILABLE:
        st.error("Recording not available")
        return None

    try:
        st.info("Recording for 10 seconds...")
        duration = 10
        audio = sd.rec(int(duration * AUDIO_SAMPLE_RATE),
                       samplerate=AUDIO_SAMPLE_RATE,
                       channels=AUDIO_CHANNELS,
                       dtype=np.float32)
        sd.wait()

        int16_audio = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(AUDIO_CHANNELS)
            wav.setsampwidth(2)
            wav.setframerate(AUDIO_SAMPLE_RATE)
            wav.writeframes(int16_audio.tobytes())

        return buffer.getvalue()

    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None


def audio_player_ui(label, audio_bytes):
    st.audio(audio_bytes, format="audio/mp3")
    st.download_button(
        f"Download {label}",
        audio_bytes,
        file_name="response.mp3",
        mime="audio/mpeg"
    )


# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------
def main():
    st.set_page_config(page_title="Audio Support Agent", layout="wide")
    init_session_state()

    st.title(" Audio Customer Support Agent (Enhanced Version)")

    with st.sidebar:
        st.header("Server Config")
        st.session_state.server_url = st.text_input(
            "Server URL", st.session_state.server_url
        )

        if st.button("Check Server Health"):
            st.session_state.server_status = check_server_status(
                st.session_state.server_url
            )

        if isinstance(st.session_state.server_status, dict):
            if st.session_state.server_status.get("server_running"):
                st.success("Server Running")
            else:
                st.error("Server not reachable")

    tab1, tab2, tab3 = st.tabs(["Text Chat", "Audio Chat", "Health Monitor"])

    # ---------------------------------------------------------
    # TAB 1 — TEXT CHAT
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Text Chat")

        user_msg = st.text_input("Enter your question")

        if st.button("Send Text"):
            with st.spinner("Processing..."):
                response = send_text_message(st.session_state.server_url, user_msg)

            if response["success"]:
                ans = response["data"]
                st.markdown(f"**Agent:** {ans['response_text']}")
                st.caption(f"Processing: {ans['processing_time_ms']}ms")
            else:
                st.error(response["error"])

    # ---------------------------------------------------------
    # TAB 2 — AUDIO CHAT (UPDATED)
    # ---------------------------------------------------------
    with tab2:
        st.header(" Audio Chat Interface")
    st.markdown("Record or upload audio → send to /chat/audio → receive audio + transcript")

    col1, col2 = st.columns(2)

    # -------------------------- LEFT: AUDIO INPUT --------------------------
    with col1:
        st.subheader(" Audio Input")

        if AUDIO_RECORDING_AVAILABLE:
            if st.button("🎙️ Record Audio", key="record_audio"):
                with st.spinner("Recording up to 10 seconds..."):
                    st.session_state.audio_data = record_audio()
                if st.session_state.audio_data:
                    st.success("Recording complete!")

        if st.session_state.audio_data:
            st.audio(st.session_state.audio_data, format="audio/wav")

        uploaded_file = st.file_uploader(
            "Or upload an audio file",
            type=["wav", "mp3", "ogg", "flac"]
        )
        if uploaded_file:
            st.session_state.audio_data = uploaded_file.read()
            st.audio(st.session_state.audio_data)

    # -------------------------- RIGHT: AUDIO OUTPUT --------------------------
    with col2:
        st.subheader(" Agent Response")

        if st.session_state.audio_data:
            if st.button("➡️ Send Audio to Agent", type="primary"):
                with st.spinner("Processing audio through pipeline..."):
                    files = {"audio": ("input.wav", st.session_state.audio_data, "audio/wav")}
                    try:
                        response = requests.post(
                            "http://localhost:8000/chat/audio",
                            files=files,
                            timeout=60
                        )
                        if response.status_code == 200:
                            data = response.json()

                            # Decode audio
                            audio_b64 = data.get("audio_response", None)
                            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else None

                            transcript = data.get("transcript", {})
                            processing_time = data.get("processing_time_ms", None)

                            # Display audio output
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mpeg")

                                st.download_button(
                                    label="Download Agent Audio",
                                    data=audio_bytes,
                                    file_name="agent_response.mp3",
                                    mime="audio/mpeg"
                                )

                            # Display transcript
                            st.markdown("### 📝 Transcript")
                            st.write(f"**User:** {transcript.get('user_input', '—')}")
                            st.write(f"**Agent:** {transcript.get('agent_response', '—')}")

                            # Display processing time
                            if processing_time is not None:
                                st.caption(f"⚡ Processing time: {processing_time} ms")

                        else:
                            st.error(f"HTTP {response.status_code}: {response.text}")

                    except Exception as e:
                        st.error(f"Error contacting server: {str(e)}")
        else:
            st.info("Record or upload audio first.")

    # ---------------------------------------------------------
    # TAB 3 — HEALTH
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Health Monitor")
        status = st.session_state.server_status

        if isinstance(status, dict) and status.get("server_running"):
            st.json(status)
        else:
            st.error("Server not running.")


# -------------------------------------------------------------
if __name__ == "__main__":
    main()
