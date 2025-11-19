"""
FastAPI Server — Enhanced for Transcript + Base64 Audio
Matches EXACTLY the Zangoh Mid-Session Requirements
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
import base64

from src.pipeline import (
    AudioSupportPipeline,
    create_pipeline,
    PipelineConfig,
    TranscriptData
)

# ------------------------------------------------------
# MODELS
# ------------------------------------------------------

class TextRequest(BaseModel):
    text: str
    parameters: Optional[Dict[str, Any]] = {}


class EnhancedAudioResponse(BaseModel):
    success: bool
    audio_response: str
    transcript: Dict[str, str]
    processing_time_ms: int


class EnhancedTextResponse(BaseModel):
    success: bool
    response_text: str
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    message: str


# ------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------

app = FastAPI(
    title="Audio Support Agent API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[AudioSupportPipeline] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------
# STARTUP — CREATE PIPELINE
# ------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Initialize STT + LLM + TTS.
    """

    global pipeline
    try:
        logger.info("Booting pipeline...")

        stt_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "whisper-1"
        }

        llm_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
        }

        tts_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "voice": "alloy"
        }

        pipeline = await create_pipeline(
            stt_config,
            llm_config,
            tts_config,
            enable_logging=True
        )

        logger.info("Pipeline initialized.")

    except Exception as e:
        logger.error(f"Pipeline failed to start: {e}")


# ------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    global pipeline

    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            message="Pipeline not initialized",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False,
            }
        )

    components = await pipeline.health_check()
    healthy = all(components.values())

    return HealthResponse(
        status="healthy" if healthy else "unhealthy",
        message="OK" if healthy else "Some components not ready",
        components=components
    )


# ------------------------------------------------------
# TEXT CHAT — RETURNS TEXT + TIMING
# ------------------------------------------------------

@app.post("/chat/text", response_model=EnhancedTextResponse)
async def chat_text(request: TextRequest):
    global pipeline

    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    try:
        response_text, time_ms = await pipeline.process_text_with_timing(
            request.text
        )

        return EnhancedTextResponse(
            success=True,
            response_text=response_text,
            processing_time_ms=time_ms
        )

    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------
# AUDIO CHAT — RETURNS JSON WITH BASE64 AUDIO + TRANSCRIPT
# ------------------------------------------------------

@app.post("/chat/audio", response_model=EnhancedAudioResponse)
async def chat_audio(audio: UploadFile = File(...)):
    global pipeline

    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(400, "Empty audio file")

        audio_out, transcript, time_ms = await pipeline.process_audio_with_transcript(
            audio_bytes
        )

        # convert audio to base64
        audio_b64 = base64.b64encode(audio_out).decode("utf-8")

        return EnhancedAudioResponse(
            success=True,
            audio_response=audio_b64,
            transcript={
                "user_input": transcript.user_input,
                "agent_response": transcript.agent_response
            },
            processing_time_ms=time_ms
        )

    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------
# ROOT
# ------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Audio Customer Support Agent API",
        "version": "2.0.0"
    }


# ------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
