import logging

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware


from Voice_app.Voice_api.Voice_audio import (
    router as audio_router
)

from Voice_app.Voice_api.text_to_audio import (
    router as text_to_audio_router
)


logger = logging.getLogger(
    "voice-service"
)


# ============================================
# FastAPI Application
# ============================================

Voice_app = FastAPI(

    title="Audio Attribute Service",

    version="1.0.0"
)


# ============================================
# CORS
# ============================================

Voice_app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================
# Routers
# ============================================

Voice_app.include_router(
    audio_router
)

Voice_app.include_router(
    text_to_audio_router
)


# ============================================
# Startup
# ============================================

@Voice_app.on_event("startup")
async def startup_event():

    logger.info(
        "Audio Attribute Service started"
    )


# ============================================
# Shutdown
# ============================================

@Voice_app.on_event("shutdown")
async def shutdown_event():

    logger.info(
        "Audio Attribute Service stopped"
    )


# ============================================
# Health
# ============================================

@Voice_app.get("/health")
async def health():

    return {

        "status": "healthy",

        "service":
            "audio-attribute-service"

    }


# ============================================
# Ready
# ============================================

@Voice_app.get("/ready")
async def ready():

    return {

        "status": "ready"

    }