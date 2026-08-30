import uuid
import time
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException
)

from Voice_app.Voice_services.session_manager import (
    session_manager
    
)

from Voice_app.Voice_services.audio_decoder import (
    AudioDecoder
)

from Voice_app.Voice_services.audio_processor import (
    AudioProcessor
)

from Voice_app.Voice_services.attribute_model import (
    attribute_inference
)


import logging

logger = logging.getLogger(
    "voice-service.audio"
)

router = APIRouter(
    prefix="/audio",
    tags=["Audio"]
)


MAX_CHUNK_SIZE = 256 * 1024

MIN_AUDIO_SECONDS = 3

SAMPLE_RATE = 16000

BYTES_PER_SAMPLE = 2

MIN_AUDIO_BYTES = (
    MIN_AUDIO_SECONDS
    * SAMPLE_RATE
    * BYTES_PER_SAMPLE
)

MAX_BUFFER_SIZE = 10 * 1024 * 1024


@router.post("/session")
async def create_session():

    session_id = str(
        uuid.uuid4()
    )

    session_manager.create(
        session_id
    )

    return {
        "session_id": session_id,
        "status": "created"
    }


@router.websocket(
    "/stream/{session_id}"
)
async def audio_stream(
    websocket: WebSocket,
    session_id: str
):

    await websocket.accept()

    session = session_manager.get(
        session_id
    )

    if session is None:

        await websocket.send_json({
            "status": "error",
            "message": "Invalid session ID"
        })

        await websocket.close(
            code=1008
        )

        return

    logger.info(
        "WebSocket connected session=%s",
        session_id
        )

    try:

        while True:

            chunk = (
                await websocket.receive_bytes()
            )

            if not chunk:

                continue

            # --------------------------------
            # Validate chunk
            # --------------------------------

            if len(chunk) > MAX_CHUNK_SIZE:

                await websocket.send_json({
                    "status": "error",
                    "message": (
                        "Audio chunk too large"
                    )
                })

                continue

            # --------------------------------
            # Add to session
            # --------------------------------

            try:

                session.add_chunk(
                    chunk
                )

            except ValueError as exc:

                await websocket.send_json({
                    "status": "error",
                    "message": str(exc)
                })

                break

            print(
                f"Session={session_id} "
                f"chunk={len(chunk)} "
                f"buffer={session.size()}"
            )

            # --------------------------------
            # Check minimum audio
            # --------------------------------

            if (
                session.size()
                < MIN_AUDIO_BYTES
            ):

                await websocket.send_json({
                    "status": "buffering",
                    "audio_bytes": session.size(),
                    "required_bytes": (
                        MIN_AUDIO_BYTES
                    )
                })

                continue

            # --------------------------------
            # Convert PCM16 → float32
            # --------------------------------

            pcm_audio = AudioDecoder.decode_to_pcm(
                session.get_audio()
            )

            audio = AudioProcessor.pcm16_to_float32(
                pcm_audio
            )

            audio = AudioProcessor.normalize(
                audio
            )

            # --------------------------------
            # Run age/gender inference
            # --------------------------------

            result = attribute_inference.predict(
                    audio,
                    SAMPLE_RATE
                )
            print(
                "Audio shape:",
                audio.shape
            )

            print(
                "Audio dtype:",
                audio.dtype
            )

            print(
                "Audio duration:",
                audio.size / SAMPLE_RATE,
                "seconds"
            )

            # --------------------------------
            # Send prediction
            # --------------------------------

            await websocket.send_json({

                "status": "prediction",

                "contact_id": session_id,

                "gender": {
                    "prediction": result.get(
                        "gender",
                        "unknown"
                    ),
                    "confidence": result.get(
                        "gender_confidence",
                        0.0
                    )
                },

                "age_bracket": {
                    "prediction": result.get(
                        "age_bracket",
                        "unknown"
                    ),
                    "confidence": result.get(
                        "age_confidence",
                        0.0
                    )
                },

                "processing_ms": result.get(
                    "processing_ms",
                    0
                ),

                "audio_quality": result.get(
                    "audio_quality",
                    "good"
                )

            })

            # Clear buffer after prediction
            session.clear()

    except WebSocketDisconnect:

        print(
            f"Client disconnected: "
            f"{session_id}"
        )

    except Exception as exc:

        print(
            f"WebSocket error: {exc}"
        )

        try:

            await websocket.send_json({

                "status": "error",

                "message": str(exc)

            })

        except Exception:

            pass

    finally:

        session_manager.remove(
            session_id
        )
        
        
@router.post("/predict")
async def predict_audio(
    file: UploadFile = File(...)
):

    start_time = time.perf_counter()

    try:

        allowed_types = {
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
            "audio/mp3",
            "audio/ogg",
            "audio/webm"
        }

        if file.content_type not in allowed_types:

            raise HTTPException(
                status_code=415,
                detail={
                    "error": "unsupported_audio_format",
                    "message": "Unsupported audio format"
                }
            )


        # Read uploaded file

        audio_bytes = await file.read()


        if not audio_bytes:

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "empty_audio",
                    "message": "Audio file is empty"
                }
            )


        # Check maximum file size

        if len(audio_bytes) > MAX_BUFFER_SIZE:

            raise HTTPException(
                status_code=413,
                detail={
                    "error": "audio_too_large",
                    "message": (
                        "Audio file exceeds "
                        "maximum size"
                    )
                }
            )


        # ========================================
        # Decode audio to PCM16
        # ========================================

        pcm_audio = AudioDecoder.decode_to_pcm(
            audio_bytes
        )


        # ========================================
        # PCM16 → Float32
        # ========================================

        audio = AudioProcessor.pcm16_to_float32(
            pcm_audio
        )


        # ========================================
        # Normalize
        # ========================================

        audio = AudioProcessor.normalize(
            audio
        )


        # ========================================
        # Validate duration
        # ========================================

        duration = (
            len(audio) / SAMPLE_RATE
        )


        if duration < MIN_AUDIO_SECONDS:

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "insufficient_audio",

                    "message": (
                        f"At least "
                        f"{MIN_AUDIO_SECONDS} seconds "
                        "of audio is required"
                    ),

                    "duration": round(
                        duration,
                        2
                    )
                }
            )


        # ========================================
        # Age / Gender inference
        # ========================================

        result = attribute_inference.predict(
            audio,
            SAMPLE_RATE
        )


        # ========================================
        # Response
        # ========================================

        return {

            "contact_id": str(
                uuid.uuid4()
            ),

            "gender": {

                "prediction": result.get(
                    "gender",
                    "unknown"
                ),

                "confidence": result.get(
                    "gender_confidence",
                    0.0
                )
            },

            "age_bracket": {

                "prediction": result.get(
                    "age_bracket",
                    "unknown"
                ),

                "confidence": result.get(
                    "age_confidence",
                    0.0
                )
            },

            "processing_ms": result.get(
                "processing_ms",
                round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2
                )
            ),

            "audio_quality": result.get(
                "audio_quality",
                "insufficient"
            ),

            "audio_duration": round(
                duration,
                2
            )
        }


    except HTTPException:

        raise


    except Exception as exc:

        logger.exception(
            "REST inference error"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "inference_failed",
                "message": "Audio inference failed"
            }
        )
        
        
        
        
        
        