import io
import asyncio
import edge_tts

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


router = APIRouter(
    prefix="/text-to-audio",
    tags=["Text To Audio"]
)


VOICE_MAP = {
    "en-male": "en-IN-PrabhatNeural",
    "en-female": "en-IN-NeerjaNeural",

    "hi-male": "hi-IN-MadhurNeural",
    "hi-female": "hi-IN-SwaraNeural",

    "mr-male": "mr-IN-ManoharNeural",
    "mr-female": "mr-IN-AarohiNeural",

    "ta-male": "ta-IN-ValluvarNeural",
    "ta-female": "ta-IN-PallaviNeural",
}


class TextToAudioRequest(BaseModel):

    text: str

    voice: str = "en-male"


@router.post("/")
async def text_to_audio(
    request: TextToAudioRequest
):

    text = request.text.strip()
    print('request print :- ', request)

    if not text:

        raise HTTPException(
            status_code=400,
            detail="Text is required"
        )

    if len(text) > 5000:

        raise HTTPException(
            status_code=400,
            detail="Maximum text length is 5000 characters"
        )

    voice_name = VOICE_MAP.get(
        request.voice
    )

    if not voice_name:

        raise HTTPException(
            status_code=400,
            detail="Invalid voice"
        )

    try:

        audio_buffer = io.BytesIO()

        communicate = edge_tts.Communicate(
            text,
            voice_name
        )

        async for chunk in communicate.stream():

            if chunk["type"] == "audio":

                audio_buffer.write(
                    chunk["data"]
                )

        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",

            headers={
                "Content-Disposition":
                    "inline; filename=text-to-audio.mp3"
            }
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Unable to generate audio"
        )