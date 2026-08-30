import subprocess


class AudioDecoder:

    @staticmethod
    def decode_to_pcm(audio_bytes: bytes) -> bytes:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if process.returncode != 0:
            raise ValueError(
                f"Unable to decode audio: "
                f"{process.stderr.decode(errors='ignore')}"
            )

        return process.stdout