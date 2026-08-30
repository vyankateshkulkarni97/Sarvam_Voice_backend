import numpy as np


class AudioProcessor:

    @staticmethod
    def normalize(
        pcm_bytes: bytes
    ) -> bytes:

        audio = np.frombuffer(
            pcm_bytes,
            dtype=np.int16
        )

        if len(audio) == 0:

            return pcm_bytes

        audio_float = audio.astype(
            np.float32
        )

        max_value = np.max(
            np.abs(audio_float)
        )

        if max_value == 0:

            return pcm_bytes

        audio_float = (
            audio_float / max_value
        ) * 30000

        return audio_float.astype(
            np.int16
        ).tobytes()
        
        
        
import numpy as np


class AudioProcessor:

    SAMPLE_RATE = 16000

    @staticmethod
    def pcm16_to_float32(
        pcm_bytes: bytes
    ) -> np.ndarray:

        audio = np.frombuffer(
            pcm_bytes,
            dtype=np.int16
        )

        if len(audio) == 0:

            return np.array(
                [],
                dtype=np.float32
            )

        audio = (
            audio.astype(
                np.float32
            ) / 32768.0
        )

        return audio

    @staticmethod
    def normalize(
        audio: np.ndarray
    ) -> np.ndarray:

        if len(audio) == 0:

            return audio

        max_value = np.max(
            np.abs(audio)
        )

        if max_value > 0:

            audio = audio / max_value

        return audio