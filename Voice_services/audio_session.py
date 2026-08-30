import time


MAX_BUFFER_SIZE = 10 * 1024 * 1024


class AudioSession:

    def __init__(self, session_id: str):

        self.session_id = session_id

        self.buffer = bytearray()

        self.total_bytes = 0

        self.created_at = time.time()

        self.last_activity = time.time()

    def add_chunk(self, chunk: bytes):

        if (
            len(self.buffer) + len(chunk)
            > MAX_BUFFER_SIZE
        ):
            raise ValueError(
                "Audio buffer limit exceeded"
            )

        self.buffer.extend(chunk)

        self.total_bytes += len(chunk)

        self.last_activity = time.time()

    def get_audio(self) -> bytes:

        return bytes(self.buffer)

    def clear(self):

        self.buffer.clear()

    def size(self) -> int:

        return len(self.buffer)