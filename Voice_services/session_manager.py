from .audio_session import AudioSession


class SessionManager:

    def __init__(self):

        self.sessions = {}

    def create(
        self,
        session_id: str
    ):

        session = AudioSession(
            session_id
        )

        self.sessions[
            session_id
        ] = session

        return session

    def get(
        self,
        session_id: str
    ):

        return self.sessions.get(
            session_id
        )

    def remove(
        self,
        session_id: str
    ):

        self.sessions.pop(
            session_id,
            None
        )


session_manager = SessionManager()