"""Local JSON-file session storage (spec section 3: "SQLite or local JSON
files for shot/session metadata at first"). One directory per session
under data/shots/<session_id>/, containing session.json plus uploaded
video files and any future debug artifacts.

Swapping this for SQLite later should only require rewriting this one
module -- routers call SessionStore, never the filesystem directly.
"""

from __future__ import annotations

from pathlib import Path

from golfie_core.schemas import Session

from golfie_api.config import SHOTS_DIR


class SessionNotFoundError(Exception):
    pass


class SessionStore:
    def __init__(self, root_dir: Path = SHOTS_DIR):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.root_dir / session_id

    def _session_file(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def create(self, **initial_fields) -> Session:
        session = Session(**initial_fields)
        self.session_dir(session.session_id).mkdir(parents=True, exist_ok=True)
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        self.session_dir(session.session_id).mkdir(parents=True, exist_ok=True)
        self._session_file(session.session_id).write_text(session.model_dump_json(indent=2))

    def load(self, session_id: str) -> Session:
        path = self._session_file(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"No session found with id {session_id!r}")
        return Session.model_validate_json(path.read_text())

    def list_ids(self) -> list[str]:
        if not self.root_dir.exists():
            return []
        return sorted(
            p.name for p in self.root_dir.iterdir() if p.is_dir() and (p / "session.json").exists()
        )


# Module-level singleton used by routers -- fine for a local-first
# single-process prototype; revisit if/when this moves to a real
# multi-worker deployment.
session_store = SessionStore()
