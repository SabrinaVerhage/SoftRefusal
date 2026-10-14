import os
from dataclasses import dataclass, field
from pathlib import Path


MODELS = {
    "gemma": {
        "repo": "google/gemma-3-12b-it",
        "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "kind": "chat",
    },
    "hermes": {
        "repo": "NousResearch/Hermes-4-14B",
        "revision": "d6ce765c8b83f847357b98254be079afa0c6ca76",
        "kind": "chat",
    },
    "flux": {
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "kind": "image",
    },
}


@dataclass
class Settings:
    data: Path = field(default_factory=lambda: Path(os.getenv("SOFT_REFUSAL_DATA", "runtime")))
    auth_file: Path = field(default_factory=lambda: Path(os.getenv("SOFT_REFUSAL_AUTH_FILE", "secrets/auth.json")))
    queue_limit: int = field(default_factory=lambda: int(os.getenv("SOFT_REFUSAL_QUEUE_LIMIT", "10")))
    per_credential_limit: int = field(default_factory=lambda: int(os.getenv("SOFT_REFUSAL_PER_CREDENTIAL_LIMIT", "3")))
    chat_input_limit: int = field(default_factory=lambda: int(os.getenv("SOFT_REFUSAL_CHAT_INPUT_LIMIT", "8192")))
    job_timeout: float = field(default_factory=lambda: float(os.getenv("SOFT_REFUSAL_JOB_TIMEOUT", "900")))
    queue_timeout: float = field(default_factory=lambda: float(os.getenv("SOFT_REFUSAL_QUEUE_TIMEOUT", "1800")))
    idle_unload: float = field(default_factory=lambda: float(os.getenv("SOFT_REFUSAL_IDLE_UNLOAD", "600")))
    hermes_enabled: bool = field(default_factory=lambda: os.getenv("SOFT_REFUSAL_ENABLE_HERMES", "0") == "1")
    accept_requests: bool = field(default_factory=lambda: os.getenv("SOFT_REFUSAL_ACCEPT_REQUESTS", "0") == "1")
    origins: list[str] = field(default_factory=lambda: [s.strip() for s in os.getenv("SOFT_REFUSAL_ORIGINS", "").split(",") if s.strip()])
    web_root: str = field(default_factory=lambda: os.getenv("SOFT_REFUSAL_WEB_ROOT", ""))

    def __post_init__(self):
        if not 1 <= self.queue_limit <= 100:
            raise ValueError("queue_limit must be between 1 and 100")
        if not 1 <= self.per_credential_limit <= 10 or not 1 <= self.chat_input_limit <= 8192:
            raise ValueError("Invalid per-credential or chat-input limit")
        if min(self.job_timeout, self.queue_timeout, self.idle_unload) <= 0:
            raise ValueError("timeouts must be positive")
        if "*" in self.origins:
            raise ValueError("Specify exact allowed website origins")
