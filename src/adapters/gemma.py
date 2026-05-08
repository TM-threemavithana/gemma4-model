import litert_lm
from typing import Optional, Dict, Any
from src.config.settings import MODEL_PATH, BACKEND, MAX_TOKENS

BACKEND_MAP = {
    "cpu": litert_lm.Backend.CPU,
    "gpu": getattr(litert_lm.Backend, "GPU", litert_lm.Backend.CPU),
    "npu": getattr(litert_lm.Backend, "NPU", litert_lm.Backend.CPU),
}

class GemmaAdapter:
    def __init__(self, model_path: str = MODEL_PATH, backend: str = BACKEND, max_tokens: Optional[int] = MAX_TOKENS):
        self.model_path = model_path
        self.backend_str = backend
        self.max_tokens = max_tokens
        self.engine: Optional[litert_lm.Engine] = None
        self.has_async_send = False

    def load(self):
        kwargs = {}
        if self.max_tokens is not None:
            kwargs["max_num_tokens"] = self.max_tokens

        self.engine = litert_lm.Engine(
            self.model_path,
            backend=BACKEND_MAP.get(self.backend_str, litert_lm.Backend.CPU),
            **kwargs,
        )

        # Probe for async streaming support
        try:
            with self.engine.create_conversation() as _probe_conv:
                self.has_async_send = callable(getattr(_probe_conv, "send_message_async", None))
        except Exception:
            self.has_async_send = False

    def create_conversation(self, **kwargs):
        if not self.engine:
            self.load()
        return self.engine.create_conversation(**kwargs)

    def close(self):
        if self.engine:
            self.engine.close()
            self.engine = None
