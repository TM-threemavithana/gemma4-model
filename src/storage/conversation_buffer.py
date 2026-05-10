import threading
import time
from dataclasses import dataclass
from typing import List

@dataclass
class Turn:
    speaker: str
    text: str
    turn_index: int
    timestamp: float

class ConversationBuffer:
    def __init__(self, session_id: str, caller_number: str):
        self.session_id = session_id
        self.caller_number = caller_number
        self._turns: List[Turn] = []
        self._lock = threading.Lock()
        self._turn_index = 0

    def add_user_turn(self, text: str):
        with self._lock:
            turn = Turn(
                speaker="user",
                text=text,
                turn_index=self._turn_index,
                timestamp=time.time()
            )
            self._turns.append(turn)
            self._turn_index += 1

    def add_assistant_turn(self, text: str):
        with self._lock:
            turn = Turn(
                speaker="assistant",
                text=text,
                turn_index=self._turn_index,
                timestamp=time.time()
            )
            self._turns.append(turn)
            self._turn_index += 1

    def get_history(self) -> List[Turn]:
        with self._lock:
            return list(self._turns)

    def get_plain_text(self) -> str:
        with self._lock:
            lines = []
            for turn in self._turns:
                speaker_label = "User" if turn.speaker == "user" else "Assistant"
                lines.append(f"{speaker_label}: {turn.text}")
            return "\n".join(lines)
