from aisuite.framework.message import Message
from typing import Literal, Optional


class Choice:
    finish_reason: Optional[Literal["stop", "tool_calls"]]
    message: Optional[Message]
