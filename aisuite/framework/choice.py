from aisuite.framework.message import Message


class Choice:
    def __init__(self):
        self.finish_reason = None  # "stop", "tool_calls"
        self.message = Message()
