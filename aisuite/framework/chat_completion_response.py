from aisuite.framework.choice import Choice
from pydantic import BaseModel
from typing import Literal, Optional, List


class ChatCompletionResponse(BaseModel):
    """Used to conform to the response model of OpenAI"""

    choices: List[Choice] = [Choice()]  # Adjust the range as needed for more choices

