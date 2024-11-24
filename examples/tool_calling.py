import json
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append("../../aisuite")

# Load from .env file if available
load_dotenv(find_dotenv())


import aisuite as ai

client = ai.Client()


# Mock tool functions.
def get_current_temperature(location: str, unit: str):
    # Simulate fetching temperature from an API
    return {"location": location, "unit": unit, "temperature": 72}


def get_rain_probability(location: str):
    # Simulate fetching rain probability
    return {"location": location, "probability": 40}


# Create a tool object for each function
get_current_temperature_tool = Tool(
    name="get_current_temperature",
    description="Get the current temperature in a given location",
    func=get_current_temperature,
)

get_rain_probability_tool = Tool(
    name="get_rain_probability",
    description="Get the rain probability in a given location",
    func=get_rain_probability,
)

tool_manager = ToolManager()
tool_manager.add_tool(get_current_temperature)
tool_manager.add_tool(get_rain_probability)

messages = [
    {
        "role": "user",
        "content": "What is the current temperature in San Francisco in Celsius?",
    }
]
response = client.chat.completions.create(
    model="openai:gpt-4", messages=messages, tools=tool_manager.tools(format="openai")
)
print(response.choices[0].message)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        tool_result, result_as_message = tool_manager.execute_tool(tool_call)

        messages.append(response.choices[0].message)  # Model's function call message
        messages.append(result_as_message)
        # Send the tool response back to the model
        final_response = client.chat.completions.create(
            model="openai:gpt-4",
            messages=messages,
            tools=tool_manager.tools(format="openai"),
        )

        # Output the final response from the model
        print(final_response.choices[0].message.content)
