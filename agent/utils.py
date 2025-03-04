from langchain_core.tools import BaseTool
import json
from IPython.display import display, Markdown
from typing import Optional
from langchain_core.messages import BaseMessage

def format_tools_description(tools: list[BaseTool]) -> str:
    """Format the tools description for the prompt."""
    tool_descriptions = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            args_desc = ""
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    # Try Pydantic v2 method first
                    schema = tool.args_schema.model_json_schema()
                except AttributeError:
                    # Fallback to v1 method
                    schema = tool.args_schema.schema()
                args_desc = f"\nInput arguments: {json.dumps(schema.get('properties', {}), indent=2)}"
            
            tool_descriptions.append(
                f"- {tool.name}: {tool.description}{args_desc}"
            )
    return "\n\n".join(tool_descriptions)

async def print_stream(app, input: BaseMessage) -> Optional[BaseMessage]:
    display(Markdown("## New research running"))
    display(Markdown(f"### Input:\n\n{input.content}\n\n"))
    display(Markdown("### Stream:\n\n"))

    # Stream the results 
    all_messages = []
    async for chunk in app.astream(
        {"messages": [input]},
        stream_mode="updates"
    ):
        for updates in chunk.values():
            if messages := updates.get("messages"):
                all_messages.extend(messages)
                for message in messages:
                    message.pretty_print()
                    print("\n\n")
 
    # Return the last message if any
    if not all_messages:
        return None
    return all_messages[-1]

async def call_stream(app, input: BaseMessage) -> Optional[BaseMessage]:
    # Stream the results 
    all_messages = []
    async for chunk in app.astream(
        {"messages": [input]},
        stream_mode="updates"
    ):
        for updates in chunk.values():
            if messages := updates.get("messages"):
                all_messages.extend(messages)
    # Return the last message if any
    if not all_messages:
        return None
    return all_messages[-1]
