import time
import json
from langgraph.graph import END, StateGraph
from langchain_core.messages import (
    SystemMessage, 
    AIMessage, 
    ToolMessage, 
    HumanMessage
)
from .models import AgentState
from .prompts import (
    decision_making_prompt,
    planning_prompt,
    agent_prompt,
    judge_prompt
)
from .tools import tools, tools_dict
from .mistral_wrapper import mistral, decision_making_llm, agent_llm, judge_llm
from .utils import format_tools_description, call_stream

def decision_making_node(state: AgentState):
    """Entry point of the workflow using Mistral"""
    system_prompt = SystemMessage(content=decision_making_prompt)
    messages = [system_prompt] + state["messages"]
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    response = decision_making_llm(messages_str)
    output = {"requires_research": response.requires_research}
    if response.answer:
        output["messages"] = [AIMessage(content=response.answer)]
    return output

def router(state: AgentState):
    """Router directing the user query to the appropriate branch of the workflow."""
    if state["requires_research"]:
        return "planning"
    else:
        return "end"

def planning_node(state: AgentState):
    """Planning node using Mistral"""
    system_prompt = SystemMessage(content=planning_prompt.format(
        tools=format_tools_description(tools)
    ))
    messages = [system_prompt] + state["messages"]
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    response = mistral.llm.invoke(messages_str)
    return {"messages": [AIMessage(content=response)]}


def tools_node(state: AgentState):
    """Tool call node that executes the tools based on the plan."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_dict[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def agent_node(state: AgentState):
    """Agent node using Mistral with tool calling"""
    system_prompt = SystemMessage(content=agent_prompt)
    messages = [system_prompt] + state["messages"]
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    tool_outputs = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    
    if tool_outputs:
        # Process tool outputs and generate summary
        papers_info = [tool_msg.content for tool_msg in tool_outputs]
        summary_prompt = f"Process and summarize the following research results:\n{papers_info}"
        summary = agent_llm(summary_prompt)
        return {"messages": [AIMessage(content=summary)]}
    else:
        # Handle initial query or follow-up
        response = agent_llm(messages_str)
        
        # Check for tool calls in response
        if "TOOL:" in response:
            tool_lines = response.split("\n")
            tool_name = None
            tool_args = {}
            
            for line in tool_lines:
                if line.startswith("TOOL:"):
                    tool_name = line.replace("TOOL:", "").strip()
                elif line.startswith("ARGS:"):
                    try:
                        tool_args = json.loads(line.replace("ARGS:", "").strip())
                    except json.JSONDecodeError:
                        continue
            
            if tool_name and tool_args:
                return {"messages": [AIMessage(
                    content=response,
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"{tool_name}-{time.time()}"
                    }]
                )]}
        
        return {"messages": [AIMessage(content=response)]}

def should_continue(state: AgentState):
    """Check if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"

def judge_node(state: AgentState):
    """Judge node using Mistral"""
    system_prompt = SystemMessage(content=judge_prompt)
    messages = [system_prompt] + state["messages"]
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    response = judge_llm(messages_str)
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": state.get("num_feedback_requests", 0) + 1
    }
    if response.feedback:
        output["messages"] = [AIMessage(content=response.feedback)]
    return output

def final_answer_router(state: AgentState):
    """Router to end the workflow or improve the answer."""
    return "end" if state["is_good_answer"] else "planning"

# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("decision_making", decision_making_node)
workflow.add_node("planning", planning_node)
workflow.add_node("tools", tools_node)
workflow.add_node("agent", agent_node)
workflow.add_node("judge", judge_node)

# Set the entry point of the graph
workflow.set_entry_point("decision_making")

# Add edges between nodes
workflow.add_conditional_edges(
    "decision_making",
    router,
    {
        "planning": "planning",
        "end": END,
    }
)
workflow.add_edge("planning", "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "judge",
    },
)
workflow.add_conditional_edges(
    "judge",
    final_answer_router,
    {
        "planning": "planning",
        "end": END,
    }
)

# Compile the graph
app = workflow.compile()

async def run_agent(message: str) -> str:
    """Main entry point for the agent."""
    input_message = HumanMessage(content=message)
    final_answer = await call_stream(app, input_message)
    return final_answer.content
