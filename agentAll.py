import io
import json
import os
import urllib3
import time
import asyncio
import pdfplumber
from dotenv import load_dotenv
from IPython.display import display, Markdown
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool, tool
# from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, ClassVar, Sequence, TypedDict, Optional
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, Any, Optional
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    ToolMessage, 
    AIMessage, 
    HumanMessage  # Add this import
)



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# You can set your own keys here
# os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["CORE_API_KEY"] ="cGFyD2T5qob6f18BxZvW4AaEMXPINwum"

class MistralWrapper:
    def __init__(self, model_name="mistral", temperature=0.0, verbose=False):
        callbacks = [StreamingStdOutCallbackHandler()] if verbose else []
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            callbacks=callbacks
        )
    
    def structured_output(self, pydantic_model):
        """Creates a wrapper that outputs in structured format"""
        schema = {
            "requires_research": "boolean",
            "answer": "string or null"
        } if pydantic_model == DecisionMakingOutput else {
            "is_good_answer": "boolean",
            "feedback": "string or null"
        }
        
        def wrapper(messages):
            formatted_prompt = f"""
            Based on the following conversation, provide output in this exact JSON format:
            {json.dumps(schema, indent=2)}

            Conversation:
            {messages}
            """
            response = self.llm.invoke(formatted_prompt)
            try:
                # Parse the response as JSON first
                json_response = json.loads(response)
                # Then create the Pydantic model
                return pydantic_model(**json_response)
            except Exception as e:
                print(f"Parsing error: {e}")
                return pydantic_model()
        
        return wrapper

    def with_tools(self, tools):
        """Creates a wrapper that can use tools"""
        tools_description = "\n".join([
            f"Tool {tool.name}: {tool.description}" 
            for tool in tools
        ])
        
        def wrapper(messages):
            formatted_prompt = f"""
            You have access to the following tools:
            {tools_description}
            
            To use a tool, output in this format:
            TOOL: <tool_name>
            ARGS: <tool_arguments_in_json>
            
            After getting tool output, provide your final response.
            
            Conversation:
            {messages}
            """
            return self.llm.invoke(formatted_prompt)
        
        return wrapper

# Initialize Mistral
mistral = MistralWrapper(verbose=False)


# Decision making prompt needs structure guidance
decision_making_prompt = """
You are an experienced scientific researcher.
Your goal is to help the user with their scientific research.

Based on the user query, decide if you need to perform a research or if you can answer the question directly.
- You should perform a research if the user query requires any supporting evidence or information.
- You should answer the question directly only for simple conversational questions, like "how are you?".

Output your decision in this exact format:
{
    "requires_research": true/false,
    "answer": "your direct answer here if no research needed, otherwise null"
}
"""

# Prompt to create a step by step plan to answer the user query
planning_prompt = """
# IDENTITY AND PURPOSE

You are an experienced scientific researcher.
Your goal is to make a new step by step plan to help the user with their scientific research .

Subtasks should not rely on any assumptions or guesses, but only rely on the information provided in the context or look up for any additional information.

If any feedback is provided about a previous answer, incorportate it in your new planning.


# TOOLS

For each subtask, indicate the external tool required to complete the subtask. 
Tools can be one of the following:
{tools}
"""

agent_prompt = """
# IDENTITY AND PURPOSE

You are an experienced scientific researcher. 
Your goal is to help the user with their scientific research. You have access to a set of external tools to complete your tasks.
Follow the plan you wrote to successfully complete the task.

Add extensive inline citations to support any claim made in the answer.

IMPORTANT: To use any tool, you must format your response exactly like this:
TOOL: <tool_name>
ARGS: {
    "argument_name": "argument_value"
}
Wait for the tool output before continuing.

# RESPONSE STRUCTURE

When processing tool outputs:
1. Analyze the results systematically
2. Provide a comprehensive summary with inline citations
3. Highlight key findings and methodologies
4. Identify research gaps or future directions
5. Suggest next steps if more research is needed

# EXTERNAL KNOWLEDGE

## CORE API

The CORE API has a specific query language that allows you to explore a vast papers collection and perform complex queries. See the following table for a list of available operators:

| Operator       | Accepted symbols         | Meaning                                                                                      |
|---------------|-------------------------|----------------------------------------------------------------------------------------------|
| And           | AND, +, space          | Logical binary and.                                                                           |
| Or            | OR                     | Logical binary or.                                                                            |
| Grouping      | (...)                  | Used to prioritise and group elements of the query.                                           |
| Field lookup  | field_name:value       | Used to support lookup of specific fields.                                                    |
| Range queries | fieldName(>, <,>=, <=) | For numeric and date fields, it allows to specify a range of valid values to return.         |
| Exists queries| _exists_:fieldName     | Allows for complex queries, it returns all the items where the field specified by fieldName is not empty. |

Use this table to formulate more complex queries filtering for specific papers, for example publication date/year.
Here are the relevant fields of a paper object you can use to filter the results:
{
  "authors": [{"name": "Last Name, First Name"}],
  "documentType": "presentation" or "research" or "thesis",
  "publishedDate": "2019-08-24T14:15:22Z",
  "title": "Title of the paper",
  "yearPublished": "2019"
}

Example queries:
- "machine learning AND yearPublished:2023"
- "maritime biology AND yearPublished>=2023 AND yearPublished<=2024"
- "cancer research AND authors:Vaswani, Ashish AND authors:Bello, Irwan"
- "title:Attention is all you need"
- "mathematics AND _exists_:abstract"

# ERROR HANDLING

If you encounter any issues:
1. Clearly state the problem
2. Try alternative query formulations
3. Use the ask-human-feedback tool if needed
4. Document any limitations or constraints found

Remember: Always maintain academic rigor and provide evidence-based responses.
"""


# Prompt for the judging step to evaluate the quality of the final answer
judge_prompt = """
You are an expert scientific researcher.
Your goal is to review the final answer you provided for a specific user query.

Look at the conversation history between you and the user. Based on it, you need to decide if the final answer is satisfactory or not.

A good final answer should:
- Directly answer the user query. For example, it does not answer a question about a different paper or area of research.
- Answer extensively the request from the user.
- Take into account any feedback given through the conversation.
- Provide inline sources to support any claim made in the answer.

In case the answer is not good enough, provide clear and concise feedback on what needs to be improved to pass the evaluation.
"""

class CoreAPIWrapper(BaseModel):
    """Simple wrapper around the CORE API."""
    base_url: ClassVar[str] = "https://api.core.ac.uk/v3"
    api_key: ClassVar[str] = os.environ["CORE_API_KEY"]

    top_k_results: int = Field(description = "Top k results obtained by running a query on Core", default = 1)

    def _get_search_response(self, query: str) -> dict:
        http = urllib3.PoolManager()

        # Retry mechanism to handle transient errors
        max_retries = 5    
        for attempt in range(max_retries):
            response = http.request(
                'GET',
                f"{self.base_url}/search/outputs", 
                headers={"Authorization": f"Bearer {self.api_key}"}, 
                fields={"q": query, "limit": self.top_k_results}
            )
            if 200 <= response.status < 300:
                return response.json()
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx response from CORE API: {response.status} {response.data}")

    def search(self, query: str) -> str:
        response = self._get_search_response(query)
        results = response.get("results", [])
        if not results:
            return "No relevant results were found"

        # Format the results in a string
        docs = []
        for result in results:
            published_date_str = result.get('publishedDate') or result.get('yearPublished', '')
            authors_str = ' and '.join([item['name'] for item in result.get('authors', [])])
            docs.append((
                f"* ID: {result.get('id', '')},\n"
                f"* Title: {result.get('title', '')},\n"
                f"* Published Date: {published_date_str},\n"
                f"* Authors: {authors_str},\n"
                f"* Abstract: {result.get('abstract', '')},\n"
                f"* Paper URLs: {result.get('sourceFulltextUrls') or result.get('downloadUrl', '')}"
            ))
        return "\n-----\n".join(docs)

class SearchPapersInput(BaseModel):
    """Input object to search papers with the CORE API."""
    query: str = Field(description="The query to search for on the selected archive.")
    max_papers: int = Field(
        default=1,
        description="The maximum number of papers to return. It's default to 1, but you can increase it up to 10 in case you need to perform a more comprehensive search.",
        ge=1,
        le=10
    )
    
    # Add this for Pydantic v2 compatibility
    model_config = ConfigDict(
        json_schema_extra={
            "properties": {
                "query": {"type": "string"},
                "max_papers": {"type": "integer", "minimum": 1, "maximum": 10}
            }
        }
    )
class DecisionMakingOutput(BaseModel):
    requires_research: bool = Field(default=False)
    answer: Optional[str] = None
    
class JudgeOutput(BaseModel):
    is_good_answer: bool = Field(default=False)
    feedback: Optional[str] = None

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

async def print_stream(app: CompiledStateGraph, input: HumanMessage) -> Optional[BaseMessage]:
    display(Markdown("## New research running"))
    display(Markdown(f"### Input:\n\n{input.content}\n\n"))
    display(Markdown("### Stream:\n\n"))

    # Stream the results 
    all_messages = []
    async for chunk in app.astream(
        {"messages": [input]},  # Now passing a proper HumanMessage
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

async def call_stream(app: CompiledStateGraph, input: HumanMessage) -> Optional[BaseMessage]:

    # Stream the results 
    all_messages = []
    async for chunk in app.astream(
        {"messages": [input]},  # Now passing a proper HumanMessage
        stream_mode="updates"
    ):
        for updates in chunk.values():
            if messages := updates.get("messages"):
                all_messages.extend(messages)
    # Return the last message if any
    if not all_messages:
        return None
    return all_messages[-1]




class AgentState(TypedDict):
    """The state of the agent during the paper research process."""
    requires_research: bool = False
    num_feedback_requests: int = 0
    is_good_answer: bool = False
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
@tool("search-papers", args_schema=SearchPapersInput)
def search_papers(query: str, max_papers: int = 1) -> str:
    """Search for scientific papers using the CORE API.

    Example:
    {"query": "Attention is all you need", "max_papers": 1}

    Returns:
        A list of the relevant papers found with the corresponding relevant information.
    """
    try:
        return CoreAPIWrapper(top_k_results=max_papers).search(query)
    except Exception as e:
        return f"Error performing paper search: {e}"

@tool("download-paper")
def download_paper(url: str) -> str:
    """Download a specific scientific paper from a given URL.

    Example:
    {"url": "https://sample.pdf"}

    Returns:
        The paper content.
    """
    try:        
        http = urllib3.PoolManager(
            cert_reqs='CERT_NONE',
        )
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        max_retries = 5
        for attempt in range(max_retries):
            response = http.request('GET', url, headers=headers)
            if 200 <= response.status < 300:
                pdf_file = io.BytesIO(response.data)
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx when downloading paper: {response.status} {response.text}")
    except Exception as e:
        return f"Error downloading paper: {e}"

@tool("ask-human-feedback")
def ask_human_feedback(question: str) -> str:
    """Ask for human feedback. You should call this tool when encountering unexpected errors."""
    return input(question)

tools = [search_papers, download_paper, ask_human_feedback]
tools_dict = {tool.name: tool for tool in tools}


# LLMS

decision_making_llm = mistral.structured_output(DecisionMakingOutput)
agent_llm = mistral.with_tools(tools)
judge_llm = mistral.structured_output(JudgeOutput)

# Modify the decision making node
def decision_making_node(state: AgentState):
    """Entry point of the workflow using Mistral"""
    system_prompt = SystemMessage(content=decision_making_prompt)
    messages = [system_prompt] + state["messages"]
    
    # Convert messages to string format for Mistral
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    response: DecisionMakingOutput = decision_making_llm(messages_str)
    output = {"requires_research": response.requires_research}
    if response.answer:
        output["messages"] = [AIMessage(content=response.answer)]
    return output

# Task router function
def router(state: AgentState):
    """Router directing the user query to the appropriate branch of the workflow."""
    if state["requires_research"]:
        return "planning"
    else:
        return "end"

# Modify the planning node
def planning_node(state: AgentState):
    """Planning node using Mistral"""
    system_prompt = SystemMessage(content=planning_prompt.format(
        tools=format_tools_description(tools)
    ))
    messages = [system_prompt] + state["messages"]
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    response = mistral.llm.invoke(messages_str)
    return {"messages": [AIMessage(content=response)]}

# Tool call node
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
    
    # Check if we have tool output in the messages
    tool_outputs = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    
    if tool_outputs:
        # Process tool outputs
        papers_info = []
        for tool_msg in tool_outputs:
            try:
                # Handle the tool output dynamically
                papers_info.append(tool_msg.content)
            except Exception as e:
                print(f"Error processing tool output: {e}")
                continue
        
        # Let the LLM process and summarize the results
        summary_prompt = f"""
        Process and summarize the following research results:
        {papers_info}
        """
        summary = agent_llm(summary_prompt)
        return {"messages": [AIMessage(content=summary)]}
    else:
        # Get the original query from the messages
        user_query = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")
        
        # Let the LLM decide how to handle the query
        response = agent_llm(messages_str)
        
        # Extract tool calls from the response
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
                        print(f"Error parsing tool arguments: {line}")
            
            if tool_name and tool_args:
                return {"messages": [AIMessage(
                    content=response,
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"{tool_name}-{time.time()}"
                    }]
                )]}
        
        # If no tool calls were found, return the response as is
        return {"messages": [AIMessage(content=response)]}


# Should continue function
def should_continue(state: AgentState):
    """Check if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # End execution if there are no tool calls
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Modify the judge node
def judge_node(state: AgentState):
    """Judge node using Mistral"""
    system_prompt = SystemMessage(content=judge_prompt)
    messages = [system_prompt] + state["messages"]
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    response: JudgeOutput = judge_llm(messages_str)
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": state.get("num_feedback_requests", 0) + 1
    }
    if response.feedback:
        output["messages"] = [AIMessage(content=response.feedback)]
    return output

# Final answer router function
def final_answer_router(state: AgentState):
    """Router to end the workflow or improve the answer."""
    if state["is_good_answer"]:
        return "end"
    else:
        return "planning"


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

async def test_agent():
    test_inputs = [
        "Download and summarize the findings of this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11379842/pdf/11671_2024_Article_4070.pdf",

        "Can you find 8 papers on quantum machine learning?",

        """Find recent papers (2023-2024) about CRISPR applications in treating genetic disorders, 
        focusing on clinical trials and safety protocols""",

        """Find and analyze papers from 2023-2024 about the application of transformer architectures in protein folding prediction, 
        specifically looking for novel architectural modifications with experimental validation."""
    ]

    # Run tests and store the results for later visualisation
    outputs = []
    for test_input in test_inputs:
        final_answer = await print_stream(app, test_input)
        outputs.append(final_answer.content)
    for input, output in zip(test_inputs, outputs):
        display(Markdown(f"## Input:\n\n{input}\n\n"))
        display(Markdown(f"## Output:\n\n{output}\n\n"))



async def run_agent(sample: str) -> list:
    outputs = []
    # Convert the input string to a HumanMessage
    input_message = HumanMessage(content=sample)
    final_answer = await call_stream(app, input_message)
    #outputs.append(final_answer.content)
    return final_answer.content

if __name__ == "__main__":
    #asyncio.run(test_agent())
    outputs = asyncio.run(run_agent("Search and analyze 5 recent papers about image steganography techniques from 2023-2024"))
    # print(f"THE OUTPUT TYPE IS: {type(outputs)}")
    # print(len(outputs))
    print(outputs)