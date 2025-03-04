import json
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from .models import DecisionMakingOutput, JudgeOutput

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

# Initialize LLMs
decision_making_llm = mistral.structured_output(DecisionMakingOutput)
agent_llm = mistral.with_tools([])  # Tools will be set in workflow.py
judge_llm = mistral.structured_output(JudgeOutput)
