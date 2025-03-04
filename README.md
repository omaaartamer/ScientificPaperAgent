# Scientific Research Assistant Agent

A LangGraph-based research assistant that helps users find, analyze, and summarize scientific papers using the CORE API.

## Overview

This project implements a research assistant agent that can:
- Search for scientific papers
- Download and analyze PDFs
- Provide structured summaries
- Answer research-related questions

## Challenges and Approach

### 1. LLM Integration Challenge

**Original Implementation:**
- The original paper used OpenAI's GPT models
- Required API keys and had usage costs
- Had specific message formatting requirements

**Solution:**
- Migrated to locally-hosted Mistral using Ollama
- Created a custom wrapper for Mistral to handle message formatting
- Implemented structured output parsing for consistent responses
- Eliminated API costs and privacy concerns

### 2. Architecture Challenges

**Workflow Management:**
- Complex state management between different agent nodes
- Need for consistent message formatting
- Tool integration and execution flow

**Solution:**
- Used LangGraph for structured workflow
- Implemented state management through AgentState
- Created modular tools system for paper search and download
- Built clear node transitions and decision making

### 3. Frontend Integration

**Challenges:**
- Real-time communication with the agent
- Handling async operations
- Displaying structured research outputs

**Solution:**
- Created a Flask-based API endpoint
- Implemented async processing with proper error handling
- Built a responsive frontend for real-time interaction
- Structured output formatting for better readability

## Technical Stack

- **Backend Framework:** Flask
- **LLM:** Mistral (via Ollama)
- **Workflow Management:** LangGraph
- **API Integration:** CORE API
- **PDF Processing:** pdfplumber
- **Frontend:** HTML, CSS, JavaScript

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Install Ollama and Mistral:
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh
# Pull Mistral model
ollama pull mistral
```
3. Set up environment variables:
```bash
# Create .env file
CORE_API_KEY=your_key_here
```
Run the application:
```bash
python run.py
```
```bash
Project Structure
.
├── run.py                 # Flask app entry point
├── requirements.txt       # Project dependencies
├── flaskApp/             # Flask application
│   ├── __init__.py
│   ├── config.py
│   ├── views.py
│   ├── static/
│   └── templates/
└── agent/                  # Agent implementation
    ├── __init__.py
    ├── models.py         # Pydantic models
    ├── core_wrapper.py   # CORE API wrapper
    ├── mistral_wrapper.py # LLM wrapper
    ├── prompts.py        # System prompts
    ├── tools.py          # Agent tools
    ├── utils.py          # Helper functions
    └── workflow.py       # LangGraph workflow
```

## Contributing
- Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License
- This project is licensed under the Apache License 2.0 - see the LICENSE file for details.


## Acknowledgments

- This implementation is based on the scientific research agent from [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents), specifically their implementation of the Scientific Papers Researcher
- Modified the original implementation to:
  - Replaced OpenAI with locally-hosted Mistral via Ollama
  - Added Flask web interface
  - Restructured code for modularity
- Uses the CORE API for academic paper access
- Built with LangGraph and Mistral

### Original Paper and Implementation
The original implementation and research was done by:
- Repository: [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)
- Authors: Nir Diamant and contributors
- Implementation: Scientific Papers Researcher Jupyter notebook

This project maintains the core functionality and workflow of the original implementation while adapting it for web deployment and local LLM usage.

## Known Limitations and Future Work

### Local Model Performance

While the integration of tools and API is functional, the locally hosted Mistral model shows some limitations:

1. **Hallucinations**:
   - Generated paper titles and authors that don't exist
   - Created sequential DOI numbers
   - Produced plausible but fake content
   - Inconsistent date ranges in results

2. **Response Consistency**:
   - Debug messages appearing in responses
   - Repetitive content in single responses
   - Internal JSON formatting visible to users
   - Tool execution flow issues

3. **Comparison with OpenAI Models**:
   - Less accurate paper summaries
   - Lower quality of research analysis
   - More prone to fabricating details
   - Less reliable tool usage

### Future Improvements

1. **Model Enhancement**:
   - Experiment with different local models
   - Fine-tune for research paper analysis
   - Optimize for tool usage
   - Improve response formatting

2. **Integration Refinement**:
   - Better state management
   - Cleaner response handling
   - Improved error handling
   - Debug output filtering

3. **Alternative Approaches**:
   - Explore hybrid solutions (local + cloud)
   - Simplify modular architecture
   - Enhance validation checks
   - Add fact-checking mechanisms

Note: For production use cases requiring high accuracy and reliability, consider using OpenAI's models or other cloud-based solutions.