"""
LangChain + LangGraph integration with the local Gemma 4 server.
================================================================

Prerequisites:
  1. Start the Gemma 4 server:
       cd ~/gemma-server && source venv/bin/activate && python server.py

  2. Install LangChain deps (in another venv or the same one):
       pip install langchain langchain-openai langgraph

Usage:
  python langchain_example.py
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# 1. Basic LangChain connection
# ---------------------------------------------------------------------------
# Point ChatOpenAI at your local Gemma 4 server.
# No real API key needed — just pass a dummy string.

GEMMA_BASE_URL = "http://localhost:8000/v1"  # adjust if you changed the port

llm = ChatOpenAI(
    model="gemma-4-e2b-it",
    base_url=GEMMA_BASE_URL,
    api_key="not-needed",       # required param but not validated locally
    temperature=0.7,
    max_tokens=512,
)


def demo_basic_chat():
    """Simple one-shot chat."""
    print("=" * 60)
    print("DEMO 1: Basic Chat")
    print("=" * 60)
    response = llm.invoke("Explain quantum computing in 3 sentences.")
    print(response.content)
    print()


def demo_with_system_prompt():
    """Chat with a system prompt."""
    print("=" * 60)
    print("DEMO 2: System Prompt")
    print("=" * 60)
    messages = [
        SystemMessage(content="You are a helpful pirate assistant. Always answer in pirate speak."),
        HumanMessage(content="What is machine learning?"),
    ]
    response = llm.invoke(messages)
    print(response.content)
    print()


def demo_streaming():
    """Streaming response."""
    print("=" * 60)
    print("DEMO 3: Streaming")
    print("=" * 60)
    for chunk in llm.stream("Write a short poem about AI."):
        print(chunk.content, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# 2. LangChain Expression Language (LCEL) — chains & prompts
# ---------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def demo_lcel_chain():
    """LCEL chain with a prompt template."""
    print("=" * 60)
    print("DEMO 4: LCEL Chain")
    print("=" * 60)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {domain} teacher. Explain concepts clearly and concisely."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"domain": "physics", "question": "What is entropy?"})
    print(result)
    print()


# ---------------------------------------------------------------------------
# 3. LangGraph — stateful agent with tool use
# ---------------------------------------------------------------------------
def demo_langgraph_agent():
    """
    A minimal LangGraph ReAct-style agent that uses a simple tool.
    """
    print("=" * 60)
    print("DEMO 5: LangGraph Agent")
    print("=" * 60)

    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool
    except ImportError:
        print("⚠️  langgraph not installed. Install with: pip install langgraph")
        print("   Skipping LangGraph demo.\n")
        return

    # Define a simple tool
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression. Input should be a valid Python math expression."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the agent
    agent = create_react_agent(llm, tools=[calculator, get_current_time])

    # Run it
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is 42 * 17 + 123?")]}
    )

    # Print the final response
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")
    print()


# ---------------------------------------------------------------------------
# 4. Multi-turn conversation
# ---------------------------------------------------------------------------
def demo_multi_turn():
    """Simulating a multi-turn conversation."""
    print("=" * 60)
    print("DEMO 6: Multi-turn Conversation")
    print("=" * 60)

    from langchain_core.messages import AIMessage

    history = [
        SystemMessage(content="You are a helpful coding assistant."),
    ]

    questions = [
        "What is a Python decorator?",
        "Can you show me a simple example?",
        "How would I use it for timing a function?",
    ]

    for q in questions:
        print(f"👤 User: {q}")
        history.append(HumanMessage(content=q))
        response = llm.invoke(history)
        print(f"🤖 Gemma: {response.content[:300]}...")
        history.append(AIMessage(content=response.content))
        print()


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Gemma 4 + LangChain/LangGraph Integration Demos\n")
    print(f"Connecting to: {GEMMA_BASE_URL}\n")

    demo_basic_chat()
    demo_with_system_prompt()
    demo_streaming()
    demo_lcel_chain()
    demo_langgraph_agent()
    demo_multi_turn()

    print("✅ All demos complete!")
