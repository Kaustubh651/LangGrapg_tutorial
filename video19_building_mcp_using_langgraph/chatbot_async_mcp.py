# backend.py
import sys
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
import requests
import asyncio

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    think=False
)


from pathlib import Path
import sys

server_path = Path(__file__).parent / "mcp-math-server" / "main.py"

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_path)],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://soft-pink-leopon.fastmcp.app/mcp",
        }
    }
)

# -------------------


# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def build_graph():


    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        response =await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    # -------------------
    # 5. Checkpointer
    # -------------------
    # conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
    # checkpointer = SqliteSaver(conn=conn)

    # -------------------
    # 6. Graph
    # -------------------
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")

    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge('tools', 'chat_node')

    chatbot = graph.compile()


    return chatbot


async def main():
    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Adding the expense of 1000 in travelling"
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "test-thread"}},
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
