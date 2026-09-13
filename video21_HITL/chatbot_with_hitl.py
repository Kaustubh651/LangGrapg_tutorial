from multiprocessing.reduction import steal_handle
from typing import TypedDict,Annotated
from xml.dom import Node

from langchain_core.messages import HumanMessage,AnyMessage,AIMessage,BaseMessage

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ------------Load LLM ------------

from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    think=False
)

# ----------------------------
# 2 tool
# ----------------------------

@tool
def get_the_stock_price(symbol:str)->dict :

    """
    Fetch the lastest stock price for a given symbol (e.g 'AAPL' ,'TSLA')
    using the aplha vantage with API key in the url.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"

    r = requests.get(url)

    return r.json()

@tool
def purchase_stock(symbol:str, quantity:int)->dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """

    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    if isinstance(decision,str) and decision.lower() == "yes":
        return{
            "status":"sucess",
            "message":f"Purchase order places for {quantity} shares of {symbol}.",
            "symbol":symbol,
            "quantity":quantity
        }
    
    else:
        return{
            "status":"Cancelled",
            "message":f"Purchase order places for {quantity} shares of {symbol} was declined by human.",
            "symbol":symbol,
            "quantity":quantity
        } 

tools = [purchase_stock, get_the_stock_price]
llm_with_tool = llm.bind_tools(tools)

# ----------- 
# 3 State defination
# ----------------

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# -------------------
# 4 Node
# ----------------------

def chat_node(state:ChatState):
    """LLM node that may answer or request a tool call"""
    message = state['messages']
    response = llm_with_tool.invoke(message)

    return {'messages':[response]}

tool_node = ToolNode(tools)

# ----------------------------
# #5 Checkpointer
# ----------------------------

checkpointer = InMemorySaver()

# ----------------------------
# 6 graph
# ----------------------------

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


chatbot = graph.compile(checkpointer=checkpointer)

# ----------------------------
# 7 Simple usage example (CLI with HITL)
# ----------------------------

if __name__ =='__main__':
    thread_id = 'demo_thread'

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {'exit','quit','bye'}:
            print("GoodBye!")
            break

        state = {'messages':[HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config = {"configurable":{"thread_id":thread_id}}
        )

        interrupts = result.get("__interrupt__",[])

        if interrupts:

            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            result = chatbot.invoke(
                Command(resume=decision),
                config = {"configurable":{"thread_id":thread_id}}
            )

        messages = result['messages']
        last_msg= messages[-1].content
        print(f"Bot: {last_msg}\n ")