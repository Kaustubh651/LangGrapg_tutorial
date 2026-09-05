from email import message
from json import load

from langgraph.graph import StateGraph ,START,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,AIMessage,SystemMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen3.8-2.4T-A95B",
#     task="text-generation",
#     max_new_tokens=512,
# )

# model = ChatHuggingFace(llm=llm)
model = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState):
    messages = state['messages']
    response = model.invoke(messages)

    return {'messages':[response]}


# checkpointer
checkpointer = InMemorySaver()

## GRAPH
graph=  StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)


chatbot = graph.compile(checkpointer = checkpointer)

