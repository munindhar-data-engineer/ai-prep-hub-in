from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(model='gpt-5.4-nano', temperature=0)
print(llm)


# load the knowledge base and create a retriever  tool
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserializaion=True
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4}) # here 4 is -> best matching count
retriever_tool = create_retriever_tool(
    retriever,
    name="search_vpn_knowledge_base",
    description=(
        "Search Acme Corp's internal VPN user guide document for installation, trouble shooting,"
        " and configuration details."
    )
)

# bind the tool to LLM

llm_with_tools = llm.bind_tools([retriever_tool])

SYSTEM_PROMPT = """You are an tech support assistant for a Pvt LTD. \
    Your only responsibility is to assist employees with their VPN issues.\
     Always respond clearly and politely when the user asks about VPN-issues \
     If the user's question is NOT about VPN issues please respond politely.\
     'Sorry, I can only assist with VPN related issues, please ask questions about your VPN issues.'\
     Do not provide any troubleshooting, suggestions or information."""


def chat_bot(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

# 4. connect everything into a Graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chat_bot)
graph_builder.add_node("tools", ToolNode(tools=[retriever_tool]))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditions_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)
agent = graph_builder.compile()



if __name__ == "__main__":
    print("Agent ready. Type 'quit' to exit.\n")
    conversation = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ('quit', 'exit'):
            break
        conversation.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": conversation})
        conversation = result["messages"]
        print("Agent:", conversation[-1].conent, "\n")