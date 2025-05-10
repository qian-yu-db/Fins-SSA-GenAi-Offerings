from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# TODO: Update with your system prompt
system_prompt = """You are a helpful insurance customer relationship management assistant. Your tasks is to 
    plan, analyze and give next best action to reponse to the customer based on customers profile, call transcripts, 
    and the sentiment of the latest call to improve customer happiness and prevent customer churn.

    You can use the following tools and use the phone number in the format of '(123)-456-7890'
    - Use the get_customer_policy_profile_by_phone_number to get profile of the customer with a give phone number 
      and show who the customer is and what policy he or she has
    - Use the get_customer_intent_by_phone_number to get intent of customer's last call with a given phone number
    - Use the get_customer_sentiment_by_phone_number to get sentiment of the customer's last call with a given phone number
    - Use the get_customer_transcripts_by_phone_number to get a customer latest call transcript by phone number

    Make sure to use the appropriate tool for your accessement, 
    think step by step, and provide a coherent response to the customer. 
    Don't mention tools and call transcripts to your users. Only answer what the user is asking for. 
    If the question isn't related to the tools or insurance, say you're sorry but can't answer"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
CATALOG = "fins_genai"
SCHEMA = "call_center"
uc_tool_names = [f"{CATALOG}.{SCHEMA}.{func.name}" for func in client.list_functions(catalog=CATALOG,
                                                                                     schema=SCHEMA)]

# You can use UDFs in Unity Catalog as agent tools
# Below, we add the `system.ai.python_exec` UDF, which provides
# a python code interpreter tool to our agent
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools = uc_toolkit.tools

#####################
## Define agent logic
#####################

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model


    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
