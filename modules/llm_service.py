import asyncio
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.config import RunnableConfig  # Import RunnableConfig
from .config import LLM_MODEL_NAME

llm_instance = None
runnable_with_history_global = None
message_history_store: dict[str, BaseChatMessageHistory] = {}

PROMPT_TEMPLATE_STR = """
You are a voice assistant that detects user intents and responds appropriately. You can set reminders, check calendars, provide weather updates, or answer questions. Use the conversation history for context.

Conversation history: {history}
User input: {input}

If the user's input is clearly a request for a reminder, calendar action, or weather update, proceed with that.
If the user's input is a general question, answer it concisely and helpfully.
If the user's input is unclear, ambiguous, or doesn't seem to match any of your capabilities:
1. Ask for clarification.
2. Or, state that you don't understand.
Do NOT try to guess or initiate an action (like setting a reminder) if the input is vague or doesn't clearly ask for it.

Your response:
"""


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_history_store:
        message_history_store[session_id] = ChatMessageHistory()
    return message_history_store[session_id]


def initialize_llm():
    global llm_instance, runnable_with_history_global
    print("Initializing LLM service...")
    llm_instance = Ollama(model=LLM_MODEL_NAME)
    prompt = PromptTemplate(
        input_variables=["history", "input"], template=PROMPT_TEMPLATE_STR
    )

    chain = prompt | llm_instance  # Create the chain: prompt -> llm

    runnable_with_history_global = RunnableWithMessageHistory(
        chain,  # Wrap the chain with history
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    print("LLM service initialized.")


async def get_llm_response(input_text: str) -> str:
    if runnable_with_history_global is None:
        raise RuntimeError("LLM service not initialized. Call initialize_llm() first.")
    try:
        session_id = "user_session"  # Fixed session ID for single user assistant
        config: RunnableConfig = {
            "configurable": {"session_id": session_id}
        }  # Explicitly type the config
        response = await asyncio.to_thread(  # type: ignore
            runnable_with_history_global.invoke, {"input": input_text}, config=config
        )
        return response
    except Exception as e:
        print(f"[ERROR] LLM connection failed: {e}")
        return ""
def get_llm_response_sync(input_text: str) -> str:
    if runnable_with_history_global is None:
        raise RuntimeError("LLM service not initialized. Call initialize_llm() first.")
    try:
        session_id = "user_session"  # Fixed session ID for single user assistant
        config: RunnableConfig = {
            "configurable": {"session_id": session_id}
        }  # Explicitly type the config
        response = runnable_with_history_global.invoke({"input": input_text}, config=config)  # type: ignore
        return response
    except Exception as e:
        print(f"[ERROR] LLM connection failed: {e}")
        return ""
