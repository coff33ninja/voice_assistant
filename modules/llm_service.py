import asyncio
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.config import RunnableConfig  # Import RunnableConfig
from .config import LLM_MODEL_NAME
from typing import Dict, Any, Optional, cast # Import cast

llm_instance: Optional[Ollama] = None
# Simplified type for Pylance compatibility, though RunnableWithMessageHistory is generic.
runnable_with_history_global: Optional[RunnableWithMessageHistory] = None
message_history_store: dict[str, BaseChatMessageHistory] = {}

PROMPT_TEMPLATE_STR = """
You are a voice assistant that detects user intents and responds appropriately. You can set reminders, check calendars, provide weather updates, or answer questions. Use the conversation history for context.
Your primary role is to answer general questions and engage in conversation when the user's request isn't a specific command for actions like setting reminders, managing calendar events, or checking the weather. These specific actions are handled by other parts of the system.

Conversation history: {history}
User input: {input}

**Your Task:**
Based on the user's input and conversation history:

1.  **If the input is a general question or a conversational statement:**
    *   Provide a concise, helpful, and relevant answer or continuation of the conversation.
    *   Example: User: "What's the capital of France?" You: "The capital of France is Paris."
    *   Example: User: "Tell me a fun fact." You: "Certainly! Did you know honey never spoils?"

2.  **If the input seems like it *might* be a command for a specific action (e.g., reminder, calendar, weather) BUT it's unclear, ambiguous, or incomplete:**
    *   Do NOT attempt to guess or perform the action.
    *   Instead, ask for clarification or state that you need more information.
    *   Example: User: "Remind me about the thing." You: "Okay, I can set a reminder. What would you like to be reminded about, and when?"
    *   Example: User: "Weather." You: "Sure, I can get the weather for you. For which location?"
    *   Example: User: "Put meeting on my schedule." You: "I can help with that. What's the meeting about, and for what date and time?"

3.  **If the input is completely unintelligible or doesn't make sense:**
    *   Politely state that you didn't understand.
    *   Example: User: "Blah blah foobar." You: "I'm sorry, I didn't quite understand that. Could you please rephrase?"

**Important:**
*   You are NOT directly setting reminders, adding calendar events, or fetching weather. If the user's request is clear for these actions, another part of the system will handle it. Your role is to respond when the request is general or needs clarification *before* such an action can be taken by the system.
*   Do NOT try to guess or initiate an action (like setting a reminder) if the input is vague or doesn't clearly ask for it.
*   Use the conversation history to maintain context.
*   Keep your responses natural and conversational.

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

    # Ensure llm_instance is treated as Ollama for the chain construction
    if llm_instance is None:
        raise RuntimeError("LLM instance was not initialized before chain creation.") # Should not happen
    chain = prompt | cast(Ollama, llm_instance)  # Create the chain: prompt -> llm

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
        response = await asyncio.to_thread(
            runnable_with_history_global.invoke, {"input": input_text}, config=config
        )
        return str(response) # Ensure return type is str
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
        response = runnable_with_history_global.invoke({"input": input_text}, config=config)
        return str(response) # Ensure return type is str
    except Exception as e:
        print(f"[ERROR] LLM connection failed: {e}")
        return ""
