import asyncio
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from .config import LLM_MODEL_NAME

llm_instance = None
conversation_memory = None
llm_chain = None

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

def initialize_llm():
    global llm_instance, conversation_memory, llm_chain
    print("Initializing LLM service...")
    llm_instance = Ollama(model=LLM_MODEL_NAME)
    conversation_memory = ConversationBufferMemory(ai_prefix="Assistant:")
    prompt = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE_STR)
    llm_chain = ConversationChain(prompt=prompt, llm=llm_instance, memory=conversation_memory)
    print("LLM service initialized.")

async def get_llm_response(input_text: str) -> str:
    if llm_chain is None:
        raise RuntimeError("LLM service not initialized. Call initialize_llm() first.")
    response = await asyncio.to_thread(llm_chain.run, input=input_text)
    return response
