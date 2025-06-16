# greeting_module.py
# Contains greeting and goodbye message variations and intent logic

import random

greeting_variations = [
    "Hello! How can I help you?",
    "Hi there! What can I do for you today?",
    "Greetings! How may I assist you?",
    "Hey! How can I be of service?",
    "Good day! What would you like to do?"
]

goodbye_variations = [
    "Goodbye! Have a great day!",
    "See you later!",
    "Take care!",
    "Bye! If you need anything, just call me again.",
    "Farewell!"
]

def get_greeting():
    return random.choice(greeting_variations)

def get_goodbye():
    return random.choice(goodbye_variations)
