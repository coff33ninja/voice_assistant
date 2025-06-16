import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

os.getenv("YOUR_ENV_VAR")
