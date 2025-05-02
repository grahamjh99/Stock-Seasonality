from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv('.ipynb_checkpoints/aplhavantage_api_key-checkpoint.env')

# Get the API key
key = os.getenv('key')

# Now you can use the key variable
print(f"API Key loaded: {key[:4]}...")  # Only showing first 4 characters for security 