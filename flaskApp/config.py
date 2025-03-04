import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    CORE_API_KEY = os.getenv('CORE_API_KEY')
    