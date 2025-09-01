from dotenv import load_dotenv
from pydantic_ai import Agent
import os
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
from pydantic_ai.models.google import GoogleProvider, GoogleModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup provider & model
provider = GoogleProvider(api_key=GOOGLE_API_KEY)
model = GoogleModel("gemini-2.5-flash", provider=provider)


agent = Agent(
    model,
    tools=[duckduckgo_search_tool()],
    system_prompt="Use DuckDuckGo search to find information. Always search before answering.",
)



async def web_search(query):
    answer=await  agent.run(query)
    return answer