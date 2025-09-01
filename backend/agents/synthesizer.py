import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os# Force dotenv to load from project root
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from pydantic_ai.models.google import GoogleModel, GoogleProvider

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
provider = GoogleProvider(api_key=GOOGLE_API_KEY)
model = GoogleModel('gemini-2.5-flash', provider=provider)
from pydantic import BaseModel, Field
from typing import List, Optional
#
class Source(BaseModel):
    type: str = Field(..., description="Type of source, e.g. 'web', 'vector_db', 'knowledge_graph'")
    reference: str = Field(..., description="Link, document id, or node id of the source")
    snippet: Optional[str] = Field(None, description="Small snippet or summary from the source")

class SynthesisOutput(BaseModel):
    # query: str = Field(..., description="The original query the agent was answering")
    summary: str = Field(..., description="High-level synthesized answer to the query")
    # key_points: List[str] = Field(..., description="Bullet-point breakdown of the most important facts")
    # reasoning: str = Field(..., description="How the agent combined different sources to form the answer")
    # sources: List[Source] = Field(..., description="List of sources used in the synthesis")

SYNTH_PROMPT = """
You are a Helpful Legal Research Synthesizer AI. 
You receive:
1. A legal research question
2. Draft answer(s) generated from retrieved sources
3. Raw retrieved context (vector DB chunks, knowledge graph facts, or web search results)

Your job:
- Verify consistency between the draft answer and retrieved context
- Eliminate redundancy and contradictions
- Organize the answer in a clear, professional legal memo style:
  - **Issue** (restate the legal question clearly)
  - **Rule** (summarize key laws, principles, or precedents)
  - **Analysis** (apply the rules to the issue, with reasoning)
  - **Conclusion** (short, decisive statement)
- Include pinpoint citations where possible (e.g., "Art. 6 GDPR", "UK DPA 2018, s.35").

Output: A polished, accurate, and concise legal analysis that could be presented to a lawyer or compliance officer.
"""


final_agent= Agent(model,
                   system_prompt=SYNTH_PROMPT,
                   output_type=SynthesisOutput
                   )

async def test():
    text="""
    The General Data Protection Regulation (GDPR) establishes several key principles that govern the processing of personal data, as outlined in Article 5 GDPR. These principle
s ensure that personal data is processed lawfully, fairly, and transparently, while also protecting the rights and freedoms of data subjects. The core principles are:\n\n1.  **Lawf
ulness, Fairness, and Transparency (Art. 5(1)(a) GDPR)**:\n    *   **Lawfulness**: Personal data must be processed only if there is a valid legal basis, such as the data subject's 
consent, necessity for a contract, compliance with a legal obligation, protection of vital interests, performance of a task carried out in the public interest, or legitimate intere
sts of the controller (Art. 6 GDPR). Stricter conditions apply to special categories of personal data (Art. 9 GDPR).\n    *   **Fairness**: Processing must be in line with the reas
onable expectations of the data subjects.\n    *   **Transparency**: Data subjects must be informed about the processing of their personal data in a concise, transparent, intelligi
ble, and easily accessible form, using clear and plain language.\n\n2.  **Purpose Limitation (Art. 5(1)(b) GDPR)**:\n    *   Personal data must be collected for specified, explicit
, and legitimate purposes and not further processed in a manner that is incompatible with those purposes.\n\n3.  **Data Minimisation (Art. 5(1)(c) GDPR)**:\n    *   Personal data p
rocessed must be adequate, relevant, and limited to what is necessary in relation to the purposes for which they are processed.\n\n4.  **Accuracy (Art. 5(1)(d) GDPR)**:\n    *   Pe
rsonal data must be accurate and, where necessary, kept up to date. Every reasonable step must be taken to ensure that personal data that are inaccurate, having regard to the purpo
ses for which they are processed, are erased or rectified without delay.\n\n5.  **Storage Limitation (Art. 5(1)(e) GDPR)**:\n    *   Personal data must be kept in a form that permi
ts identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed.\n\n6.  **Integrity and Confidentiality (Security) (Ar
t. 5(1)(f) GDPR)**:\n    *   Personal data must be processed in a manner that ensures appropriate security of the personal data, including protection against unauthorised or unlawf
ul processing and against accidental loss, destruction, or damage, using appropriate technical or organisational measures.\n\n7.  **Accountability (Art. 5(2) GDPR)**:\n    *   The 
data controller is responsible for, and must be able to demonstrate compliance with, the principles outlined in Article 5(1) GDPR. This often requires maintaining records of proces
sing activities and implementing appropriate data protection policies.\n\nThese principles form the foundation of GDPR compliance and guide organisations in their handling of personal data within the European Union and for data subjects located there
    """
    s= await final_agent.run(text)
    print(s)
    return s

if __name__=='__main__':
    asyncio.run(test())
