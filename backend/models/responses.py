from pydantic import BaseModel
from typing import List, Optional


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source: str  # e.g., "vector_db", "web_search"



class GraphContext(BaseModel):
    entities: List[str]
    relationships: List[str]
    notes: Optional[str] = None
