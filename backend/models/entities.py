from pydantic import BaseModel
from typing import List, Optional


class LegalEntity(BaseModel):
    id: str
    name: str
    entity_type: str  # e.g., "Case", "Statute", "Person"
    description: Optional[str] = None
    jurisdiction:str


class LegalRelationship(BaseModel):
    source_id: str
    target_id: str
    relation_type: str  # e.g., "cites", "overrules", "authored_by"
    confidence:float
