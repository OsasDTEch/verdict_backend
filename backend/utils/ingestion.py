import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ..scrapper.scrape import scrape_url

load_dotenv()

# ---------------- Environment & Clients ----------------
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URL")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize clients globally
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
qdrant_client = QdrantClient(
    url="https://dc2f8be5-710e-49ab-a78b-f72c10c8dacb.eu-west-2-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.d2pifJ83uRTgShXzOMh2jJLoPiDPXTis5bylcZkqiiA",
)
embeddings_client = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Collection name
COLLECTION_NAME = "legal_knowledge_base"


# ---------------- Data Models ----------------
class LegalEntity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # statute, case, concept, person, organization
    jurisdiction: Optional[str] = None
    description: Optional[str] = None


class LegalRelationship(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str  # GOVERNS, CITES, SUPERSEDES, DEFINES
    confidence: float = 1.0


class GraphComponents(BaseModel):
    entities: List[LegalEntity]
    relationships: List[LegalRelationship]


class SearchResult(BaseModel):
    content: str
    score: float
    entity_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------- Setup Functions ----------------
def setup_qdrant_collection():
    """Ensure Qdrant collection exists"""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists")
    except Exception as e:
        if 'Not found' in str(e):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            print(f"Created collection '{COLLECTION_NAME}'")
        else:
            raise e


def setup_neo4j_constraints():
    """Create Neo4j constraints and indexes"""
    with neo4j_driver.session() as session:
        constraints = [
            "CREATE CONSTRAINT legal_entity_id IF NOT EXISTS FOR (e:LegalEntity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX legal_entity_name IF NOT EXISTS FOR (e:LegalEntity) ON (e.name)",
            "CREATE INDEX legal_entity_type IF NOT EXISTS FOR (e:LegalEntity) ON (e.type)",
        ]

        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception:
                pass  # Constraint might already exist


# ---------------- Knowledge Extraction ----------------
async def extract_legal_knowledge(text: str) -> GraphComponents:
    """Extract legal entities and relationships from text using LLM"""

    provider = GoogleProvider(api_key=GOOGLE_API_KEY)
    model = GoogleModel('gemini-2.5-flash', provider=provider)

    extraction_prompt = """
    You are a legal knowledge extraction expert. Analyze the legal text and extract:
    1. Legal entities (statutes, cases, concepts, organizations, people)
    2. Relationships between entities (governs, cites, defines, etc.)

    Format as JSON with this structure:
    {
        "entities": [
            {
                "name": "entity name",
                "type": "statute|case|concept|organization|person",
                "jurisdiction": "jurisdiction if applicable",
                "description": "brief description"
            }
        ],
        "relationships": [
            {
                "source_id": "source entity name",
                "target_id": "target entity name", 
                "relationship_type": "GOVERNS|CITES|DEFINES|SUPERSEDES",
                "confidence": 0.9
            }
        ]
    }
    """

    extraction_agent = Agent(
        model,
        system_prompt=extraction_prompt,
        output_type=GraphComponents
    )

    prompt = f"Extract legal knowledge from this text: {text}"
    result = await extraction_agent.run(prompt)
    return result.output


# ---------------- Knowledge Graph Functions ----------------
def add_entities_to_neo4j(entities: List[LegalEntity]) -> Dict[str, str]:
    """Add legal entities to Neo4j knowledge graph"""
    entity_id_map = {}

    with neo4j_driver.session() as session:
        for entity in entities:
            session.run("""
                MERGE (e:LegalEntity {name: $name})
                SET e.id = $id,
                    e.type = $type,
                    e.jurisdiction = $jurisdiction,
                    e.description = $description,
                    e.updated_at = datetime()
            """,
                        id=entity.id,
                        name=entity.name,
                        type=entity.type,
                        jurisdiction=entity.jurisdiction,
                        description=entity.description
                        )
            entity_id_map[entity.name] = entity.id

    return entity_id_map


def add_relationships_to_neo4j(relationships: List[LegalRelationship], entity_map: Dict[str, str]):
    """Add relationships between entities to Neo4j"""

    with neo4j_driver.session() as session:
        for rel in relationships:
            # Map names to IDs
            source_id = entity_map.get(rel.source_id, rel.source_id)
            target_id = entity_map.get(rel.target_id, rel.target_id)

            if source_id and target_id:
                session.run("""
                    MATCH (a:LegalEntity {id: $source_id})
                    MATCH (b:LegalEntity {id: $target_id})
                    MERGE (a)-[r:LEGAL_RELATION {type: $rel_type}]->(b)
                    SET r.confidence = $confidence,
                        r.created_at = datetime()
                """,
                            source_id=source_id,
                            target_id=target_id,
                            rel_type=rel.relationship_type,
                            confidence=rel.confidence
                            )


def search_neo4j_entities(entity_name: str = None, entity_type: str = None, jurisdiction: str = None) -> Dict[str, Any]:
    """Search Neo4j for entities and their relationships"""

    # Build query conditions
    conditions = []
    params = {}

    if entity_name:
        conditions.append("e.name CONTAINS $entity_name")
        params["entity_name"] = entity_name

    if entity_type:
        conditions.append("e.type = $entity_type")
        params["entity_type"] = entity_type

    if jurisdiction:
        conditions.append("e.jurisdiction = $jurisdiction")
        params["jurisdiction"] = jurisdiction

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
    MATCH (e:LegalEntity)
    WHERE {where_clause}
    OPTIONAL MATCH (e)-[r:LEGAL_RELATION]-(related:LegalEntity)
    RETURN e, r, related
    LIMIT 20
    """

    nodes = set()
    edges = []
    entities = []
    relationships = []

    with neo4j_driver.session() as session:
        result = session.run(query, **params)

        for record in result:
            entity = record["e"]
            relation = record["r"]
            related = record["related"]

            # Add entity to results
            entities.append(dict(entity))
            nodes.add(entity["name"])

            # Add relationship if exists
            if relation and related:
                nodes.add(related["name"])
                edges.append(f"{entity['name']} -[{relation['type']}]-> {related['name']}")
                relationships.append({
                    "source": entity["name"],
                    "target": related["name"],
                    "type": relation["type"],
                    "confidence": relation.get("confidence", 1.0)
                })

    return {
        "nodes": list(nodes),
        "edges": edges,
        "entities": entities,
        "relationships": relationships
    }


# ---------------- Vector Database Functions ----------------
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks for vector storage"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def add_text_to_qdrant(text: str, entity_ids: List[str], metadata: Dict[str, Any] = None) -> str:
    """Add text chunks to Qdrant vector database"""
    if metadata is None:
        metadata = {}

    chunks = chunk_text(text)
    point_ids = []

    for chunk in chunks:
        # Generate embedding
        embedding = embeddings_client.embed_query(chunk)

        # Create unique point ID
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)

        # Prepare payload
        payload = {
            "text": chunk,
            "entity_ids": entity_ids,
            "metadata": metadata,
            "timestamp": str(asyncio.get_event_loop().time())
        }

        # Add to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{
                "id": point_id,
                "vector": embedding,
                "payload": payload
            }]
        )

    return f"Added {len(chunks)} text chunks to vector database"


def search_qdrant_vectors(query: str, limit: int = 5, jurisdiction: str = None) -> List[SearchResult]:
    """Search Qdrant for semantically similar content"""

    # Generate query embedding
    query_embedding = embeddings_client.embed_query(query)

    # Build filter for jurisdiction if specified
    query_filter = None
    if jurisdiction:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.jurisdiction",
                    match=models.MatchValue(value=jurisdiction)
                )
            ]
        )

    # Search Qdrant
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )

    # Format results
    results = []
    for hit in search_result:
        result = SearchResult(
            content=hit.payload["text"],
            score=hit.score,
            entity_id=hit.payload.get("entity_ids", [None])[0],
            metadata=hit.payload.get("metadata", {})
        )
        results.append(result)

    return results


# ---------------- Combined Functions ----------------
async def add_legal_document(text: str, jurisdiction: str = None) -> str:
    """Extract knowledge from text and add to both Neo4j and Qdrant"""

    try:
        # Extract knowledge using LLM
        knowledge = await extract_legal_knowledge(text)

        # Add entities to Neo4j
        entity_map = add_entities_to_neo4j(knowledge.entities)

        # Add relationships to Neo4j
        add_relationships_to_neo4j(knowledge.relationships, entity_map)

        # Prepare metadata for vector storage
        metadata = {"jurisdiction": jurisdiction} if jurisdiction else {}

        # Add text to Qdrant
        vector_result = add_text_to_qdrant(
            text=text,
            entity_ids=list(entity_map.values()),
            metadata=metadata
        )

        return f"""Successfully processed legal document:
- Extracted {len(knowledge.entities)} entities: {[e.name for e in knowledge.entities][:3]}...
- Created {len(knowledge.relationships)} relationships
- {vector_result}
"""

    except Exception as e:
        return f"Error processing legal document: {str(e)}"


def hybrid_search(query: str, jurisdiction: str = None) -> Dict[str, Any]:
    """Search both Neo4j and Qdrant for comprehensive results"""

    # Search vector database
    vector_results = search_qdrant_vectors(query, jurisdiction=jurisdiction)

    # Search knowledge graph (try to find entities mentioned in query)
    query_words = query.split()
    graph_results = None

    for word in query_words:
        if len(word) > 3:  # Skip short words
            graph_results = search_neo4j_entities(
                entity_name=word,
                jurisdiction=jurisdiction
            )
            if graph_results["nodes"]:  # Found something
                break

    return {
        "vector_results": vector_results,
        "graph_context": graph_results,
        "query": query,
        "jurisdiction": jurisdiction
    }


async def main():
    setup_neo4j_constraints()
    setup_qdrant_collection()

    docs = scrape_url()
    text = " ".join([doc.page_content for doc in docs])  # join all docs


    print(docs)
    s= await add_legal_document(text)
    print(s)

async def question():
    setup_neo4j_constraints()
    setup_qdrant_collection()
    s= search_qdrant_vectors('What are the employment condition in the us.')
    t= search_neo4j_entities('Which sections of the UK Data Protection Act cite GDPR Article 6?')
    v= hybrid_search('what is the employment condition in us , and how does it relate to the data protection law')
    print(f' qdrant: {s}')
    print(f'noe4j: {t}')
    print(f'combine: {v}')


if __name__== "__main__":
    asyncio.run(question())

