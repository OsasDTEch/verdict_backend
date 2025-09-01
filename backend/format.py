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
COLLECTION_NAME = "verdictai_knowledge_base"


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
    model = GoogleModel('gemini-1.5-flash', provider=provider)

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


# ---------------- Pydantic AI Tools ----------------
def create_legal_agent():
    """Create Pydantic AI agent with legal research tools"""

    provider = GoogleProvider(api_key=GOOGLE_API_KEY)
    model = GoogleModel('gemini-1.5-flash', provider=provider)

    # Tool 1: Search Knowledge Graph
    async def search_knowledge_graph(ctx: RunContext[None],
                                     entity_name: str = None,
                                     entity_type: str = None,
                                     jurisdiction: str = None) -> str:
        """Search the legal knowledge graph for entities and relationships."""

        results = search_neo4j_entities(
            entity_name=entity_name,
            entity_type=entity_type,
            jurisdiction=jurisdiction
        )

        if not results["nodes"]:
            return "No relevant entities found in knowledge graph."

        response = f"Found {len(results['nodes'])} entities:\n"
        response += f"Entities: {', '.join(results['nodes'][:10])}\n"

        if results["edges"]:
            response += f"\nKey relationships:\n"
            for edge in results["edges"][:5]:
                response += f"- {edge}\n"

        return response

    # Tool 2: Search Vector Database
    async def search_vector_database(ctx: RunContext[None],
                                     query: str,
                                     jurisdiction: str = None,
                                     limit: int = 3) -> str:
        """Search the vector database for semantically similar legal content."""

        results = search_qdrant_vectors(
            query=query,
            limit=limit,
            jurisdiction=jurisdiction
        )

        if not results:
            return "No relevant content found in vector database."

        response = f"Found {len(results)} relevant documents:\n\n"

        for i, result in enumerate(results, 1):
            response += f"{i}. (Score: {result.score:.3f})\n"
            response += f"{result.content[:300]}...\n"
            if result.metadata:
                response += f"Metadata: {result.metadata}\n"
            response += "\n"

        return response

    # Tool 3: Add Legal Content
    async def add_legal_content(ctx: RunContext[None],
                                text: str,
                                jurisdiction: str = None) -> str:
        """Extract legal knowledge from text and add to both knowledge graph and vector database."""

        return await add_legal_document(text, jurisdiction)

    # Tool 4: Hybrid Search
    async def hybrid_legal_search(ctx: RunContext[None],
                                  query: str,
                                  jurisdiction: str = None) -> str:
        """Perform comprehensive search using both vector similarity and knowledge graph relationships."""

        results = hybrid_search(query, jurisdiction)

        response = f"Hybrid search results for: '{query}'\n\n"

        # Vector results
        if results["vector_results"]:
            response += "SIMILAR CONTENT:\n"
            for i, result in enumerate(results["vector_results"][:3], 1):
                response += f"{i}. {result.content[:200]}... (Score: {result.score:.3f})\n"

        # Graph results
        if results["graph_context"] and results["graph_context"]["nodes"]:
            response += f"\nRELATED ENTITIES: {', '.join(results['graph_context']['nodes'][:5])}\n"
            if results["graph_context"]["edges"]:
                response += "KEY RELATIONSHIPS:\n"
                for edge in results["graph_context"]["edges"][:3]:
                    response += f"- {edge}\n"

        return response

    # Create the agent
    agent = Agent(
        model,
        system_prompt="""You are a legal research assistant with access to a comprehensive legal knowledge base.

You have these tools available:
1. search_knowledge_graph - Find legal entities and their relationships
2. search_vector_database - Find semantically similar legal content  
3. add_legal_content - Add new legal information to the knowledge base
4. hybrid_legal_search - Comprehensive search using both methods

Use these tools to provide accurate, well-researched legal information. Always cite your sources and indicate the jurisdiction when relevant.""",
        tools=[
            search_knowledge_graph,
            search_vector_database,
            add_legal_content,
            hybrid_legal_search
        ]
    )

    return agent


# ---------------- Main Function ----------------
async def main():
    # Setup databases
    setup_qdrant_collection()
    setup_neo4j_constraints()

    # Create agent
    legal_agent = create_legal_agent()

    # Example usage
    sample_legal_text = """
    The Fair Labor Standards Act (FLSA) establishes minimum wage and overtime pay requirements.
    Under FLSA, employees must receive overtime pay at one and one-half times their regular rate 
    for hours worked over 40 in a workweek. The Department of Labor enforces FLSA compliance.
    FLSA applies to enterprises with annual gross sales of $500,000 or more.
    """

    print("=== Adding Legal Content ===")
    add_result = await legal_agent.run(
        f"Add this US employment law content: {sample_legal_text}"
    )
    print(add_result.output)

    print("\n=== Knowledge Graph Search ===")
    kg_result = await legal_agent.run(
        "Search the knowledge graph for Fair Labor Standards Act"
    )
    print(kg_result.output)

    print("\n=== Vector Search ===")
    vector_result = await legal_agent.run(
        "Search for content about overtime pay requirements"
    )
    print(vector_result.output)

    print("\n=== Hybrid Search ===")
    hybrid_result = await legal_agent.run(
        "What are the overtime requirements under US employment law?"
    )
    print(hybrid_result.output)


if __name__ == "__main__":
    asyncio.run(main())