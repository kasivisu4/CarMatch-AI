from openai import OpenAI
from qdrant_client import QdrantClient
from app.logging_config import logger

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test_api_key",
)

qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "car_embeddings"
embed_model = "text-embedding-ada-002"


class CarSearchEngine:
    def generate_embedding(self, query: str):
        embeddings = client.embeddings.create(
            model=embed_model,
            input=query,
            encoding_format="float",
        )
        return embeddings

    def search_similar_cars(self, embedding):
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding.data[0].embedding,
            with_vectors=False,
            limit=5,
        )
        logger.info(f"Search returned results., {search_result}")
        results = [
            {
                "company": point.payload.get("company"),
                "car": point.payload.get("car"),
                "engine": point.payload.get("engine"),
                "hp": point.payload.get("hp"),
                "price": point.payload.get("price"),
                "score": point.score,
            }
            for point in search_result.points
        ]
        return results
