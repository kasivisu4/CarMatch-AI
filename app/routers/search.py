from fastapi import APIRouter, Request, HTTPException, status
import json
from app.logging_config import step
from app.services.search_engine import CarSearchEngine
from pydantic import BaseModel

router = APIRouter()
engine = CarSearchEngine()


class CarQuery(BaseModel):
    query: str


@router.post("/search")
async def search_cars(request: Request, query: CarQuery):
    """Handle a search POST request expecting JSON body like {"query": "..."}.

    Returns 415 if Content-Type is not application/json and 400 for invalid/empty JSON.
    """

    try:
        data = await request.json()
        query = data.get("query", "")
        logger = request.state.logger

        with step("Recieved query"):
            logger.info(f"Query received: {query}")

        with step("Generating embeddings"):
            embedding = engine.generate_embedding(query)

        with step("Searching for similar cars"):
            results = engine.search_similar_cars(embedding)

        with step("Preparing response"):
            response = {"results": results}
            logger.info(f"Search completed with {len(results)} results.")
            return response
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body."
        )
