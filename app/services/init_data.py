from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI, embeddings
from polars import read_csv
import os
from app.logging_config import logger, step

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test_api_key",
)
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "car_embeddings"
embed_model = "text-embedding-ada-002"


def emdedings_from_openai(texts):
    with step("Generating embeddings..."):
        response = client.embeddings.create(
            model=embed_model,
            input=texts,
            encoding_format="float",
        )
        embeddings = [item.embedding for item in response.data]
    return embeddings


def init_qdrant_collection(collection_name=COLLECTION_NAME):
    existing_collections = qdrant_client.get_collections().collections
    logger.info(
        f"Existing Qdrant collections: {[col.name for col in existing_collections]}"
    )
    if collection_name in [col.name for col in existing_collections]:
        logger.info(f"Deleting existing collection: {collection_name}")
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
    )


def load_initial_data():
    data_file = os.path.join(os.path.dirname(__file__), "Cars_Datasets_2025.csv")

    init_qdrant_collection()
    # Company Names,Cars Names,Engines,CC/Battery Capacity,HorsePower,Total Speed,Performance(0 - 100 )KM/H,Cars Prices,Fuel Types,Seats,Torque
    df = read_csv(data_file, encoding="utf8-lossy")
    car_data = df.to_dicts()

    texts = [
        f"{car['Company Names']} {car['Cars Names']} with {car['Engines']} engine, {car['HorsePower']} HP, {car['Cars Prices']} price."
        for car in car_data
    ]
    embeddings = emdedings_from_openai(texts)
    points = []
    for i, car in enumerate(car_data):
        payload = {
            "company": car["Company Names"],
            "car": car["Cars Names"],
            "engine": car["Engines"],
            "hp": car["HorsePower"],
            "price": car["Cars Prices"],
        }

        points.append(
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload=payload,
            )
        )

    for i in range(0, len(points), 100):
        logger.info(f"Upserting points {i} to {min(i+100, len(points))}...")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + 100],
        )


if __name__ == "__main__":
    with step("Loading initial data into Qdrant"):
        load_initial_data()
