from fastapi import FastAPI
from app.logging_config import logger, assign_request_id
from app.routers import search

app = FastAPI(title="CarMatch AI")

@app.middleware("http")
async def add_request_id_middleware(request, call_next):
    request.state.logger = assign_request_id()
    response = await call_next(request)
    return response

app.include_router(search.router, prefix="/search", tags=["Search"])