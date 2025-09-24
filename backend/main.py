from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from backend.pipeline_full import DrugRAGPipeline


app = FastAPI(title="ddi-ksu: Drug–Drug Interaction RAG API")
pipeline: Optional[DrugRAGPipeline] = None


class BatchQueryRequest(BaseModel):
    queries: List[str]
    include_sources: bool = True
    response_format: str = "comprehensive"  # "simple" | "comprehensive" | "structured"
    k: int = 5


@app.on_event("startup")
def _startup():
    global pipeline
    pipeline = DrugRAGPipeline()


@app.get("/health")
def health() -> Dict[str, Any]:
    global pipeline
    status = pipeline.health_check() if pipeline else {"status": "uninitialized"}
    return status


@app.get("/query")
def query(q: str, include_sources: bool = True, response_format: str = "comprehensive", k: int = 5) -> Dict[str, Any]:
    global pipeline
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialized"}
    return pipeline.query(q, k=k, include_sources=include_sources, response_format=response_format)


@app.post("/batch_query")
def batch_query(body: BatchQueryRequest) -> Dict[str, Any]:
    global pipeline
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialized"}
    queries_limited = body.queries[:20]
    results = [
        pipeline.query(q, k=body.k, include_sources=body.include_sources, response_format=body.response_format)
        for q in queries_limited
    ]
    return {"results": results, "count": len(results)}


