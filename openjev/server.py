"""
Server module for OpenJev: FastAPI-based high-concurrency decision microservice.
"""
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from openjev.engine import OpenJevEngine
from openjev.types import FieldSpec, DecisionResult

app = FastAPI(
    title="OpenJev Decision Service",
    description="High-performance, non-autoregressive decision API with calibrated probabilities.",
    version="0.1.0"
)

engine = OpenJevEngine()


class DynamicDecisionRequest(BaseModel):
    context: str
    fields: List[FieldSpec]
    temperature: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok", "service": "openjev", "version": "0.1.0"}


@app.post("/decide", response_model=Dict[str, Any])
def decide_endpoint(req: DynamicDecisionRequest):
    try:
        # Build dynamic schema representation
        from openjev.types import ConfidenceScore
        import time, numpy as np, math
        
        start = time.perf_counter()
        values = {}
        scores = {}
        
        for field in req.fields:
            logits = engine.backend.score_candidates(
                context=req.context,
                question=field.description or field.name,
                candidates=field.options
            )
            probs = engine.default_scaler.scale_probabilities(logits.reshape(1, -1))[0]
            best_idx = int(np.argmax(probs))
            best_val = field.options[best_idx]
            conf = float(probs[best_idx])
            
            entropy = float(-np.sum([p * math.log(max(p, 1e-12)) for p in probs if p > 0]))
            prob_map = {opt: float(probs[i]) for i, opt in enumerate(field.options)}

            final_val: Any = best_val
            if field.is_boolean:
                final_val = (best_val.lower() == "true")

            values[field.name] = final_val
            scores[field.name] = {
                "selected_value": final_val,
                "confidence": round(conf, 4),
                "entropy": round(entropy, 4),
                "probabilities": prob_map,
                "is_confident": conf >= 0.5
            }

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "values": values,
            "confidence_scores": scores,
            "latency_ms": round(latency_ms, 2),
            "total_tokens_evaluated": len(req.context.split()),
            "is_fully_calibrated": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
