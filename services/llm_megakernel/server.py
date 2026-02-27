"""Streaming LLM server using FastAPI and Server-Sent Events."""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import os

from wrapper import MegakernelDecoder

app = FastAPI(title="Qwen Megakernel LLM Service")

# Global decoder instance
decoder = MegakernelDecoder()


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: int = 100
    weights_path: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Optionally preload weights on startup."""
    if os.getenv("LLM_PRELOAD_WEIGHTS", "0") == "1":
        decoder.load_weights(os.getenv("QWEN_MEGAKERNEL_MODEL_NAME"))


@app.post("/generate")
async def generate_stream(request: GenerateRequest):
    """
    Stream tokens as they are generated.
    
    Returns Server-Sent Events (SSE) stream of tokens.
    """
    # If weights_path is provided and different from current, reload
    if request.weights_path:
        decoder.load_weights(request.weights_path)
    
    def token_generator():
        """Generator that yields SSE-formatted token events."""
        try:
            for token in decoder.generate_stream(request.prompt, request.max_tokens):
                # Format as SSE: "data: <json>\n\n"
                event_data = json.dumps({"token": token})
                yield f"data: {event_data}\n\n"
            # Send completion event
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


class LoadWeightsRequest(BaseModel):
    """Request model for loading weights."""
    weights_path: str


@app.post("/load_weights")
async def load_weights(request: LoadWeightsRequest):
    """Explicitly load model weights."""
    try:
        decoder.load_weights(request.weights_path)
        return {"status": "success", "weights_path": request.weights_path}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "weights_loaded": decoder._weights_loaded,
        "model_name": decoder.loaded_model_name,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
