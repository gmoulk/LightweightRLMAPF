import sys
import threading
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI

sys.path.append(".")

from app.mapf.model import CommTransformerNet
from app.mapf.training import train_a2c
from app.api import router as api_router, state

device = 'cpu'
global_model = CommTransformerNet().to(device)
state["model"] = global_model

def run_background_training():
    print(f"Starting MAPF model background training on device: {device}...")
    train_a2c(
        model=global_model,
        episodes=5000,
        lr=1e-4,
        gamma=0.99,
        entropy_coeff=0.05,
        il_coeff=0.5,
        seq_len=10,
        device=device
    )
    state["is_training"] = False
    print("Background training completed. Model is ready to solve requests!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    training_thread = threading.Thread(target=run_background_training, daemon=True)
    training_thread.start()
    yield

app = FastAPI(title="MAPF Backend Service", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "message": "MAPF API Service is Running",
        "training_active": state["is_training"],
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)