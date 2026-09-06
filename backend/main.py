import sys
import os
import threading
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(".")

from app.mapf.model import CommTransformerNet
from app.mapf.training import train_a2c
from app.api import router as api_router, state

device = 'cpu'
WEIGHTS_PATH = "backend/app/mapf/pretrained_model.pth"

global_model = CommTransformerNet().to(device)
state["model"] = global_model

def run_background_training():
    print(f"Starting MAPF model background training on device: {device}...")
    train_a2c(
        model=global_model,
        episodes=20000,
        lr=1e-4,
        gamma=0.99,
        entropy_coeff=0.05,
        il_coeff=0.5,
        seq_len=10,
        device=device
    )
    torch.save(global_model.state_dict(), WEIGHTS_PATH)
    state["is_training"] = False
    print(f"Training complete. Weights saved to {WEIGHTS_PATH}.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading pre-trained model weights from {WEIGHTS_PATH}...")
        global_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        global_model.eval()
        state["is_training"] = False
        print("Pre-trained model loaded successfully. Ready to serve requests!")
    else:
        print("No pre-trained weights found. Starting background training...")
        training_thread = threading.Thread(target=run_background_training, daemon=True)
        training_thread.start()
    yield

app = FastAPI(title="MAPF Backend Service", lifespan=lifespan)

# Allow React app requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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