from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI()
device = torch.device("cpu")

# Model cache directory
model_dir = "./model"

# Use a CPU-friendly model (default = tiny GPT-2)
model_name = os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")

# Global vars for tokenizer & model
tokenizer = None
model = None

class RequestData(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.on_event("startup")
def load_model():
    global tokenizer, model
    os.makedirs(model_dir, exist_ok=True)
    print(f"🔄 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir)
    model.to(device)
    print("✅ Model ready")

@app.post("/generate")
def generate_text(data: RequestData):
    try:
        inputs = tokenizer(data.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=data.max_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": f"🚀 CPU-friendly model API running with {model_name}"}
