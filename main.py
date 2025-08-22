from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI()
device = torch.device("cpu")

# Model cache folder inside container
model_dir = "./model"

# Hugging Face GPTQ 4-bit CPU-friendly model (can override via env variable)
model_name = os.getenv("MODEL_NAME", "TheBloke/llama-2-7b-GPTQ-4bit-128g")
hf_token = os.getenv("HF_TOKEN")  # <-- Hugging Face token from environment

# Download and cache model if not already cached
if not os.path.exists(model_dir):
    print(f"Downloading GPTQ 4-bit model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_dir,
        use_auth_token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        cache_dir=model_dir,
        use_auth_token=hf_token
    )
    print("✅ Model downloaded and cached in ./model")
else:
    print("Loading model from cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map={"": device}, use_auth_token=hf_token)

# Request schema
class RequestData(BaseModel):
    prompt: str
    max_tokens: int = 100

# Endpoint for text generation
@app.post("/generate")
def generate_text(data: RequestData):
    inputs = tokenizer(data.prompt, return_tensors="pt").to(device)
    with torch.no_grad():  # prevents gradient computation, saves CPU/memory
        outputs = model.generate(**inputs, max_new_tokens=data.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}

# Health check endpoint
@app.get("/")
def home():
    return {"status": "CPU-friendly GPTQ 4-bit LLaMA API running"}
