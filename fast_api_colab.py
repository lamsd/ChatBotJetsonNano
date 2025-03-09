import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import uvicorn

# Khởi tạo FastAPI
app = FastAPI()

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Định nghĩa cấu trúc dữ liệu đầu vào
class MessageInput(BaseModel):
    messages: list[str]

@app.post("/generate")
async def generate_text(data: MessageInput):
    input_messages = data.messages
    input_message_str = "\n".join(input_messages[-20:]) + "\nAI:"
    
    inputs = tokenizer.encode(input_message_str, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=inputs.shape[1] + 50, pad_token_id=tokenizer.eos_token_id)

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    last_response = response_text.split("AI:")[-1].strip()

    return {"response": last_response}

@app.post("/extract_name")
async def extract_name(data: MessageInput):
    input_messages = data.messages
    formatted_input_messages = [msg.replace("Human: ", "") for msg in input_messages if msg.startswith("Human: ")]
    
    input_message_str = "Extract the name from this conversation:\n" + "\n".join(formatted_input_messages) + "\n\nName:"
    
    inputs = tokenizer.encode(input_message_str, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=inputs.shape[1] + 10, pad_token_id=tokenizer.eos_token_id)

    res_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    name = res_text.split("Name:")[-1].strip().lower()

    if 'unknown' in name:
        return {"name": None}

    return {"name": name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
