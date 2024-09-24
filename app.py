
import os
from flask import Flask, request, jsonify
from airllm import AutoModel

app = Flask(__name__)

MAX_LENGTH = 128
MODEL_PATH = "/app/model"

# Load the model
model = AutoModel.from_pretrained(MODEL_PATH)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    
    # Combine all messages into a single input text
    input_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    input_tokens = model.tokenizer(input_text,
        return_tensors="pt", 
        return_attention_mask=False, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding=False)
               
    generation_output = model.generate(
        input_tokens['input_ids'].cuda(), 
        max_new_tokens=50,  # Adjust as needed
        use_cache=True,
        return_dict_in_generate=True)

    output = model.tokenizer.decode(generation_output.sequences[0])
    
    response = {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "airllm-meta-llama-3.1-405b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(input_tokens['input_ids'][0]),
            "completion_tokens": len(generation_output.sequences[0]) - len(input_tokens['input_ids'][0]),
            "total_tokens": len(generation_output.sequences[0])
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
