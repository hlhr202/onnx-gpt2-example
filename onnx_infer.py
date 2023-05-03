from onnxruntime import InferenceSession
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
import sys

length = 100
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
session = InferenceSession("gpt2.onnx")

text = "My name is Teven and I am"

print(text, end="")

# I do not understand very well for this logits sampling.
for i in range(length):
    input_tokens = tokenizer(text, return_tensors="np")
    outputs = session.run(
        output_names=['last_hidden_state'], input_feed=dict(input_tokens))
    logits = outputs[0]
    logits = torch.from_numpy(logits[:, -1, :])
    log_probs = F.softmax(logits, dim=-1)
    _, prev = torch.topk(log_probs, k=1, dim=-1)
    if prev.item() == tokenizer.eos_token_id:
        break
    current_token = tokenizer.decode(prev.item())
    print(current_token, end="")
    sys.stdout.flush()
    text += current_token
