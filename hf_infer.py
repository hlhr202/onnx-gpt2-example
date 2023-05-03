import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

batch_size = 1
length = 50
device = torch.device("cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained(
    'gpt2', pad_token_id=tokenizer.eos_token_id)

input_ids = tokenizer.encode("My name is Teven and I am", return_tensors='pt')

greedy_output = model.generate(input_ids, max_length=length)

print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
