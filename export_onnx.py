from pathlib import Path
from transformers.onnx import export
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2 import GPT2OnnxConfig, GPT2Config

onnx_path = Path("gpt2.onnx")
config = GPT2Config.from_pretrained('gpt2')
onnx_config = GPT2OnnxConfig(config=config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(
    'gpt2', pad_token_id=tokenizer.eos_token_id)

onnx_inputs, onnx_outputs = export(
    preprocessor=tokenizer, model=model, config=onnx_config,
    opset=onnx_config.default_onnx_opset, output=onnx_path)
