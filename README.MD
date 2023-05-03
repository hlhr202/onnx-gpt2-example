# gpt2 onnx example
This is an example that shows how to convert hf transformers(gpt2) to onnx, and then use onnxruntime to do inference

## install
```
pip install -r requirements.txt
```

## export
```
python export_onnx.py
```

## infer
```
python onnx_infer.py
```