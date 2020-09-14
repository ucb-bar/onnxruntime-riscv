from transformers import BertTokenizer
from INPUT import INPUT
import os
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model_inputs = tokenizer(INPUT, return_tensors="pt")
inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
for inp in inputs_onnx:
    inputs_onnx[inp].tofile(os.path.join('inputs', inp + '.data'))

np.array(inputs_onnx['input_ids'].shape).tofile(os.path.join('inputs', 'dims.data'))

print(inputs_onnx)