from transformers import BertTokenizer
from INPUT import INPUT
import os
import numpy as np
import pathlib
from onnx import numpy_helper

OUTPUT_DIR = '/home/centos/onnxruntime-riscv/systolic_runner/quantization/models/bert/onnx'

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model_inputs = tokenizer(INPUT, return_tensors="pt")
inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
for inp in inputs_onnx:
    inputs_onnx[inp].tofile(os.path.join('inputs', inp + '.data'))

for i in range(1):
    pathlib.Path(os.path.join(OUTPUT_DIR, 'test_data_set_{}'.format(i))).mkdir(parents=True, exist_ok=True) 

    with open(os.path.join(OUTPUT_DIR, 'test_data_set_{}'.format(i), 'input_0.pb'), 'wb') as f:
        f.write(numpy_helper.from_array(inputs_onnx['input_ids']).SerializeToString())
    with open(os.path.join(OUTPUT_DIR, 'test_data_set_{}'.format(i), 'input_1.pb'), 'wb') as f:
        f.write(numpy_helper.from_array(inputs_onnx['attention_mask']).SerializeToString())
    with open(os.path.join(OUTPUT_DIR, 'test_data_set_{}'.format(i), 'input_2.pb'), 'wb') as f:
        f.write(numpy_helper.from_array(inputs_onnx['token_type_ids']).SerializeToString())