import onnxruntime
from INPUT import CONTEXT, QUERY
from preprocess import preprocess
import numpy as np

cw, cc = preprocess(CONTEXT)
qw, qc = preprocess(QUERY)
sess = onnxruntime.InferenceSession('/home/centos/onnxruntime-riscv/systolic_runner/quantization/models/bidaf/model_opt.onnx')
answer = sess.run(['start_pos', 'end_pos'], {'context_word':cw, 'context_char':cc, 'query_word':qw, 'query_char':qc})
print(answer)
start = np.asscalar(answer[0])
end = np.asscalar(answer[1])
print(cw[start:end+1])