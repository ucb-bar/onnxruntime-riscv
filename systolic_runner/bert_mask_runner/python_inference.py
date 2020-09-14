from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
	assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
	# Few properties that might have an impact on performances (provided by MS)
	options = SessionOptions()
	options.intra_op_num_threads = 1
	options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
	# Load the model as a graph and prepare the CPU backend 
	session = InferenceSession(model_path, options, providers=[provider])
	session.disable_fallback()
	return session

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model_inputs = tokenizer("Never going to [MASK] you up.", return_tensors="pt")
inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

cpu_model = create_model_for_provider("/home/centos/onnxruntime-riscv/systolic_runner/quantization/models/bert/bert.opt.onnx", "CPUExecutionProvider")

output = cpu_model.run(None, inputs_onnx)
output = output[0] # Model returns only 1 output tensor

output.shape # (1, 10, 28996)

MASK_TOKEN_IDX = [idx for idx, val in enumerate(inputs_onnx['input_ids'][0]) if val == tokenizer.mask_token_id]
assert(len(MASK_TOKEN_IDX) == 1)
MASK_TOKEN_IDX = MASK_TOKEN_IDX[0]

BATCH = 0

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


probs = softmax(output[BATCH][MASK_TOKEN_IDX][:])

# Get top 5

predictions = probs.argsort()[-5:][::-1]
values = probs[predictions]

result = []
for v, p in zip(values.tolist(), predictions.tolist()):
	tokens =  np.copy(inputs_onnx['input_ids'][0])
	tokens[MASK_TOKEN_IDX] = p
	tokens = tokens[np.where(tokens != tokenizer.pad_token_id)]
	result.append(
                    {
                        "sequence": tokenizer.decode(tokens),
                        "score": v,
                        "token": p,
                        "token_str": tokenizer.convert_ids_to_tokens(p),
                    }
                )

print([x["token_str"]  for x in result])