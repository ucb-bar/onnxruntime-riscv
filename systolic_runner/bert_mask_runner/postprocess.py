from transformers import BertTokenizer
from INPUT import INPUT
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model_inputs = tokenizer(INPUT, return_tensors="pt")

batch_seq_dims = np.fromfile("inputs/dims.data", dtype=np.int64)
input_ids = np.fromfile("inputs/input_ids.data", dtype=np.int64).reshape((batch_seq_dims[0], batch_seq_dims[1]))
output = np.fromfile("outputs/output.data", dtype=np.float32).reshape((batch_seq_dims[0], batch_seq_dims[1], 28996))

BATCH = 0

MASK_TOKEN_IDX = [idx for idx, val in enumerate(input_ids[BATCH]) if val == tokenizer.mask_token_id]
assert(len(MASK_TOKEN_IDX) == 1)
MASK_TOKEN_IDX = MASK_TOKEN_IDX[0]



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
	tokens =  np.copy(input_ids[BATCH])
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