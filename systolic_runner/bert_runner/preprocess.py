import tokenization as tokenization
from run_onnx_squad import *

# preprocess input
predict_file = 'inputs.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(
  eval_examples,
  tokenizer,
  max_seq_length,
  doc_stride,
  max_query_length
)

print("input ids", input_ids.shape)
print("input masks", input_ids.shape)
print("segment ids", input_ids.shape)
print("extra data", input_ids.shape)

if not os.path.exists("input_data"):
  os.makedirs("input_data")

with open(os.path.join("input_data", "unique_ids.bin"), "wb") as fh:
  unique_ids = np.array([ex.qas_id for ex in eval_examples], dtype=np.int64)
  unique_ids.tofile(fh)

with open(os.path.join("input_data", "input_ids.bin"), "wb") as fh:
  input_ids.tofile(fh)

with open(os.path.join("input_data", "input_masks.bin"), "wb") as fh:
  input_mask.tofile(fh)

with open(os.path.join("input_data", "segment_ids.bin"), "wb") as fh:
  segment_ids.tofile(fh)

with open(os.path.join("input_data", "dimensions.bin"), "wb") as fh:
  np.array(input_ids.shape, dtype=np.int64).tofile(fh)
