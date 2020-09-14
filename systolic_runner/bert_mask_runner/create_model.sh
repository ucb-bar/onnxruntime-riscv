# Create a BERT model from the fill-mask pipeline.
# Choose this since it's the easiest to write a runner and verify results for
python3 /usr/local/lib/python3.6/site-packages/transformers/convert_graph_to_onnx.py  \
    --framework pt --model bert-base-cased --pipeline fill-mask onnx/bert-base-cased.onnx