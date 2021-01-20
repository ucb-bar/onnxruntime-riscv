# BERT Mask Runner

Runs a BERT mask-word model on onnxruntime. See below for E2E example.

## E2E Example

* Install the Huggingface transformers python package. 
* Run ./create_model.sh to create the ONNX mask word model
* Run `optimize_model.py` to create the fused attention nodes (needs installation of onnxruntime pip package)
* At this point you should be able to verify that the fp model successfully infers the masked word
    * Type your sentence in `INPUT.py`, run `preprocess.py`, build & run `ort_test`, then run `postprocess.py`
* To quantize, you will need to first run `create_pb_input.py` to create the pb calibration inputs for some sample text inputs
    * Call with `--static=True --data_preprocess=None --mode=int8`
    * If you call with mode=uint8 you should be able to verify that the model runs fine.
    * But with int8 quantization to run on Systolic you need one more step:


## QKV Quantization to int8

Because Systolic only has an int32 accumulator but the Attention bias are in floating point, we need to quantize the bias as well. However, this is complicated by the fact that the QAttention op is composed of both the matmul and the Gelu – we need access to the intermediary matmul output. To do this we recompile ORT with the `PRINT_QUANTIZATION_SCALES` ifdef on Systolic's `attention_quant` op.

This will dump out quantization scale info which can be fed into `bert_qkv_scales.py`. That will augment the quantized int8 model created in the last step with the appropriate scale factors.