# ort_test

Runs imagenet models. Example invocation that runs a given model (`-m`) on an input image (`-i`) with caffe style image processing (`-p`) on CPU only (`-x`).
```
qemu ort_test -m bvlc_alexnet/model_int8_quant.onnx -i images/dog.jpg -p caffe2 -x 0
```

```
spike --extension=gemmini pk ort_test -m googlenet.onnx  -i images/cat.jpg  -p caffe2 -x 1 -O 0
```

# Tracing

Trace files can be emitted by passing the `-t` option along with a filename. Emitted files are in the Google Trace Event format, and can be viewed as a flame graph via `chrome://tracing`. (Tip: w/a/s/d keys allow you to navigate around/zoom, and `f` key changes the zoom level to focus on the selected event).

For a given event, most relevant information will be present in the `Args` section of the bottom panel. In particular, the the provider (CPU or Systolic), and operator name (e.g. `QuantizeLinear`) can be found here. For operations scheduled on systolic, additional intra-operator information may be available. For instance, convolutions scheduled on Systolic will show the breakdown of im2col, bias splat, and matmul time.
