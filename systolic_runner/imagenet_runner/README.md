# ort_test

Runs imagenet models. Example invocation that runs a given model (`-m`) on an input image (`-i`) with caffe style image processing (`-p`) on CPU only (`-x`).
```
qemu ort_test -m bvlc_alexnet/model_int8_quant.onnx -i images/dog.jpg -p caffe -x 0
```
