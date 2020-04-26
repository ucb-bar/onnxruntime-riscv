# ort_test

Runs imagenet models. Example invocation that runs a given model (`-m`) on an input image (`-i`) with caffe style image processing (`-p`) on CPU only (`-x`).
```
qemu ort_test -m bvlc_alexnet/model_int8_quant.onnx -i images/dog.jpg -p caffe2 -x 0
```

```
spike --extension=gemmini pk ort_test -m googlenet.onnx  -i images/cat.jpg  -p caffe2 -x 1 -O 0
```
