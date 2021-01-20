# MNIST Trainer

All examples below use the [linked](https://github.com/microsoft/onnxruntime/issues/3706#issuecomment-621372668) models and test data.

The models are built from the script in `orttraining/tools/mnist_model_builder`.

```
qemu mnist_train --model_type conv --model_name mnist/mnist_gemm.onnx --train_data_dir mnist/mnist_data/ --num_train_steps 1 -x 0
```

Note that the backward graph is serialized as an onnx file as well so it can be visualized.