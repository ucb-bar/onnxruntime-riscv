# MNIST Trainer

All examples below use the [linked](https://github.com/microsoft/onnxruntime/issues/3706#issuecomment-621372668) models and test data.

## Matmul 

```
qemu mnist_train --model_name mnist/mnist_gemm.onnx --train_data_dir mnist/mnist_data/ --num_train_steps 1 -x 0
```