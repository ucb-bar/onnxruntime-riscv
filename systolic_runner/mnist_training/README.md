# MNIST Trainer

All examples below use the [linked](https://github.com/microsoft/onnxruntime/issues/3706#issuecomment-621372668) models and test data.

The models are built from the script in `orttraining/tools/mnist_model_builder`.

```
qemu mnist_train --model_type conv --model_name mnist/mnist_conv.onnx --train_data_dir mnist/mnist_data/ --num_train_steps 1 -x 0
```

You will want to remove all non-input initializers from the graph before doing the training

```
python3 ../../../tools/python/remove_initializer_from_input.py --input mnist_conv_w_batchnorm.onnx  --output mnist_conv_w_batchnorm_noinitializers.onnx
```

Note that the backward graph is serialized as an onnx file as well so it can be visualized. 
See the source file for all the training-related options that can be passed.