# Quantizer Script

## Overview

For a general overview on the quantizer script first read the quantization.py [README](/systolic_runner/quantization/README.md)

In short, the quantization script converts a floating point model into one where compatible operators have been replaced
by int8 quantized equivalents.

While the majority of the process is identical to Microsoft's original script, a few Systolic specific transformations have been added which will be detailed below. Unless otherwise specified all transformations specified below only apply when quantizing to int8 (not uint8 – which Microsoft's original quantizer was limited to).

## Calibration script

The calibration script is responsible for running the floating point ONNX model and determining quantization parameters for every node.
Under dynamic quantization, compouted parameters are limited to Conv and Matmul nodes; the rest are computed dynamically at runtime (see dynamic quantization in the quantizaiton README).

## Quantizer 

The quantizer goes through each node in the original ONNX proto, converting nodes to their quantized equivalents.

Currently the following nodes are handled

* Conv: Converted to a QLinearConv or QLinearConv_nhwc depending on whether NHWC parameter is passed (see following section)

* Matmul: Converted to a QLinearMatmul

* Relu: When quantizing to uint8 Relu is removed entirely. When quantizing to int8, if the input to Relu is already quantized we output a custom-typed QLinearRelu node that onnxruntime internally fuses with a QLinearConv/QLinearMatmul.

We keep track of and propagate the quantization status of every edge in the graph, so given any input to a node we can determine whether it is already in int8 or not. If the input is floating point but the node can be converted to a quantized equivalent, we insert a QuantizeLinear node with the parameters from the output of the calibration script for that layer. If the input is quantized but the operator has no quantized equivalent, we insert a DequantizeLinear node.

Whenever we update a node to be quantized, we also quantize its weights and update the initializer in the ONNX proto. Similarly, the input scale/zero point information is also added to the initializer proto.

## NHWC support

For QLinearConv we support NHWC data layouts via our own custom type QLinearConv_nhwc when the `--nhwc` parameter is used with the quantization script. The process remains identical, except we also keep track of the data layouts of each value, inserting `Transpose` nodes where appropriate if we try to pass a value in NHWC format to an operator that cannot handle it (or vice-versa).

## Example

Consider the following snippet from the grpah of Googlenet:

![](googlenet_unquantized.png)

After running the quantization script the model is transformed to:

![](googlenet_quantized.png)
