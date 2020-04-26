# Halide Interop

ORT can interop with Halide by using Halide ahead of time to schedule nodes or subgraphs, generating a custom
ORT kernel that can be linked with the imagenet runner. 

`./build.sh` will automatically download a binary release of LLVM 9.0 and use it to build Halide for a risc-v target. 

`model_converer/` contains scripts to process and generate a transformed graph along with a suitable kernel-spec for ORT.
Currently any non-conv/matmul node is scheduled individually on Halide, although this can be tweaked to schedule entire subgraphs.

We identify Halide-scheduled nodes by replacing the original OpType in the ONNX proto with an OpType of `CustomOp` + `hash`
where `hash` is computed as `node.op_type + "_"+ hash_dn(node.op_type + ''.join(node.input) + ''.join(node.output))`.

This allows multiple Halide-scheduled nodes for the same OpType to co-exist simultaneously
(TODO: perhaps this should be extended to hash attributes as well? Or just use a GUID?)

When `imagenet` runner is built, it will look inside the `./generated/` folder to see if a kernel spec exists.
If so, it will be automatically linked in to the runner.
Note that while it is possible to build the kernel spec as a shared library and dynamically load it from the runner, this does not play well with spike.
