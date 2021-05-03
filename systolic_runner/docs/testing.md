# Testing

ONNX Runtime has three kinds of tests that can be run: unit tests in the form of mlas tests, unit tests in the form of op-level tests (or graph transform tests), and "end-to-end" tests where you run a given model and compare it against a reference.

## MLAS Unit Tests

These unit tests are the lowest level of tests -- in the upstream ORT they ensure that SIMD kernels for Gemm/Conv/etc. are implemented properly; in our fork (where we call into gemmini.h), they mainly serve to ensure that the _hardware_ is implemented correctly. This is one of the most important tests, as hardware bugs have been common -- which will manifest as either stalling (simulation freeze) or incorrect results; both of which would be hard to track down and fix without the MLAS unit tests providing a smaller replicable test case.

The main driver for this test is located in `onnxruntime/test/mlas/unittest.cpp`, and we have separate headers for systolic and hwacha tests which we include in here. These tests are built as part of building ORT, and the test binary is located in `build/Release/onnxruntime_mlas_test`. You can run the test in WS mode like

```
./onnxruntime_mlas_test foo bar
```

where the number of arguments indicates which mode should be run (`foo bar` -> `2` -> `WS`). E.g. if you wanted to run in CPU mode (`-x 0`) then you don't use any arguments: `./onnxruntime_mlas_test`.(Yes I realize this is a terrible mechanism, but I didn't want to drag in an arg parsing library. I suppose I could have just used `atoi` though).

One important thing to note is that these tests _do not_ depend on the main ORT libraries. That is, you can basically take the .cpp and .h files it depends on, and compile them standalone -- either to a linux executable or a bare metal binary. This ability to compile a baremetal version has been very useful in debugging some edge cases.

Also note that the MLAS tests make use of a "guard region" to detect OOB reads or writes. This is accomplished via `mprotect` and must be disabled if you want to compile a baremetal version. If you see stalling, disabling the guard region should be your first step to diagnose if this is an OOB read/write issue or not.

## Operator Level Unit tests

We have operator-level tests for most operators implemented by the Systolic backend. These op-level tests are more generally part of ONNXRuntime's comprehensive unit-test suite (using the gTest framework). You can see the systolic tests in `onnxruntime/test/providers/systolic`. N.b. these _do not_ work in Spike.

After building ORT you will see an `onnxruntime_test_all` binary in the build folder. You can run the test like

```
qemu onnxruntime_test_all --gtest_filter="*SystolicConvGradTest*NHWC*"
``` 

Note that `gtest_filter` arg; you'll probably want to use this to limit the tests to a specific one.

## E2E Model Tests

There's no automated tests for these, this is just what I call "running a model and eyeballing that the results make sense." We should probably set up some record of what the expected model outputs are so we can do golden-master testing. When you make a large modification to ORT though (e.g. bumping to upstream) running a model is probably the easiest way to exercise the main Gemmini-related codepaths and make sure you haven't broken anything.
