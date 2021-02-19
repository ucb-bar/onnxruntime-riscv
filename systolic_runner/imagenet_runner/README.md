# Imagenet Runner

Runs imagenet models. Example invocation that runs a given model (`-m`) on an input image (`-i`) with caffe style image processing (`-p`) on CPU only (`-x`).
```
qemu ort_test -m bvlc_alexnet/model_int8_quant.onnx -i images/dog.jpg -p caffe2 -x 0
```

```
spike --extension=gemmini pk ort_test -m googlenet.onnx  -i images/cat.jpg  -p caffe2 -x 1 -O 0
```

## E2E Resnet Example

Download the "ResNet50-caffe2" (Opset 9) and associated test data from the ONNX Model zoo.

First optimize the model via: 

```
python3 optimize.py --input=models/resnet50/model.onnx  --output=models/resnet50/model_opt.onnx
```

then run

```
python3 calibrate.py --model_path $MODEL/model_opt.onnx   --dataset_path $MODEL --output_model_path $MODEL/model_opt_quantized.onnx  --static=True --data_preprocess=mxnet --mode=int8
```

Finally, inference can be performed via

```
qemu ort_test -m ../quantization/models/resnet50/model_opt_quantized.onnx   -i images/dog.jpg   -p mxnet -x 0 -O 99
```

where `-x 0` runs the CPU simulation of Gemmini (needed since we run on Qemu) and `-O 99` enables the NHWC optimization.

## Tracing

Trace files can be emitted by passing the `-t` option along with a filename. Emitted files are in the Google Trace Event format, and can be viewed as a flame graph via `chrome://tracing`. (Tip: w/a/s/d keys allow you to navigate around/zoom, and `f` key changes the zoom level to focus on the selected event). You can also try loading it in Perfetto, Tracy or Speedscope.

For a given event, most relevant information will be present in the `Args` section of the bottom panel. In particular, the the provider (CPU or Systolic), and operator name (e.g. `QuantizeLinear`) can be found here. For operations scheduled on systolic, additional intra-operator information may be available. For instance, convolutions scheduled on Systolic will show the breakdown of im2col, bias splat, and matmul time.

More detailed analysis can be performed by parsing the resulting json directly. For instance, the following provides a breakdown of inference time by op type:

```
map(select(.args.op_name != null))  | group_by(.args.op_name) | map({(.[0].args.op_name) : map(.dur) | add}) | sort_by(.[]) | add
```

Some other links of interest that will parse files to generate report:
https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/perfstats.py
https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/tensorrt/perf

## Benchmark

We provide some tools to make benchmarking on the ILSVRC2012 easy. Use `preprocess_batch` to sample a number of images, preprocess by cropping, and write a list of file paths that can be consumed by the runner. Use `batch_infer` to parallelize execution, spawning several runners for each split. Since the results of CPU emulation of Gemmini should be equivalent to results from the real thing, we recommend using `qemu` to get a guage of accuracy quickly. `postprocess_batch.py` will analyze the output files from the batch run to get the aggregated statistics.

If you do not want to download the entire imagenet data set and only need to run on a sample, you can use an imagenet downloader such as seen [here](https://github.com/mf1024/ImageNet-Datasets-Downloader). Patch `downloader.py` to the [following](https://gist.github.com/pranav-prakash/7b8a292776ffd35a6517f414de05e3c8) to automatically preprocess the images (you will need to ensure that `classes_to_labels.txt` is copied from `tools` to the run directory. The following will select 10 imagenet classes from the IVLSRC2012 set, and download 2 images inside each class:

```
python3 downloader.py -data_root ../imagenet -use_class_list True -number_of_classes 10 -images_per_class 2
```

The output folder is structured so the batch input file can be immediately constructed from the file path. E.g.

```
imagenet/imagenet_images/233/420837932_f9977dd3b7.jpg
```


## Regression Test Info

For reference, when running resnet50 converted using the procedure above on the dog image, the results should be:

```
Element count 1000. Top 5 classes:
0.002819 Great Dane
0.002825 German short-haired pointer
0.009029 curly-coated retriever
0.013360 flat-coated retriever
0.961574 Labrador retriever
```

Minor (< 1%) deviations between version bumps might be due to changes in ORT internal implementation; large unexpected deviations should be analyzed more carefully.
