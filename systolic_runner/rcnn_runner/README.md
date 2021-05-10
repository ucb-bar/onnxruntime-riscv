# RCNN runner

## Procuring a model

There are two places you can get a suitable pre-trained model: one is from the ONNX model zoo, which has an opset10 mask-rcnn model. The other place is exporting from pytorch/torchvision (more on that later). The choice of which model you get will affect the preprocessing options needed. So based on the model type, you will have to modify 3 files:

* runner.cpp (search for "pytorch" to see the two places that need changing)
* postprocess.py (see the comments in there)
* ../quantization/data_preprocess.py (see the commented out maskrcnn function there)

The mask-rcnn model from the model-zoo is the easiest to work with: it quantizes nicely out of the box, and seems to have good accuracy even after quantization (see the section on quantization for more precise info). However, a big downside to using this model is that during the export batch-norm was unrolled into a sequence of mul-add chains which cannot be quantized without losing accuracy; this becomes a bottleneck to performance.

The mask-rcnn model from pytorch is more flexible and can be coaxed into what we want. But you have to do some hacks to get it to export in a format we want. Additionally, it seems that the exported version doesn't give proper predictions (I opened an issue [here](https://github.com/pytorch/vision/issues/3588) – hopefully that is fixed or root cause diagnosed soon).

## Exporting the pytorch model

The stock torchvision model uses a `FrozenBatchNorm2d` operator, which we don't since during export that is mapped onto add + mul instead of the ONNX BN op. We hack around this by changing the pytorch source. In `torchvision/ops/misc.py` replace `FrozenBatchNorm2d` with

```
def FrozenBatchNorm2d(*args, **kwargs):
    return torch.nn.BatchNorm2d(*args, **kwargs)
```

Then in `torchvision/models/detection/_utils.py:369` replace

```
        if isinstance(module, FrozenBatchNorm2d):
```
with
```
        if isinstance(module, torch.nn.BatchNorm2d):
```

Now you can run

```
model_tv = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model_tv.eval()
torch.onnx.export(model_tv, torch.rand(1,3,800,800), "mask_rcnn_r50_fpn.onnx",
                  do_constant_folding=True,
                  opset_version=12  # opset_version 11 required for Mask R-CNN
                  )
```

and when you load the exported model in netron you should see that the BN was folded into the conv during export,
so you should have a nice clean resnet50 backbone.

## Quantization

If you want to quantize the model from the model zoo, you will have to comment out the `QLinearAdd` and `QLinearMul` in registry.py. Hand wavily,  the model from the model zoo has the batchnorm unrolled to add + mul ops, and BN can be sensitive so quantization so quantizing those two ops will destroy accuracy. (Note that if you're using a PyTorch export with the BN fused into the Conv, then you can go ahead and enable `QLinearAdd`)

Before running either model through the quantizer, you'll want to run `optimize.py` first with `replace_gemm=True` (can experiment with this to see how it affects accuracy). Then follow the usual calibrate/quantize one-shot method

```
python3 calibrate.py --model_path $MODEL/model_opt.onnx   --dataset_path $MODEL --output_model_path $MODEL/model_quantized.onnx --static=True --data_preprocess=rcnn --mode=int
```

Note that if you're using a model exported from pytorch, you will have to manually create a suitable calibration dataset. You could do something like

```
import cv2
import numpy as np
def load_image(img_path):
    # CV loads in BGR, and rcnn expects rgb
    loaded = cv2.imread(img_path)
    loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    img_data = loaded.transpose(2, 0, 1)
    
    # The mean values provided are in RGB format
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):  
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    norm_img_data = np.expand_dims(norm_img_data, axis=0)
    return norm_img_data
load_image("/home/centos/onnxruntime-riscv/systolic_runner/rcnn_runner/images/square.jpg")
from onnx import numpy_helper
tensor = numpy_helper.from_array(load_image("/home/centos/onnxruntime-riscv/systolic_runner/rcnn_runner/images/square.jpg"))
with open('tensor.pb', 'wb') as f:
    f.write(tensor.SerializeToString())
```

and then use `--data_preprocess=None` during calibration since we've already preprocessed the image.



## Invocation

With the above out the way, one can run this like
```
qemu ort_test -m model_quantized.onnx  -i images/square.jpg -x 0 -O 99 -d 0 
```

Refer to the imagenet runner (or the source) for info on what the various options do.

There's also a native ort runner for a uint8 quantized (or FP) version:

```
python3 ort_native.py --model ../quantization/models/maskrcnn/rcnn_12/model_quantized.onnx     --image images/square.jpg 
```
