import onnx, sys
from onnx import version_converter, helper

# Preprocessing: load the model to be converted.
model_path = sys.argv[1]

original_model = onnx.load(model_path)

#print('The model before conversion:\n{}'.format(original_model))

converted_model = version_converter.convert_version(original_model, 9)

#print('The model after conversion:\n{}'.format(converted_model))

new_model_path = model_path[:-5] + "_op9.onnx"
onnx.save(converted_model, new_model_path)
