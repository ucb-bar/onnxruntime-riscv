import onnxruntime as rt
import numpy
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='ImageNet native ORT')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
args, args_other = parser.parse_known_args()

# sess_options = rt.SessionOptions()
# sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED


sess = rt.InferenceSession(args.model)
input_name = sess.get_inputs()[0].name


def get_expected(path):
    return int(path.split('/')[-2])

# This assumes a preprocessing for model exported by pytorch
# Model from model zoo is different
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

def perform_inference(file):
    preprocessed = load_image(file)
    print(file)
    result = sess.run(None, {input_name: preprocessed})
    import pdb; pdb.set_trace()


perform_inference(args.image)
