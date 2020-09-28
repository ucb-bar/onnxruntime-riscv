import onnxruntime as rt
import numpy
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='ImageNet native ORT')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--batch', type=str, required=True)
args, args_other = parser.parse_known_args()

sess = rt.InferenceSession(args.model)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


inputs = []
with open(args.batch, 'r') as f:
    inputs = f.read().splitlines()

def get_expected(path):
    return int(path.split('/')[-2])

def load_image(img_path):
    # CV loads in BGR, and caffe2 expects BGR
    loaded = cv2.imread(img_path)
    #loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    img_data = loaded.transpose(2, 0, 1)

    # However, the mean values provided are in RGB format
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])

    # So swap the mean vec to BGR
    mean_vec = mean_vec[::-1] 
    stddev_vec = stddev_vec[::-1]

    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):  
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    norm_img_data = np.expand_dims(norm_img_data, axis=0)
    return norm_img_data

def perform_inference(file):
    preprocessed = load_image(file)
    print(file)
    result = sess.run(None, {input_name: preprocessed})
    prob = np.ravel(result[0])
    return np.argmax(prob)


num_correct = 0
num_total = 0
for f in inputs:
    num_total += 1
    res = perform_inference(f)
    if (res == get_expected(f)):
        num_correct += 1

print('Top 1 accuracy: {}/{}'.format(num_correct, num_total))
