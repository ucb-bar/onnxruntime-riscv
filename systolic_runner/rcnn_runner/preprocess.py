import numpy as np
from PIL import Image

def preprocess(image):
	# Resize
	ratio = 800.0 / min(image.size[0], image.size[1])
	image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)
	image = np.array(image)

	# Pad to be divisible of 32
	import math
	padded_h = int(math.ceil(image.shape[0] / 32) * 32)
	padded_w = int(math.ceil(image.shape[1] / 32) * 32)

	padded_image = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
	padded_image[:image.shape[0], :image.shape[1], :] = image
	image = padded_image

	return image

img = Image.open('demo.jpg')
img_data = preprocess(img)
Image.fromarray(img_data).save("preprocess.jpg")