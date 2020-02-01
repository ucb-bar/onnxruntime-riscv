#include <vector>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {

 size_t input_tensor_size = 3 * 224 * 224;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  printf("Loading image\n");
  std::vector<const char*> output_node_names = {"vgg0_dense2_fwd"};


	int dimX, dimY, numChannels;
	unsigned char *data = stbi_load("dog.jpg", &dimX, &dimY, &numChannels, 0);
  printf("Loaded Image: %d %d %d", dimX, dimY, numChannels);
	
	float *input_tensor_values = new float[input_tensor_size];
	
	for (int i = 0; i < 224; i++) {
		for (int j = 0; j < 224; j++) {
			input_tensor_values[(0*224 + i)*224 + j] = ((*(data++))/255.0 - 0.485)/0.229;
			input_tensor_values[(1*224 + i)*224 + j] = ((*(data++))/255.0 - 0.456)/0.224;
			input_tensor_values[(2*224 + i)*224 + j] = ((*(data++))/255.0 - 0.225)/0.225;	
		}
	}
  printf("First few image values %f %f %f", input_tensor_values[0], input_tensor_values[1], input_tensor_values[2]);

}
