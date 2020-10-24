void HwachaDepthWiseConv
MLASCALL(const size_t batch_size,
         const size_t group_count,
         const size_t channels,
         const size_t in_height, const size_t in_width,
         const size_t filter_count,
         const size_t kernel_height, const size_t kernel_width,
         const size_t pad_left_height, const size_t pad_left_width,
         const size_t pad_right_height, const size_t pad_right_width,
         const size_t dilation_height, const size_t dilation_width,
         const size_t stride_height, const size_t stride_width,
         const size_t out_height, const size_t out_width,
         const int8_t* input, const int8_t* filter, const int32_t* bias,
         int8_t* output, const unsigned int rounded_divisor);

void Hwachaim2col
MLASCALL(size_t batch_size, size_t height, size_t width, size_t channels,
         size_t I, size_t K,
         const int8_t* input_arr, const int8_t* output_arr, const struct ConvParams* params);