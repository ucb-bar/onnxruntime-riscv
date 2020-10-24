#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>

#include "util.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
void HwachaDepthWiseConv(const size_t batch_size,
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
                         int8_t* output, const unsigned int rounded_divisor) {
  // printf("Starting Hwacha Depth Wise Convolution!\n");
  // printf("Batch Size: %li\n", batch_size);
  // printf("Group Count: %li\n", group_count);
  // printf("Channels: %li\n", channels);
  // printf("Filter Count: %li\n", filter_count);
  // printf("Input Shape: Height: %li Width: %li\n", in_height, in_width);
  // printf("Output Shape: Height: %li Width: %li\n", out_height, out_width);

  printf("Padding Left Height: %li Padding Left Widdth: %li Padding Right Height: %li Padding Right Width: %li\n",
         pad_left_height, pad_left_width, pad_right_height, pad_right_width);

  // printf("\n");
  // printf("input\n");
  // for (size_t m = 0; m < in_height; m++) {
  //   for (size_t k = 0; k < in_width * channels; k++) {
  //       printf("%i ", input[m * channels * in_width + k]);
  //   }
  //   printf("\n");
  // }

  // printf("\n");
  // printf("hwdc input\n");
  // for(size_t c = 0; c < channels; c++){
  //   printf("Channel %i\n",c);
  //   for (size_t y = 0; y < in_height; y+=1) {
  //     for (size_t x = c; x < in_width * channels; x+=channels) {
  //         printf("%i ", input[x + in_width * channels * y]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }
  int16_t temp_output[channels];
  for (size_t k = 0; k < channels; k++) {
    temp_output[k] = (int16_t)0;
  }

  //set zero in hwacha
  for (size_t m = 0; m < out_height; m++) {
    for (size_t k = 0; k < out_width * channels; k++) {
      output[m * channels * out_width + k] = (int8_t)0;
    }
  }

  setvcfg(0, 0, 4, 1);
  int consumed = setvlen(channels);
  printf("Consumed length: %i\n", consumed);
  printf("Rounded Divisor: %i\n", rounded_divisor);

  int input_idx = 0;
  int input_idy = 0;

  int8_t* input_ptr = (int8_t*)input;

  for (int output_idy = 0; output_idy < out_height; output_idy += 1) {
    for (int output_idx = 0; output_idx < out_width * channels; output_idx += channels) {
      for (size_t k = 0; k < channels; k++) {
        temp_output[k] = (int16_t)0;
      }
      asm volatile("vmca va0, %0"
                   :
                   : "r"(temp_output));  //temp_output
      //asm volatile ("vmca va3, %0" : : "r" (output + output_idx + output_idy*out_width*channels)); //output

      //input_ptr = (int8_t*) input + output_idx + output_idy * in_width*channels;
      input_idy = output_idy - pad_left_height;
      for (int filter_idy = 0; filter_idy < kernel_height; filter_idy++) {
        if (output_idy == 0 && filter_idy == 0 && pad_left_height == 1) {
          printf("pad top buffer zero. output_y: %i output_x: %i filter_y: %i \n",
                 output_idy, output_idx, filter_idy);
          input_idy += 1;
          continue;
        }

        input_idx = output_idx - pad_left_width * channels;
        for (int filter_idx = 0; filter_idx < kernel_width * channels; filter_idx += channels) {
          //printf("Filter_IDX: %i; Filter_IDY:  %i; Input_IDX: %i;  Input_IDY: %i; Output_IDX: %i;  Output_IDY: %i; \n", filter_idx, filter_idy, input_idx, input_idy, output_idx, output_idy);
          //printf("Input Values: %i %i; Filter Values: %i  %i; \n", input[input_idx + input_idy*in_width*channels], input[1 + input_idx + input_idy*in_width*channels], filter[filter_idx + filter_idy*kernel_width*channels], filter[1 + filter_idx + filter_idy*kernel_width*channels]);

          if (output_idx == 0 && filter_idx == 0 && pad_left_width == 1) {
            printf("pad left buffer zero. output_y: %i output_x: %i filter_y: %i filter_x: %i \n",
                   output_idy, output_idx, filter_idy, filter_idx);
            input_idx += channels;
            continue;
          }
          // else if(output_idx == out_width*channels-channels && filter_idx == kernel_width*channels-channels && pad_right_width == 1){
          //   printf("pad right buffer zero. output_y: %i output_x: %i filter_y: %i filter_x: %i \n", output_idy, output_idx, filter_idy, filter_idx);
          //   input_idx += channels;
          //   continue;
          // }
          printf("Input Values: %i ; Filter Values: %i  Input_IDX: %i  Input_IDY: %i \n",
                 input[input_idx + input_idy * in_width * channels],
                 filter[filter_idx + filter_idy * kernel_width * channels],
                 input_idx, input_idy);

          asm volatile("vmca va1, %0"
                       :
                       : "r"(input_ptr + input_idx + input_idy * in_width * channels));
          asm volatile("vmca va2, %0"
                       :
                       : "r"(filter + filter_idx + filter_idy * kernel_width * channels));
          //asm volatile ("vmcs vs0, %0" : : "r" (channels));
          asm volatile("la t0, vtest4"
                       :
                       :
                       : "t0");
          asm volatile("lw t1, 0(t0)");
          asm volatile("vf 0(t0)");
          //printf("Output values: %i ", output2[output_idx + output_idy*out_width*channels]);
          input_idx += channels;
        }
        input_idy += 1;
      }
      //for(int i = 0; i < 10; i++) { printf("%i ", temp_output[i]); } printf("\n");
      asm volatile("vmca va0, %0"
                   :
                   : "r"(temp_output));  //output
      asm volatile("vmca va1, %0"
                   :
                   : "r"(output + output_idx + output_idy * out_width * channels));  //output
      asm volatile("vmcs vs1, %0"
                   :
                   : "r"(rounded_divisor));  //divisor
      asm volatile("la t0, vtest5"
                   :
                   :
                   : "t0");
      asm volatile("lw t1, 0(t0)");
      asm volatile("vf 0(t0)");
      //printf("\n");
    }
  }
  asm volatile("fence");

  //for loop for as many channels
  // consumed = setvlen(out_width*out_height);
  // printf("\nConsumed: %li\n", consumed);
  // asm volatile ("vmca va0, %0" : : "r" (temp_output));
  // asm volatile ("vmca va1, %0" : : "r" (temp_output+1));
  // asm volatile ("vmca va2, %0" : : "r" (output));
  // asm volatile ("vmca va3, %0" : : "r" (channels));
  // asm volatile ("la t0, accumulate_channels" : : : "t0");
  // asm volatile ("lw t1, 0(t0)");
  // asm volatile ("vf 0(t0)");

  // printf("\n");
  // printf("output from hdwk\n");
  // for (size_t m = 0; m < out_height; m++) {
  //   for (size_t k = 0; k < out_width*channels; k++) {
  //       printf("%i ", output2[m * channels * out_width + k]);
  //   }
  //   printf("\n");
  // }

  // printf("\n");
  // printf("output from hdwk\n");
  // for (size_t m = 0; m < out_height; m++) {
  //   for (size_t k = 0; k < out_width*channels; k++) {
  //       printf("%i ", output[m * channels * out_width + k]);
  //   }
  //   printf("\n");
  // }

  // printf("\n");
  // printf("output from hdwk\n");
  // for(size_t c = 0; c < channels; c++){
  //   printf("Channel %i\n",c);
  //   for (size_t y = c; y < out_height * channels; y+=channels) {
  //     for (size_t x = c; x < out_width * channels; x+=channels) {
  //         printf("%i ", temp_output[x + out_width * y]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // printf("\n");
  // printf("output from hdwk\n");
  // for(size_t c = 0; c < channels; c++){
  //   printf("Channel %i\n",c);
  //   for (size_t y = c; y < out_height * channels; y+=channels) {
  //     for (size_t x = c; x < out_width * channels; x+=channels) {
  //         printf("%i ", output[x + out_width * y]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  printf("\nFinished Hwacha Depthwise Convolution! \n");
}
#pragma GCC diagnostic pop