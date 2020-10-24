#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wformat"
#pragma GCC diagnostic ignored "-Wtype-limits"
class MlasHwachaDWCTest : public MlasTestBase {
 protected:
  void
  Test(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth) {
    printf("Testing: Image Dimensions: %i, %i; Filter Dimensions: %i, %i; # of Filters: %i; \n", InputHeight, InputWidth, KernelHeight, KernelWidth, FilterCount);
    int64_t OutputHeight64 =
        ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
         (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) /
            int64_t(StrideHeight) +
        1;
    int64_t OutputWidth64 =
        ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
         (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) /
            int64_t(StrideWidth) +
        1;

    if (OutputHeight64 <= 0 || OutputWidth64 <= 0) {
      return;
    }

    size_t OutputHeight = size_t(OutputHeight64);
    size_t OutputWidth = size_t(OutputWidth64);

    size_t InputSize = InputHeight * InputWidth;
    size_t KernelSize = KernelHeight * KernelWidth;
    size_t OutputSize = OutputHeight * OutputWidth;

    size_t InputElements = BatchCount * GroupCount * InputChannels * InputSize;
    size_t FilterElements = GroupCount * FilterCount * KernelSize;  // Depthwise InputChannels * KernelSize;
    size_t BiasElements = GroupCount * FilterCount;
    size_t OutputElements = BatchCount * GroupCount * FilterCount * OutputSize;

    //const int8_t* A = BufferA.GetBuffer(K * M);

    const int8_t* Input = BufferInput.GetBuffer(InputElements);
    const int8_t* Filter = BufferFilter.GetBuffer(FilterElements);
    const int32_t* Bias = BufferBias.GetBuffer(BiasElements);
    int8_t* Output = BufferOutput.GetBuffer(OutputElements);
    int8_t* OutputReference = BufferOutputReference.GetBuffer(OutputElements);

    HwachaDepthWiseConv(BatchCount,
                        GroupCount,
                        InputChannels,
                        InputHeight, InputWidth,
                        FilterCount,
                        KernelHeight, KernelWidth,
                        PaddingLeftHeight, PaddingLeftWidth,
                        PaddingRightHeight, PaddingRightWidth,
                        DilationHeight, DilationWidth,
                        StrideHeight, StrideWidth,
                        OutputHeight, OutputWidth,
                        Input,
                        Filter,
                        Bias,
                        Output,
                        1);

    ReferenceConv2D(BatchCount,
                    GroupCount,
                    InputChannels,
                    InputHeight, InputWidth,
                    FilterCount,
                    KernelHeight, KernelWidth,
                    PaddingLeftHeight, PaddingLeftWidth,
                    DilationHeight, DilationWidth,
                    StrideHeight, StrideWidth,
                    OutputHeight, OutputWidth,
                    Input,
                    Filter,
                    Bias,
                    OutputReference);
    // printf("\n");
    // printf("input\n");
    // for(size_t c = 0; c < InputChannels; c++){
    //     printf("Channel %i\n",c);
    //     for (size_t y = 0; y < InputHeight; y+=1) {
    //     for (size_t x = c; x < InputWidth * InputChannels; x+=InputChannels) {
    //         printf("%i ", Input[x + InputWidth * InputChannels * y]);
    //     }
    //     printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("filter\n");
    // for (size_t k = 0; k < KernelHeight; k+=1) {
    //     for (size_t l = 0; l < KernelWidth*FilterCount; l+=1) {
    //         printf("%i ", Filter[k * InputChannels * KernelWidth + l]);
    //     }
    //     printf("\n");
    // }

    // printf("\n");
    // printf("Filters:\n");

    //   for (size_t f = 0; f < FilterCount;f++){
    //     printf("filter %i\n", f);
    //     for (size_t k = 0; k < KernelHeight; k+=1) {
    //         for (size_t l = f; l < KernelWidth*FilterCount; l+=FilterCount) {
    //             printf("%i ", Filter[k * FilterCount * KernelWidth + l]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //   }

    // printf("\n");
    // printf("Reference Output:\n");
    // //printf("output\n");
    // for (size_t k = 0; k < OutputHeight; k+=1) {
    //     for (size_t l = 0; l < OutputWidth*FilterCount; l+=1) {
    //         printf("%p:%02x \t",  &OutputReference[k * FilterCount * OutputWidth + l], OutputReference[k * OutputWidth + l]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("Actual Output:\n");
    // //printf("output\n");
    // for (size_t k = 0; k < OutputHeight; k+=1) {
    //     for (size_t l = 0; l < OutputWidth*FilterCount; l+=1) {
    //         printf("%p:%02x \t", &Output[k * FilterCount * OutputWidth + l], Output[k * OutputWidth + l]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // for(size_t c = 0; c < InputChannels; c++){
    //     printf("Channel %i\n",c);
    //     for (size_t y = 0; y < OutputHeight; y+=1) {
    //         for (size_t x = c; x < OutputWidth * InputChannels; x+=InputChannels) {
    //             printf("%i ", OutputReference[x + OutputWidth * InputChannels * y]);
    //         }
    //     printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    if (memcmp(Output, OutputReference, sizeof(int8_t) * OutputElements) != 0) {
      printf("mismatch: batch=%zd,group=%zd,input(%zd,%zd,%zd),filter=%zd,kernel(%zd,%zd)!!!\n",
             BatchCount, GroupCount, InputChannels, InputHeight, InputWidth, FilterCount,
             KernelHeight, KernelWidth);
    } else {
      printf("Output Matches!\n");
    }
  }

  virtual void
  MlasConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output) {
    int64_t InputShape[] = {int64_t(InputHeight), int64_t(InputWidth)};
    int64_t KernelShape[] = {int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t DilationShape[] = {int64_t(DilationHeight), int64_t(DilationWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};
    int64_t StrideShape[] = {int64_t(StrideHeight), int64_t(StrideWidth)};
    int64_t OutputShape[] = {int64_t(OutputHeight), int64_t(OutputWidth)};

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize;

    MlasConvPrepare(&Parameters,
                    2,
                    BatchCount,
                    GroupCount,
                    InputChannels,
                    InputShape,
                    KernelShape,
                    DilationShape,
                    Padding,
                    StrideShape,
                    OutputShape,
                    FilterCount,
                    &Activation,
                    &WorkingBufferSize,
                    nullptr);

    MlasConv(&Parameters,
             Input,
             Filter,
             Bias,
             BufferWorking.GetBuffer(WorkingBufferSize),
             Output,
             nullptr);
  }

  void
  ReferenceConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const int8_t* Input,
      const int8_t* Filter,
      const int32_t* Bias,
      int8_t* Output) {
    size_t InputSize = InputHeight * InputWidth;
    size_t OutputSize = OutputHeight * OutputWidth;
    size_t KernelSize = KernelHeight * KernelWidth;
    size_t GroupSize = FilterCount * InputSize;

    size_t K = InputChannels * KernelSize;
    size_t Im2ColElements = OutputSize * K;

    for (size_t batch = 0; batch < BatchCount; batch++) {
      for (size_t group = 0; group < GroupCount; group++) {
        for (size_t channel = 0; channel < InputChannels; channel++) {
          for (size_t out_row = 0; out_row < OutputHeight; out_row++) {
            for (size_t out_col = channel; out_col < OutputWidth * InputChannels; out_col += InputChannels) {
              size_t in_row = out_row * StrideHeight - PaddingLeftHeight;

              int32_t result = 0;
              //if (params->bias) {
              //result = Bias[group];
              //}

              for (size_t kernel_row = 0; kernel_row < KernelHeight; kernel_row++) {
                size_t in_col = out_col * StrideWidth - PaddingLeftWidth;

                for (size_t kernel_col = channel; kernel_col < KernelWidth * FilterCount; kernel_col += FilterCount) {
                  if (in_row >= 0 && in_row < InputHeight && in_col >= 0 && in_col < InputWidth * InputChannels) {
                    result += Input[group * GroupSize + in_row * InputWidth * InputChannels + in_col] * Filter[group * GroupSize + kernel_row * KernelWidth * FilterCount + kernel_col];
                  }
                  //printf("Filter_IDX: %i; Filter_IDY:  %i; Input_IDX: %i;  Input_IDY: %i; Input Value: %i; Filter Value: %i; Result: %i; \n", kernel_col, kernel_row, in_col, in_row, Input[group*GroupSize + in_row*InputWidth*InputChannels + in_col], Filter[group*GroupSize + kernel_row*KernelWidth*FilterCount + kernel_col], result);

                  in_col += InputChannels;
                }

                in_row++;
              }

              /*
                            acc_t abs = result >= 0 ? result : -result;
                            int divisor = 1 << params->output_scale;
                            acc_t shifted = (abs + divisor/2) >> params->output_scale;
                            if (result < 0) {
                                shifted = -shifted;
                            }
                            */

              if (result < 0) {
                result = 0;
              }

              //int32_t shifted = ROUNDING_RIGHT_SHIFT(result, params->output_scale);

              if (result > 127) {
                result = 127;
              }
              //printf("Output_IDX: %i; Output_IDY: %i; Result Value: %i\n", out_col, out_row, result);
              Output[group * GroupSize + out_row * OutputWidth * InputChannels + out_col] = result;
              //printf("\n");
            }
          }
        }
      }
    }
  }

  MatrixGuardBuffer<int8_t> BufferInput;
  MatrixGuardBuffer<int8_t> BufferFilter;
  MatrixGuardBuffer<int32_t> BufferBias;
  MatrixGuardBuffer<int8_t> BufferOutput;
  MatrixGuardBuffer<int8_t> BufferOutputReference;
  MatrixGuardBuffer<float> BufferWorking;
  MatrixGuardBuffer<int8_t> BufferIm2Col;

 public:
  void
  ExecuteLong(
      void) override {
    // N.B. InputChannels must be a multiple of 4 if the count is greater
    // than the block size.
    // static const unsigned cis[] = { 32, 20, 5, 1 };
    // static const unsigned cos[] = { 64, 15, 1 };
    // static const unsigned is[] = { 27, 11, 5, 1 };

    // Depthwise convolutions.
    printf("Avi's Depthwise Tests\n");
    Test(1, 1, 1, 5, 5, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1);  //Input Channels 2; Filter Count 2;
    Test(1, 1, 1, 5, 5, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);  //Input Channels 2; Filter Count 2;
    Test(1, 1, 1, 5, 5, 1, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1);  //Input Channels 2; Filter Count 2;

    Test(1, 1, 2, 5, 5, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1);  //Input Channels 2; Filter Count 2;
    Test(1, 1, 3, 5, 5, 3, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1);  //Input Channels 2; Filter Count 2;

    //for (unsigned i = 16; i < 256; i <<= 1) {

    //Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);

    // Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
    // Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
    // Test(1, i, 1, 28, 28, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
    // Test(1, i, 1, 28, 28, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // Test(1, i, 1, 28, 28, 1, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // Test(12, i, 1, 11, 11, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    //}

    // Test varying FilterCounts.
    // for (unsigned i = 1; i < 128; i++) {
    //     Test(1, 1, 3, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    //     Test(1, 1, 16, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    //     Test(1, 1, 16, 34, 34, i, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // }

    // for (unsigned i = 1; i <= 32; i++) {
    //     Test(4, 18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
    //     Test(4, 18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
    //     Test(4, 18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
    // }

    // for (unsigned b = 1; b < 64; b++) {
    //     Test(b, 1, 64, 11, 11, 128, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // }

    // for (unsigned ic = 0; ic < _countof(cis); ic++) {
    //     for (unsigned ih = 0; ih < _countof(is); ih++) {
    //         for (unsigned iw = 0; iw < _countof(is); iw++) {
    //             fprintf(stderr, "Handling %ux%ux%u\n", cis[ic], is[ih], is[iw]);
    //             for (unsigned fc = 0; fc < _countof(cos); fc++) {
    //                 for (unsigned kh = 1; kh <= 5; kh++) {
    //                     if (kh == 4) continue;
    //                     for (unsigned kw = 1; kw <= 5; kw++) {
    //                         if (kw == 4) continue;
    //                         for (unsigned p0 = 0; p0 <= 3; p0++) {
    //                             for (unsigned p1 = 0; p1 <= 3; p1++) {
    //                                 for (unsigned p2 = 0; p2 <= 3; p2++) {
    //                                     for (unsigned p3 = 0; p3 <= 3; p3++) {
    //                                         for (unsigned dh = 1; dh <= 2; dh++) {
    //                                             for (unsigned dw = 1; dw <= 2; dw++) {
    //                                                 for (unsigned sh = 1; sh <= 2; sh++) {
    //                                                     for (unsigned sw = 1; sw <= 2; sw++) {
    //                                                         Test(1, 1, cis[ic], is[ih], is[iw], cos[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
    //                                                     }
    //                                                 }
    //                                             }
    //                                         }
    //                                     }
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
  }

  // public:
  //     void ExecuteShort(void) override
  //     {
  //       size_t K = 4;
  //       size_t M = 4;
  //       //int N = 4;

  //       printf("avi's test\n");
  //       printf("Testing...\n");
  //       const int8_t* A = BufferA.GetBuffer(K * M);
  //       //const int8_t* B = BufferB.GetBuffer(N * K);
  //       //const int32_t* Bias = BufferBias.GetBuffer(N * M);
  //       //int8_t* C = BufferC.GetBuffer(N * M);
  //       //int8_t* CReference = BufferCReference.GetBuffer(N * M);

  //       printf("A matrix:\n");
  //       for (size_t m = 0; m < M; m++) {
  //           for (size_t k = 0; k < K; k++) {
  //               printf("%d ", A[m * K + k]);
  //           }
  //           printf("\n");
  //       }

  //         // Should match precisely for exact multiples of systolic size
  //         // printf("Testing exact dimensions with no divisor\n");
  //         // Test(16, 16, 16, 1, 0);
  //         // Test(1*16, 2*16, 3*16, 1, 0);
  //         // Test(16, 16, 16, 1, 0, /*relu= */ true);
  //         // Test(1*16, 2*16, 3*16, 1, 0, /*relu= */ true);
  //         //
  //         // // Should match preicsely for exact multiples with divisor (right shift)
  //         // printf("Testing exact dimensions with divisor\n");
  //         // Test(16, 16, 16, 4, 0);
  //         // Test(1*16, 2*16, 3*16, 4, 0);
  //         // Test(16, 16, 16, 4, 0, /*relu= */ true);
  //         // Test(1*16, 2*16, 3*16, 4, 0, /*relu= */ true);
  //         //
  //         // printf("Testing non-exact dimensions with divisor\n");
  //         // Test(3, 5, 7, 2, 0);
  //         // Test(89, 79, 83, 4, 0);
  //         // Test(18, 45, 337, 8, 0, /*relu= */ true);
  //         // Test(1697, 2029, 1319, 16, 0, /*relu= */ true);

  //         //HwachaDepthWiseConv(24);
  //     }

  //     void ExecuteLong(void) override
  //     {
  //     }

  //     //MlasHwachaDWCTest();
};
#pragma GCC diagnostic pop