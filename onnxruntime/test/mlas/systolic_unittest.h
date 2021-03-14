#define ABS(x) (x < 0 ? -x : x)
#ifdef SYSTOLIC_INT8
#define ACC_SCALE(x, scale) \
  ({float y = nearbyint((x) * (scale)); y > INT_MAX ? INT_MAX : (y < INT_MIN ? INT_MIN : (acc_t)y); })
#define MVIN_SCALE(x, scale) \
  (scale == 1.0 ? x : ({float y = nearbyint((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y); }))
#define ELEM_T_MAX SCHAR_MAX
#define ELEM_T_MIN SCHAR_MIN
#define FMT "%d "
#else
#define ACC_SCALE(x, scale) \
  ({float y = (x) * (scale); (acc_t) y; })
#define MVIN_SCALE(x, scale) \
  ({float y = (x) * (scale); (elem_t) y; })
#define ELEM_T_MAX 3.4028235E38
#define ELEM_T_MIN -3.4028235E38
#define FMT "%f "
#endif

template <typename elem_t, typename acc_t>
class MlasSystolicMatmulTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<elem_t> BufferA;
  MatrixGuardBuffer<elem_t> BufferB;
  MatrixGuardBuffer<elem_t> BufferC;
  MatrixGuardBuffer<elem_t> BufferCReference;
  MatrixGuardBuffer<acc_t> BufferBias;
  char acceleration_type_;

  inline elem_t saturate(acc_t num, float scale, bool relu) {
    num = ACC_SCALE(num, scale);
    // Clip result
    num = num > ELEM_T_MAX ? ELEM_T_MAX : (num < ELEM_T_MIN ? ELEM_T_MIN : num);
    if (relu) {
      num = num < 0 ? 0 : num;
    }
    return num;
  }

  void mymatmul(int dimI, int dimJ, int dimK, const elem_t* in1, const elem_t* in2, elem_t* out,
                float scale, bool relu, const acc_t* bias = nullptr) {
    for (int i = 0; i < dimI; i++) {
      for (int j = 0; j < dimJ; j++) {
        acc_t res = 0;
        for (int k = 0; k < dimK; k++) {
          res += in1[i * dimK + k] * in2[k * dimJ + j];
        }
        out[i * dimJ + j] = saturate(res + (bias != nullptr ? bias[i * dimJ + j] : 0), scale, relu);
      }
    }
  }

  void NaiveCPUMultiply(int dimI, int dimJ, int dimK, const elem_t* in1, const elem_t* in2, elem_t* out,
                        float scale, bool relu, const acc_t* bias = nullptr) {
    return mymatmul(dimI, dimJ, dimK, in1, in2, out, scale, relu, bias);
  }

  void Test(size_t M, size_t N, size_t K, float scale, int tolerance, bool relu = false) {
    printf("Testing... %zu %zu %zu\n", M, N, K);
    const elem_t* A = BufferA.GetBuffer(K * M);
    const elem_t* B = BufferB.GetBuffer(N * K);
    const acc_t* Bias = BufferBias.GetBuffer(N * M);
    elem_t* C = BufferC.GetBuffer(N * M);
    elem_t* CReference = BufferCReference.GetBuffer(N * M);

    std::fill_n(C, M * N, -1);
    std::fill_n(CReference, M * N, -1);

    SystolicMultiply(acceleration_type_, relu, M, N, K, A, B, C, scale, Bias);
    NaiveCPUMultiply(M, N, K, A, B, CReference, scale, relu, Bias);

    for (size_t f = 0; f < M * N; f++) {
      if (ABS(C[f] - CReference[f]) > tolerance) {
        printf("A matrix:\n");
        for (size_t m = 0; m < M; m++) {
          for (size_t k = 0; k < K; k++) {
            printf(FMT, A[m * K + k]);
          }
          printf("\n");
        }
        printf("B matrix:\n");
        for (size_t k = 0; k < K; k++) {
          for (size_t n = 0; n < N; n++) {
            printf(FMT, B[k * N + n]);
          }
          printf("\n");
        }
        printf("Bias matrix:\n");
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            printf(FMT, Bias[m * N + n]);
          }
          printf("\n");
        }
        printf("C matrix:\n");
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            printf(FMT, C[m * N + n]);
          }
          printf("\n");
        }
        printf("C_ref matrix:\n");
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            printf(FMT, CReference[m * N + n]);
          }
          printf("\n");
        }

        printf("mismatch M=%zd, N=%zd, K=%zd, scale=%f, relu=%d. Diff=" FMT "!\n", M, N, K, scale, relu, ABS(C[f] - CReference[f]));
        return;
      }
    }
  }

 public:
  void ExecuteShort(void) override {
    // Should match precisely for exact multiples of systolic size
    printf("Testing exact dimensions with no divisor\n");
    Test(16, 16, 16, 1, /*tolerance =*/0);
    Test(1 * 16, 2 * 16, 3 * 16, 1, /*tolerance =*/0);
    Test(16, 16, 16, 1, 0, /*relu= */ true);
    Test(1 * 16, 2 * 16, 3 * 16, 1, /*tolerance =*/0, /*relu= */ true);

    // Should match preicsely for exact multiples with divisor (right shift)
    printf("Testing exact dimensions with divisor\n");
    Test(16, 16, 16, 0.25, /*tolerance =*/0);
    Test(1 * 16, 2 * 16, 3 * 16, 2, /*tolerance =*/0);
    Test(16, 16, 16, 0.0625, /*tolerance =*/0, /*relu= */ true);
    Test(1 * 16, 2 * 16, 3 * 16, 0.0625, /*tolerance =*/0, /*relu= */ true);

    printf("Testing non-exact dimensions with divisor\n");
    Test(3, 5, 7, 0.25, /*tolerance= */ 0);
    Test(89, 79, 83, 0.0625, /*tolerance= */ 0);
    Test(18, 45, 337, 0.0039, /*tolerance= */ 0, /*relu= */ true);
    Test(1697, 2029, 1319, 0.00001525, /*tolerance =*/0, /*relu= */ true);
  }

  void ExecuteLong(void) override {
  }

  MlasSystolicMatmulTest(char acceleration_type) : acceleration_type_(acceleration_type) {}
};

#ifdef SYSTOLIC_INT8

template <typename elem_t, typename acc_t>
class MlasSystolicAddTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<elem_t> BufferA;
  MatrixGuardBuffer<elem_t> BufferB;
  MatrixGuardBuffer<elem_t> BufferC;
  MatrixGuardBuffer<elem_t> BufferCReference;
  char acceleration_type_;

  inline elem_t saturate(acc_t num, float scale, bool relu) {
    num = ACC_SCALE(num, scale);
    // Clip result
    num = num > ELEM_T_MAX ? ELEM_T_MAX : (num < ELEM_T_MIN ? ELEM_T_MIN : num);
    if (relu) {
      num = num < 0 ? 0 : num;
    }
    return num;
  }

  void NaiveCPUAdd(bool relu, const int8_t* in1, float in1_scale, const int8_t* in2, float in2_scale,
                   int8_t* out, float out_scale, int dim) {
    for (int i = 0; i < dim; i++) {
      int32_t tmp1 = (int)MVIN_SCALE(*in1, in1_scale / out_scale);
      int32_t tmp2 = (int)MVIN_SCALE(*in2, in2_scale / out_scale);
      *out = saturate(tmp1 + tmp2, /*scale= */ 1, relu);

      out++;
      in1++;
      in2++;
    }
  }

  void Test(size_t M, float scaleA, float scaleB, float scaleOut, int tolerance, bool relu = false) {
    printf("Testing... %zu\n", M);
    const elem_t* A = BufferA.GetBuffer(M);
    const elem_t* B = BufferB.GetBuffer(M);

    elem_t* C = BufferC.GetBuffer(M);
    elem_t* CReference = BufferCReference.GetBuffer(M);

    SystolicAdd(acceleration_type_, relu, A, scaleA, B, scaleB, C, scaleOut, M);
    NaiveCPUAdd(relu, A, scaleA, B, scaleB, CReference, scaleOut, M);

    for (size_t f = 0; f < M; f++) {
      if (ABS(C[f] - CReference[f]) > tolerance) {
        printf("mismatch M=%zd, scaleA=%f, scaleB=%f, scaleOut=%f relu=%d. Diff=" FMT "!\n", M, scaleA, scaleB, scaleOut, relu, ABS(C[f] - CReference[f]));
        return;
      }
    }
  }

 public:
  void ExecuteShort(void) override {
    Test(1, 0.01, 0.05, 0.01, /*tolerance =*/0);
    Test(15, 0.01, 0.05, 0.01, /*tolerance =*/0);
    Test(3 * 16, 0.01, 0.05, 0.01, /*tolerance =*/0);
    Test(2 * 16, 0.5, 0.5, 0.25, /*tolerance =*/0);
    Test(2 * 16 + 1, 0.5, 0.5, 0.25, /*tolerance =*/0);
    Test(2 * 16 + 15, 0.3, 0.5, 0.25, /*tolerance =*/0);
    Test(1697, 0.17, 0.01, 0.5, /*tolerance =*/0, /*relu= */ true);
    Test(2029, 0.113, 0.01, 3, /*tolerance =*/0, /*relu= */ true);
  }

  void ExecuteLong(void) override {
  }

  MlasSystolicAddTest(char acceleration_type) : acceleration_type_(acceleration_type) {}
};

template <typename elem_t, typename acc_t>
class MlasSystolicConvTest : public MlasTestBase {
 private:
  char acceleration_type_;
  MatrixGuardBuffer<elem_t> BufferInput;
  MatrixGuardBuffer<elem_t> BufferFilter;
  MatrixGuardBuffer<acc_t> BufferBias;
  MatrixGuardBuffer<elem_t> BufferOutput;
  MatrixGuardBuffer<elem_t> BufferOutputReference;

  inline elem_t saturate(acc_t num, float scale, bool relu) {
    num = ACC_SCALE(num, scale);
    // Clip result
    num = num > ELEM_T_MAX ? ELEM_T_MAX : (num < ELEM_T_MIN ? ELEM_T_MIN : num);
    if (relu) {
      num = num < 0 ? 0 : num;
    }
    return num;
  }

  inline std::vector<elem_t> HWIOtoOIHWconvert(const elem_t* w_vals, const std::vector<size_t>& w_shape) {
    int H = w_shape[0];
    int W = w_shape[1];
    int IC = w_shape[2];
    int OC = w_shape[3];
    std::vector<elem_t> w_vals_copy = std::vector<elem_t>(H * W * IC * OC);

    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int ic = 0; ic < IC; ic++) {
          for (int oc = 0; oc < OC; oc++) {
            w_vals_copy[oc * H * W * IC + ic * H * W + h * W + w] =
                w_vals[h * W * IC * OC + w * IC * OC + ic * OC + oc];
          }
        }
      }
    }
    return w_vals_copy;
  }

  inline std::vector<elem_t> NHWCtoNCHW(const elem_t* vals, const std::vector<size_t>& shape) {
    int N = shape[0];
    int H = shape[1];
    int W = shape[2];
    int C = shape[3];
    std::vector<elem_t> vals_copy = std::vector<elem_t>(N * H * W * C);

    for (int n = 0; n < N; n++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          for (int c = 0; c < C; c++) {
            vals_copy[((n * C + c) * H + h) * W + w] = vals[((n * H + h) * W + w) * C + c];
          }
        }
      }
    }

    return vals_copy;
  }

  void Test(
      size_t BatchCount,
      size_t InputChannels,
      size_t InputDim,
      size_t OutputChannels,
      size_t KernelDim,
      size_t Padding,
      size_t Stride,
      bool relu,
      float output_scale) {
    printf("Testing... N=%zu, C=%zu, H/W=%zu, Oc=%zu Kw/Kh=%zu\n", BatchCount, InputChannels, InputDim, OutputChannels, KernelDim);
    int64_t OutputDim64 =
        1 + (int64_t(InputDim) + int64_t(Padding) + int64_t(Padding) - int64_t(KernelDim)) / int64_t(Stride);

    size_t OutputDim = size_t(OutputDim64);

    size_t InputSize = InputDim * InputDim;
    size_t KernelSize = KernelDim * KernelDim;
    size_t OutputSize = OutputDim * OutputDim;

    size_t InputElements = BatchCount * InputChannels * InputSize;
    size_t FilterElements = OutputChannels * InputChannels * KernelSize;
    size_t BiasElements = OutputChannels;
    size_t OutputElements = BatchCount * OutputChannels * OutputSize;

    const elem_t* Input = BufferInput.GetBuffer(InputElements);
    const elem_t* Filter = BufferFilter.GetBuffer(FilterElements);
    const acc_t* Bias = BufferBias.GetBuffer(BiasElements);
    elem_t* Output = BufferOutput.GetBuffer(OutputElements);
    elem_t* OutputReference = BufferOutputReference.GetBuffer(OutputElements);

    reference_conv(BatchCount, InputDim, InputChannels, OutputChannels, OutputDim, Stride, Padding, KernelDim,
                   Input, Filter, Bias, OutputReference, relu, output_scale);

    SystolicConv(acceleration_type_, BatchCount, InputDim, InputChannels, OutputChannels, OutputDim, Stride, Padding, KernelDim,
                 Input, Filter, Bias, Output, relu, output_scale);

    for (size_t f = 0; f < OutputElements; f++) {
      if (ABS(Output[f] - OutputReference[f]) > 0) {
        printf("mismatch Diff=\n" FMT "!\n", ABS(Output[f] - OutputReference[f]));
        // std::vector<elem_t> nchw_input = NHWCtoNCHW(Input, {BatchCount, InputDim, InputDim, InputChannels});
        // std::vector<elem_t> oihw = HWIOtoOIHWconvert(Filter, {KernelDim, KernelDim, InputChannels, OutputChannels});

        // std::vector<elem_t> nchw_output = NHWCtoNCHW(Output, {BatchCount, OutputDim, OutputDim, OutputChannels});
        // std::vector<elem_t> nchw_output_ref = NHWCtoNCHW(OutputReference, {BatchCount, OutputDim, OutputDim, OutputChannels});
        // printf("Input matrix:\n");
        // for (size_t m = 0; m < InputDim; m++) {
        //   for (size_t k = 0; k < InputDim; k++) {
        //     printf(FMT, nchw_input[m * InputDim + k]);
        //   }
        //   printf("\n");
        // }

        // printf("Filter matrix:\n");
        // for (size_t m = 0; m < KernelDim; m++) {
        //   for (size_t k = 0; k < KernelDim; k++) {
        //     printf(FMT, oihw[m * KernelDim + k]);
        //   }
        //   printf("\n");
        // }

        // printf("Bias:\n");
        // for (size_t m = 0; m < BiasElements; m++) {
        //   printf(FMT, Bias[m]);
        // }
        // printf("\n");

        // printf("Output Reference matrix:\n");
        // for (size_t m = 0; m < OutputDim; m++) {
        //   for (size_t k = 0; k < OutputDim; k++) {
        //     printf(FMT, nchw_output_ref[m * OutputDim + k]);
        //   }
        //   printf("\n");
        // }

        // printf("Output matrix:\n");
        // for (size_t m = 0; m < OutputDim; m++) {
        //   for (size_t k = 0; k < OutputDim; k++) {
        //     printf(FMT, nchw_output[m * OutputDim + k]);
        //   }
        //   printf("\n");
        // }

        return;
      }
    }
  }

  void reference_conv(
      int batch_size, int in_dim, int in_channels,
      int out_channels, int out_dim,
      int stride, int padding, int kernel_dim,
      const elem_t* input,
      const elem_t* weights,
      const acc_t* bias,
      elem_t* output,
      bool relu, float output_scale) {
    bool no_bias = bias == NULL;

    for (int b = 0; b < batch_size; b++) {
      for (int orow = 0; orow < out_dim; orow++) {
        for (int ocol = 0; ocol < out_dim; ocol++) {
          //printf("New output value\n");
          for (int och = 0; och < out_channels; och++) {
            acc_t opixel = no_bias ? 0 : bias[och];
            //printf("New output channel\n");
            for (int krow = 0; krow < kernel_dim; krow++) {
              const int irow = orow * stride + krow - padding;

              for (int kcol = 0; kcol < kernel_dim; kcol++) {
                const int icol = ocol * stride + kcol - padding;

                for (int kch = 0; kch < in_channels; kch++) {
                  elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ? 0 : *(input + (b * in_dim * in_dim + irow * in_dim + icol) * in_channels + kch);

                  elem_t weight = *(weights + (krow * kernel_dim * in_channels + kcol * in_channels + kch) * out_channels + och);
                  //printf("Multiplying weight and ipixel %d %d\n", weight, ipixel);
                  opixel += weight * ipixel;
                }
              }
            }
            //printf("Value before saturate %d\n", opixel);
            *(output + (b * out_dim * out_dim + orow * out_dim + ocol) * out_channels + och) =
                saturate(opixel, output_scale, relu);
          }
        }
      }
    }
  }

 public:
  void
  ExecuteShort(void) override {
    Test(/*BatchCount=*/1, /*InputChannels= */ 1, /*InputDim= */ 5,
         /*OutputChannels= */ 1, /*KernelDim= */ 2, /*Padding= */ 0,
         /*Stride= */ 1, /*relu= */ 0, /*output_scale= */ 0.1);

    Test(/*BatchCount=*/1, /*InputChannels= */ 1, /*InputDim= */ 10,
         /*OutputChannels= */ 1, /*KernelDim= */ 2, /*Padding= */ 0,
         /*Stride= */ 1, /*relu= */ 0, /*output_scale= */ 0.1);

    Test(/*BatchCount=*/1, /*InputChannels= */ 3, /*InputDim= */ 256,
         /*OutputChannels= */ 1, /*KernelDim= */ 2, /*Padding= */ 0,
         /*Stride= */ 1, /*relu= */ 0, /*output_scale= */ 0.001);

    Test(/*BatchCount=*/1, /*InputChannels= */ 3, /*InputDim= */ 224,
         /*OutputChannels= */ 64, /*KernelDim= */ 7, /*Padding= */ 3,
         /*Stride= */ 2, /*relu= */ 0, /*output_scale= */ 0.001);

    Test(/*BatchCount=*/1, /*InputChannels= */ 10, /*InputDim= */ 500,
         /*OutputChannels= */ 1, /*KernelDim= */ 10, /*Padding= */ 0,
         /*Stride= */ 1, /*relu= */ 0, /*output_scale= */ 0.001);

    Test(/*BatchCount=*/2, /*InputChannels= */ 10, /*InputDim= */ 300,
         /*OutputChannels= */ 3, /*KernelDim= */ 200, /*Padding= */ 2,
         /*Stride= */ 5, /*relu= */ true, /*output_scale= */ 0.001);
  }

  void
  ExecuteLong(void) override {
  }

  MlasSystolicConvTest(char acceleration_type) : acceleration_type_(acceleration_type) {}
};

#endif

#ifdef SYSTOLIC_FP32
template <typename T>
class MlasSystolicGemmTest : public MlasTestBase {
 private:
  void
  Test(
      size_t M,
      size_t N,
      size_t K,
      float alpha,
      float beta) {
    const T* A = BufferA.GetBuffer(K * M);
    const T* B = BufferB.GetBuffer(N * K);
    T* C = BufferC.GetBuffer(N * M);
    T* CReference = BufferCReference.GetBuffer(N * M);

    Test(false, false, M, N, K, alpha, A, B, beta, C, CReference);
    Test(false, true, M, N, K, alpha, A, B, beta, C, CReference);
    Test(true, false, M, N, K, alpha, A, B, beta, C, CReference);
  }

  void
  Test(
      size_t M,
      size_t N,
      size_t K,
      int lda,
      int ldb,
      int ldc,
      float alpha,
      float beta) {
    // We make the buffer bigger to ensure the strided is striding properly    
    const T* A = BufferA.GetBuffer(std::max(K, M)*std::max(K, M));
    const T* B = BufferB.GetBuffer(std::max(N, K)*std::max(N, K));
    T* C = BufferC.GetBuffer(std::max(N, M)*std::max(N, M));
    T* CReference = BufferCReference.GetBuffer(std::max(N, M)*std::max(N, M));

    Test(false, false, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, CReference);
    Test(false, true, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, CReference);
    Test(true, false, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, CReference);
  }

  void
  Test(
      bool TransA,
      bool TransB,
      size_t M,
      size_t N,
      size_t K,
      float alpha,
      const T* A,
      const T* B,
      float beta,
      T* C,
      T* CReference) {
    printf("Testing Systolic Gemm %zd %zd %zd | TransA? %d, TransB? %d\n", M, N, K, TransA, TransB);
    std::fill_n(C, M * N, -0.5f);
    std::fill_n(CReference, M * N, -0.5f);


    // printf("A matrix\n");
    // for (size_t i = 0; i < M; i++) {
    //   for (size_t j = 0; j < K; j++){
    //     printf("%f ", A[i*K + j]);
    //   }
    //   printf("\n");
    // }

    // printf("\nB matrix\n");
    // for (size_t i = 0; i < K; i++) {
    //   for (size_t j = 0; j < N; j++){
    //     printf("%f ", B[i*N + j]);
    //   }
    //   printf("\n");
    // }


    SystolicGemm(acceleration_type_, TransA, TransB, M, N, K, alpha, A, B, beta, C);
    ReferenceGemm(TransA, TransB, M, N, K, alpha, A, B, beta, CReference);

    // printf("\nC matrix\n");
    // for (size_t i = 0; i < M; i++) {
    //   for (size_t j = 0; j < N; j++){
    //     printf("%f ", C[i*N + j]);
    //   }
    //   printf("\n");
    // }

    // printf("\nCref matrix\n");
    // for (size_t i = 0; i < M; i++) {
    //   for (size_t j = 0; j < N; j++){
    //     printf("%f ", CReference[i*N + j]);
    //   }
    //   printf("\n");
    // }

    for (size_t f = 0; f < M * N; f++) {
      // Sensitive to comparing positive/negative zero.
      if (C[f] != CReference[f]) {
        printf("mismatch TransA=%d, TransB=%d, M=%zd, N=%zd, K=%zd, alpha=%f, beta=%f  %f %f!\n", TransA, TransB, M, N, K, alpha, beta, float(C[f]), float(CReference[f]));
        break;
      }
    }
  }

void
  Test(
      bool TransA,
      bool TransB,
      size_t M,
      size_t N,
      size_t K,
      float alpha,
      const T* A,
      int lda,
      const T* B,
      int ldb,
      float beta,
      T* C,
      int ldc,
      T* CReference) {
    printf("Testing Systolic strided Gemm %zd %zd %zd | TransA? %d, TransB? %d\n", M, N, K, TransA, TransB);
    std::fill_n(C, M * N, -0.5f);
    std::fill_n(CReference, M * N, -0.5f);

    SystolicGemm(acceleration_type_, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    ReferenceGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, CReference, ldc);

    for (size_t f = 0; f < M * N; f++) {
      // Sensitive to comparing positive/negative zero.
      if (C[f] != CReference[f]) {
        printf("mismatch TransA=%d, TransB=%d, M=%zd, N=%zd, K=%zd, alpha=%f, beta=%f  %f %f!\n", TransA, TransB, M, N, K, alpha, beta, float(C[f]), float(CReference[f]));
        break;
      }
    }
  }


  void ReferenceGemm(
      bool TransA,
      bool TransB,
      size_t M,
      size_t N,
      size_t K,
      float alpha,
      const T* A,
      const T* B,
      float beta,
      T* C) {
    int lda = TransA ? M : K;
    int ldb = TransB ? K : N;
    ReferenceGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
  }

  void
  ReferenceGemm(
      bool TransA,
      bool TransB,
      size_t M,
      size_t N,
      size_t K,
      float alpha,
      const T* A,
      size_t lda,
      const T* B,
      size_t ldb,
      float beta,
      T* C,
      size_t ldc) {
    if (!TransA) {
      if (!TransB) {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + (m * lda);
            const T* b = B + n;
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += ldb;
              a += 1;
            }

            *c = (beta != 0 ? (*c * beta) : 0) + (sum * alpha);
          }
        }

      } else {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + (m * lda);
            const T* b = B + (n * ldb);
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += 1;
              a += 1;
            }

            *c = (beta != 0 ? (*c * beta) : 0) + (sum * alpha);
          }
        }
      }

    } else {
      if (!TransB) {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + m;
            const T* b = B + n;
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += ldb;
              a += lda;
            }

            *c = (beta != 0 ? (*c * beta) : 0) + (sum * alpha);
          }
        }

      } else {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + m;
            const T* b = B + (n * ldb);
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += 1;
              a += lda;
            }

            *c = (beta != 0 ? (*c * beta) : 0) + (sum * alpha);
          }
        }
      }
    }
  }

  MatrixGuardBuffer<T> BufferA;
  MatrixGuardBuffer<T> BufferB;
  MatrixGuardBuffer<T> BufferC;
  MatrixGuardBuffer<T> BufferCReference;
  char acceleration_type_;

 public:
  void
  ExecuteShort(
      void) override {
    Test(2, 3, 2, 1.0f, 0.0f);
    Test(5, 7, 9, 1.0f, 1.0f);
    Test(13, 15, 17, 0.5f, 0.5f);
    Test(11, 15, 17, 0.5f, 0.0f);
    // Test strided
    Test(196, 200, 16, 16, 200, 200, 1.0f, 0.0f);
  }

  MlasSystolicGemmTest(char acceleration_type) : acceleration_type_(acceleration_type) {}
};
#endif