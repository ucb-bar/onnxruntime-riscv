/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    unittest.cpp

Abstract:

    This module implements unit tests of the MLAS library.

--*/

#include <stdio.h>
#include <memory.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <mlas.h>
#include "almostequal.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
#include "core/platform/threadpool.h"
#endif

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

MLAS_THREADPOOL* threadpool = nullptr;

template<typename T>
class MatrixGuardBuffer
{
public:
    MatrixGuardBuffer()
    {
        _BaseBuffer = nullptr;
        _BaseBufferSize = 0;
        _ElementsAllocated = 0;
    }

    ~MatrixGuardBuffer(void)
    {
        ReleaseBuffer();
    }

    T* GetBuffer(size_t Elements, bool ZeroFill = false)
    {
        //
        // Check if the internal buffer needs to be reallocated.
        //

        if (Elements > _ElementsAllocated) {

            ReleaseBuffer();

            //
            // Reserve a virtual address range for the allocation plus an unmapped
            // guard region.
            //

            constexpr size_t BufferAlignment = 64 * 1024;
            constexpr size_t GuardPadding = 256 * 1024;

            size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

            _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
            _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
            _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

            if (_BaseBuffer == nullptr) {
                printf("Failed to allocate base buffer memory.\n");
                ORT_THROW_EX(std::bad_alloc);
            }

            //
            // Commit the number of bytes for the allocation leaving the upper
            // guard region as unmapped.
            //

#if defined(_WIN32)
            if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
                ORT_THROW_EX(std::bad_alloc);
            }
#else
            if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
                printf("Failed to protect guard region. Retrying without guard enabled.\n");
                munmap(_BaseBuffer, _BaseBufferSize);
                _BaseBuffer = mmap(0, _BaseBufferSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (_BaseBuffer == nullptr) {
                    printf("Failed to allocate base buffer memory.\n");
                    ORT_THROW_EX(std::bad_alloc);
                }
            }
#endif

            _ElementsAllocated = BytesToAllocate / sizeof(T);
            _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
        }

        //
        //
        //

        T* GuardAddress = _GuardAddress;
        T* buffer = GuardAddress - Elements;

        if (ZeroFill) {

            std::fill_n(buffer, Elements, T(0));

        } else {

            const int MinimumFillValue = -23;
            const int MaximumFillValue = 23;

            int FillValue = MinimumFillValue;
            T* FillAddress = buffer;

            while (FillAddress < GuardAddress) {

                *FillAddress++ = (T)FillValue;

                FillValue++;

                if (FillValue > MaximumFillValue) {
                    FillValue = MinimumFillValue;
                }
            }
        }

        return buffer;
    }

    void ReleaseBuffer(void)
    {
        if (_BaseBuffer != nullptr) {

#if defined(_WIN32)
            VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
            munmap(_BaseBuffer, _BaseBufferSize);
#endif

            _BaseBuffer = nullptr;
            _BaseBufferSize = 0;
        }

        _ElementsAllocated = 0;
    }

private:
    size_t _ElementsAllocated;
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    T* _GuardAddress;
};

class MlasTestBase
{
public:
    virtual
    ~MlasTestBase(
        void
        )
    {
    }

    //
    // Contains tests that run quickly as part of a checkin integration to
    // sanity check that the functionality is working.
    //

    virtual
    void
    ExecuteShort(
        void
        )
    {
    }

    //
    // Contains tests that can run slowly to more exhaustively test that
    // functionality is working across a broader range of parameters.
    //

    virtual
    void
    ExecuteLong(
        void
        )
    {
    }
};

#ifdef USE_SYSTOLIC
#include "systolic_unittest.h"
#endif

#ifdef USE_HWACHA
#include "hwacha_unittest.h"
#endif

void
RunThreadedTests(
    void
    )
{
   // Put any threaded tests you want in here
}

//#include <sys/mman.h>

int
#if defined(_WIN32)
__cdecl
#endif
main(
    int argc, char *argv[]  __attribute__((unused))
    )
{
    (void)(argc); // Suppress unused argc
    setbuf(stdout, NULL);

#ifdef FOR_FIRESIM
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
     perror("mlockall failed");
     exit(1);
    } else {
        printf("Finished mlock\n");
    }
#endif
    
    // NOTE: IF Gemmini tests freeze, first try disabling guard page
    // To see if cuase is an OOB read into the guard page
#ifdef USE_SYSTOLIC
#ifdef SYSTOLIC_INT8
    printf("Systolic Int8 Conv tests.\n");
    std::make_unique<MlasSystolicConvTest<int8_t, int32_t>>(argc - 1)->ExecuteShort();
    printf("Systolic Int8 Resadd tests.\n");
    std::make_unique<MlasSystolicAddTest<int8_t, int32_t>>(argc - 1)->ExecuteShort();
    printf("Systolic Int8 Matmul tests.\n");
    std::make_unique<MlasSystolicMatmulTest<int8_t, int32_t>>(argc - 1)->ExecuteShort();
#endif
#ifdef SYSTOLIC_FP32
    printf("Systolic Fp32Gemm.\n");
    std::make_unique<MlasSystolicGemmTest<float>>(argc - 1)->ExecuteShort();
    printf("Systolic Fp32 Matmul tests.\n");
    std::make_unique<MlasSystolicMatmulTest<float, float>>(argc - 1)->ExecuteShort();
#endif
#endif

    //
    // Run threaded tests without the thread pool.
    //

    RunThreadedTests();

#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
    // Load bearing if statement.
    // Even though threadpool will always be nullptr and this block is never taken, do not delete.
    // No, I'm serious. Removing this block entirely (e.g. via ifdef) causes an immediate segfault.
    // You're welcome to try to debug this. I suspect it's something being optimized out.
    
    // We want to skip threaded tests for now since they are not supported in spike.
    // They work fine in qemu though.
    // IMPORTANT: If you enable the threadpool test, you have to re-link the mlas binary
    // with lpthread as whole-archive. See the comment in build.sh.
    if (threadpool != nullptr) {
        //
        // Run threaded tests using the thread pool.
        //

        threadpool = new onnxruntime::concurrency::ThreadPool(
            &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 2, true);

        RunThreadedTests();

        delete threadpool;
    }
#endif

    printf("Done.\n");

    return 0;
}