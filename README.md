# GPU Based Vector Summation
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution confi guration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## Aim:
To explore the differences between the execution configurations of PCA-GPU-based vector summation.

## Procedure:
1)The program will start executing, and you will see the name of the device being used printed on the console.

2)The vector size is set to 2^24, which corresponds to 16,777,216 elements. This can be modified by changing the nElem variable in the code.

3)The program will allocate memory for the host arrays h_A, h_B, hostRef, and gpuRef using malloc().

4)Random values will be generated and assigned to the host arrays h_A and h_B using the initialData() function.

5)The sumArraysOnHost() function will be called to perform vector addition on the host CPU. The result will be stored in the hostRef array.

6)Memory will be allocated on the GPU for the device arrays d_A, d_B, and d_C using cudaMalloc().

7)The data from the host arrays h_A and h_B will be copied to the corresponding device arrays d_A and d_B using cudaMemcpy().

8)The kernel function sumArraysOnGPU() will be invoked on the GPU using the specified grid and block dimensions. The grid dimensions are calculated based on the number of elements and the block dimensions.

9)The GPU execution time will be measured using the seconds() function and printed on the console.

10)The device array d_C will be copied back to the host array gpuRef using cudaMemcpy().

11)The checkResult() function will be called to compare the results of the host and device arrays and check if they match.

12)Finally, the device memory and host memory will be freed using cudaFree() and free() respectively.

13)The program will terminate, and you will see the result of the comparison between the host and device arrays printed on the console.

  Note: You can modify the code to experiment with different vector sizes, block dimensions, or kernel configurations to observe their impact on performance and correctness.
## Program:
```c
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <windows.h>

inline double seconds()
{
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}
void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if ( fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}
void initialData(float* ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}
void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
        if (i + blockDim.x < N)
        {
            C[i + blockDim.x] = A[i + blockDim.x] + B[i + blockDim.x];
        }
    }
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = 256;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    iStart = seconds();
    sumArraysOnGPU << <grid, block >> > (d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
        block.x, iElaps);

    // check kernel error
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}
```
## Output:
### 1-
### Block size = 1023
![235474476-543c8153-67c7-4488-b117-efaaecf4a71e](https://github.com/ragav-47/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/75235488/8d073fc9-1db9-4b60-9fda-f174091171ee)

### Block size = 1024
![235474535-547c521d-50e1-4625-94d2-ce04598cc622](https://github.com/ragav-47/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/75235488/e894c57c-c5f1-4852-b60c-5e1a891b0caf)
### 2- 
### Block size = 256. Two Threads.

![235474812-97ac4808-6fd8-4b47-a0b0-656e6d1c94f3](https://github.com/ragav-47/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/75235488/87e7e457-35bc-412f-b419-c6707500c54d)


## Result:
The result of the experiment will be a comparison of the execution times and results obtained from different execution configurations. This comparison will help determine the most efficient execution configuration for PCA-GPU-based vector summation.
