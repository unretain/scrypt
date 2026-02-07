/*
 * Scrypt Coin GPU Miner - OpenCL Backend
 * AdaptivePow Algorithm for AMD GPUs
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sstream>

// AdaptivePow constants
#define DAG_BASE_SIZE      (1ULL << 30)  // 1 GB
#define EPOCH_LENGTH       (180 * 24 * 60 * 60)
#define GROWTH_RATE        4
#define HASH_BYTES         64
#define BATCH_SIZE         (8192 * 256)  // 2M hashes per batch

// OpenCL context
struct OpenCLContext {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    // Kernels
    cl_kernel searchKernel;
    cl_kernel dagKernel;
    cl_kernel cacheKernel;

    // Buffers
    cl_mem dagBuffer;
    cl_mem headerBuffer;
    cl_mem resultsBuffer;
    cl_mem resultCountBuffer;

    // State
    uint32_t epoch;
    uint64_t dagSize;
    bool dagReady;
};

// Load kernel source from file or embedded string
static std::string loadKernelSource()
{
    // Try to load from file first
    std::ifstream file("share/scrypt-miner/adaptivepow.cl");
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    // Try alternative locations
    const char* paths[] = {
        "adaptivepow.cl",
        "../src/opencl/adaptivepow.cl",
        "/usr/share/scrypt-miner/adaptivepow.cl",
        NULL
    };

    for (int i = 0; paths[i] != NULL; i++) {
        std::ifstream f(paths[i]);
        if (f.is_open()) {
            std::stringstream buffer;
            buffer << f.rdbuf();
            return buffer.str();
        }
    }

    // If file not found, return empty (would need embedded kernel)
    fprintf(stderr, "Warning: Could not find adaptivepow.cl kernel file\n");
    return "";
}

// Calculate DAG size for epoch
static uint64_t getDagSize(uint32_t epoch)
{
    uint32_t doublings = epoch / GROWTH_RATE;
    if (doublings > 10) doublings = 10;
    return DAG_BASE_SIZE << doublings;
}

extern "C" {

int adaptivepow_opencl_init(int deviceId, uint32_t epoch, void** ctx)
{
    OpenCLContext* oclCtx = new OpenCLContext();
    memset(oclCtx, 0, sizeof(OpenCLContext));
    oclCtx->epoch = epoch;
    oclCtx->dagSize = getDagSize(epoch);

    cl_int err;

    // Get platforms
    cl_platform_id platforms[8];
    cl_uint numPlatforms;
    err = clGetPlatformIDs(8, platforms, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        delete oclCtx;
        return -1;
    }

    // Find device
    int currentDevice = 0;
    for (cl_uint p = 0; p < numPlatforms; p++) {
        cl_device_id devices[8];
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devices, &numDevices);
        if (err != CL_SUCCESS) continue;

        for (cl_uint d = 0; d < numDevices; d++) {
            if (currentDevice == deviceId) {
                oclCtx->platform = platforms[p];
                oclCtx->device = devices[d];
                break;
            }
            currentDevice++;
        }
        if (oclCtx->device) break;
    }

    if (!oclCtx->device) {
        delete oclCtx;
        return -2;
    }

    // Create context
    oclCtx->context = clCreateContext(NULL, 1, &oclCtx->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        delete oclCtx;
        return -3;
    }

    // Create command queue
    oclCtx->queue = clCreateCommandQueue(oclCtx->context, oclCtx->device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(oclCtx->context);
        delete oclCtx;
        return -4;
    }

    // Load and build program
    std::string source = loadKernelSource();
    if (source.empty()) {
        clReleaseCommandQueue(oclCtx->queue);
        clReleaseContext(oclCtx->context);
        delete oclCtx;
        return -5;
    }

    const char* srcPtr = source.c_str();
    size_t srcLen = source.length();
    oclCtx->program = clCreateProgramWithSource(oclCtx->context, 1, &srcPtr, &srcLen, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(oclCtx->queue);
        clReleaseContext(oclCtx->context);
        delete oclCtx;
        return -6;
    }

    err = clBuildProgram(oclCtx->program, 1, &oclCtx->device, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(oclCtx->program, oclCtx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* log = new char[logSize];
        clGetProgramBuildInfo(oclCtx->program, oclCtx->device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "OpenCL build error:\n%s\n", log);
        delete[] log;

        clReleaseProgram(oclCtx->program);
        clReleaseCommandQueue(oclCtx->queue);
        clReleaseContext(oclCtx->context);
        delete oclCtx;
        return -7;
    }

    // Create kernels
    oclCtx->searchKernel = clCreateKernel(oclCtx->program, "adaptivepow_search", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(oclCtx->program);
        clReleaseCommandQueue(oclCtx->queue);
        clReleaseContext(oclCtx->context);
        delete oclCtx;
        return -8;
    }

    oclCtx->dagKernel = clCreateKernel(oclCtx->program, "generate_dag", &err);
    oclCtx->cacheKernel = clCreateKernel(oclCtx->program, "generate_cache", &err);

    // Allocate DAG buffer
    oclCtx->dagBuffer = clCreateBuffer(oclCtx->context, CL_MEM_READ_ONLY, oclCtx->dagSize, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to allocate DAG buffer (%.2f GB)\n", oclCtx->dagSize / 1e9);
        clReleaseKernel(oclCtx->searchKernel);
        clReleaseProgram(oclCtx->program);
        clReleaseCommandQueue(oclCtx->queue);
        clReleaseContext(oclCtx->context);
        delete oclCtx;
        return -9;
    }

    // Allocate other buffers
    oclCtx->headerBuffer = clCreateBuffer(oclCtx->context, CL_MEM_READ_ONLY, 80, NULL, &err);
    oclCtx->resultsBuffer = clCreateBuffer(oclCtx->context, CL_MEM_WRITE_ONLY, 32 * sizeof(uint32_t), NULL, &err);
    oclCtx->resultCountBuffer = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &err);

    printf("OpenCL initialized:\n");
    printf("  Epoch: %u\n", epoch);
    printf("  DAG size: %.2f GB\n", oclCtx->dagSize / 1e9);

    *ctx = oclCtx;
    return 0;
}

int adaptivepow_opencl_generate_dag(void* ctx)
{
    OpenCLContext* oclCtx = (OpenCLContext*)ctx;
    if (!oclCtx) return -1;

    cl_int err;
    uint64_t dagItems = oclCtx->dagSize / HASH_BYTES;
    uint64_t cacheSize = oclCtx->dagSize / 64;
    uint32_t cacheItems = cacheSize / HASH_BYTES;

    printf("Generating DAG with %llu items...\n", (unsigned long long)dagItems);

    // Allocate cache
    cl_mem cacheBuffer = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE, cacheSize, NULL, &err);
    if (err != CL_SUCCESS) return -2;

    // Generate seed (simplified - in production use proper Keccak)
    uint8_t seed[32];
    memset(seed, 0, 32);
    seed[0] = oclCtx->epoch & 0xFF;
    seed[1] = (oclCtx->epoch >> 8) & 0xFF;

    cl_mem seedBuffer = clCreateBuffer(oclCtx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, seed, &err);

    // Generate cache
    if (oclCtx->cacheKernel) {
        clSetKernelArg(oclCtx->cacheKernel, 0, sizeof(cl_mem), &seedBuffer);
        clSetKernelArg(oclCtx->cacheKernel, 1, sizeof(cl_mem), &cacheBuffer);
        clSetKernelArg(oclCtx->cacheKernel, 2, sizeof(uint32_t), &cacheItems);

        size_t globalSize = cacheItems;
        size_t localSize = 256;
        globalSize = ((globalSize + localSize - 1) / localSize) * localSize;

        err = clEnqueueNDRangeKernel(oclCtx->queue, oclCtx->cacheKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
        clFinish(oclCtx->queue);
    }

    // Generate DAG from cache
    if (oclCtx->dagKernel) {
        clSetKernelArg(oclCtx->dagKernel, 0, sizeof(cl_mem), &cacheBuffer);
        clSetKernelArg(oclCtx->dagKernel, 1, sizeof(uint32_t), &cacheItems);
        clSetKernelArg(oclCtx->dagKernel, 2, sizeof(cl_mem), &oclCtx->dagBuffer);
        clSetKernelArg(oclCtx->dagKernel, 3, sizeof(uint32_t), &dagItems);

        // Generate in batches to avoid GPU timeout
        size_t batchSize = 1024 * 1024;  // 1M items per batch
        for (uint64_t offset = 0; offset < dagItems; offset += batchSize) {
            size_t remaining = dagItems - offset;
            size_t thisSize = (remaining < batchSize) ? remaining : batchSize;

            size_t globalOffset = offset;
            size_t globalSize = thisSize;
            size_t localSize = 256;
            globalSize = ((globalSize + localSize - 1) / localSize) * localSize;

            err = clEnqueueNDRangeKernel(oclCtx->queue, oclCtx->dagKernel, 1, &globalOffset, &globalSize, &localSize, 0, NULL, NULL);

            if (offset % (10 * batchSize) == 0) {
                printf("  DAG progress: %.1f%%\n", 100.0 * offset / dagItems);
            }
        }
        clFinish(oclCtx->queue);
    }

    // Cleanup
    clReleaseMemObject(seedBuffer);
    clReleaseMemObject(cacheBuffer);

    oclCtx->dagReady = true;
    printf("DAG generation complete!\n");
    return 0;
}

int adaptivepow_opencl_search(void* ctx, const uint32_t* header, uint64_t target,
                               uint64_t startNonce, uint64_t* foundNonce, uint32_t* hashCount)
{
    OpenCLContext* oclCtx = (OpenCLContext*)ctx;
    if (!oclCtx || !oclCtx->dagReady) return -1;

    cl_int err;

    // Upload header
    err = clEnqueueWriteBuffer(oclCtx->queue, oclCtx->headerBuffer, CL_FALSE, 0, 80, header, 0, NULL, NULL);

    // Reset result counter
    uint32_t zero = 0;
    err = clEnqueueWriteBuffer(oclCtx->queue, oclCtx->resultCountBuffer, CL_FALSE, 0, sizeof(uint32_t), &zero, 0, NULL, NULL);

    // Set kernel args
    uint32_t dagItems = oclCtx->dagSize / HASH_BYTES;
    clSetKernelArg(oclCtx->searchKernel, 0, sizeof(cl_mem), &oclCtx->dagBuffer);
    clSetKernelArg(oclCtx->searchKernel, 1, sizeof(uint64_t), &startNonce);
    clSetKernelArg(oclCtx->searchKernel, 2, sizeof(cl_mem), &oclCtx->headerBuffer);
    clSetKernelArg(oclCtx->searchKernel, 3, sizeof(uint64_t), &target);
    clSetKernelArg(oclCtx->searchKernel, 4, sizeof(uint32_t), &dagItems);
    clSetKernelArg(oclCtx->searchKernel, 5, sizeof(cl_mem), &oclCtx->resultsBuffer);
    clSetKernelArg(oclCtx->searchKernel, 6, sizeof(cl_mem), &oclCtx->resultCountBuffer);

    // Launch kernel
    size_t globalSize = BATCH_SIZE;
    size_t localSize = 256;

    err = clEnqueueNDRangeKernel(oclCtx->queue, oclCtx->searchKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(oclCtx->queue);

    *hashCount = BATCH_SIZE;

    // Check for results
    uint32_t resultCount;
    clEnqueueReadBuffer(oclCtx->queue, oclCtx->resultCountBuffer, CL_TRUE, 0, sizeof(uint32_t), &resultCount, 0, NULL, NULL);

    if (resultCount > 0) {
        uint32_t results[32];
        clEnqueueReadBuffer(oclCtx->queue, oclCtx->resultsBuffer, CL_TRUE, 0, sizeof(results), results, 0, NULL, NULL);
        *foundNonce = ((uint64_t)results[1] << 32) | results[0];
        return 1;  // Found!
    }

    return 0;  // Not found
}

void adaptivepow_opencl_cleanup(void* ctx)
{
    OpenCLContext* oclCtx = (OpenCLContext*)ctx;
    if (!oclCtx) return;

    if (oclCtx->dagBuffer) clReleaseMemObject(oclCtx->dagBuffer);
    if (oclCtx->headerBuffer) clReleaseMemObject(oclCtx->headerBuffer);
    if (oclCtx->resultsBuffer) clReleaseMemObject(oclCtx->resultsBuffer);
    if (oclCtx->resultCountBuffer) clReleaseMemObject(oclCtx->resultCountBuffer);

    if (oclCtx->searchKernel) clReleaseKernel(oclCtx->searchKernel);
    if (oclCtx->dagKernel) clReleaseKernel(oclCtx->dagKernel);
    if (oclCtx->cacheKernel) clReleaseKernel(oclCtx->cacheKernel);

    if (oclCtx->program) clReleaseProgram(oclCtx->program);
    if (oclCtx->queue) clReleaseCommandQueue(oclCtx->queue);
    if (oclCtx->context) clReleaseContext(oclCtx->context);

    delete oclCtx;
}

} // extern "C"
