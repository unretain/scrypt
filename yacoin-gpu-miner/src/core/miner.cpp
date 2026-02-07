/*
 * Scrypt Coin GPU Miner - Core Implementation
 * AdaptivePow Algorithm
 */

#include "miner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef WITH_CUDA
    #include <cuda_runtime.h>
    extern "C" {
        int adaptivepow_cuda_init(int deviceId, uint32_t epoch, void** ctx);
        int adaptivepow_cuda_generate_dag(void* ctx);
        int adaptivepow_cuda_search(void* ctx, const uint32_t* header, uint64_t target,
                                     uint64_t startNonce, uint64_t* foundNonce, uint32_t* hashCount);
        void adaptivepow_cuda_cleanup(void* ctx);
    }
#endif

#ifdef WITH_OPENCL
    #include <CL/cl.h>
    extern "C" {
        int adaptivepow_opencl_init(int deviceId, uint32_t epoch, void** ctx);
        int adaptivepow_opencl_generate_dag(void* ctx);
        int adaptivepow_opencl_search(void* ctx, const uint32_t* header, uint64_t target,
                                       uint64_t startNonce, uint64_t* foundNonce, uint32_t* hashCount);
        void adaptivepow_opencl_cleanup(void* ctx);
    }
#endif

// Internal miner context
struct MinerContext {
    int deviceId;
    uint32_t epoch;
    uint64_t dagSize;
    bool isCuda;
    bool dagReady;
    void* gpuContext;

    // Stats
    uint64_t totalHashes;
    uint64_t acceptedShares;
    uint64_t rejectedShares;
    time_t startTime;

    // Current nonce position
    uint64_t currentNonce;

    // Pending result
    bool hasResult;
    MiningResult pendingResult;
};

// ==================== AdaptivePow Functions ====================

uint32_t adaptivepow_get_epoch(uint64_t timestamp, uint64_t genesisTime)
{
    if (timestamp <= genesisTime) return 0;
    return (timestamp - genesisTime) / ADAPTIVEPOW_EPOCH_LENGTH;
}

uint64_t adaptivepow_get_dag_size(uint32_t epoch)
{
    uint32_t doublings = epoch / ADAPTIVEPOW_GROWTH_RATE;
    if (doublings > 10) doublings = 10;  // Cap at ~1 TB
    return ADAPTIVEPOW_DAG_BASE_SIZE << doublings;
}

void adaptivepow_get_seed(uint32_t epoch, uint8_t seed[32])
{
    // Keccak-256 of epoch number
    memset(seed, 0, 32);
    seed[0] = epoch & 0xFF;
    seed[1] = (epoch >> 8) & 0xFF;
    seed[2] = (epoch >> 16) & 0xFF;
    seed[3] = (epoch >> 24) & 0xFF;

    // Simple hash (in production, use proper Keccak)
    for (int i = 0; i < 32; i++) {
        seed[i] ^= (epoch * 0x01000193) >> (i % 4 * 8);
    }
}

// ==================== GPU Functions ====================

int enumerate_gpus(GPUDevice *devices, int maxDevices)
{
    int count = 0;

#ifdef WITH_CUDA
    int cudaDevices = 0;
    if (cudaGetDeviceCount(&cudaDevices) == cudaSuccess) {
        for (int i = 0; i < cudaDevices && count < maxDevices; i++) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                devices[count].id = count;
                strncpy(devices[count].name, prop.name, sizeof(devices[count].name) - 1);
                devices[count].memory = prop.totalGlobalMem;
                devices[count].freeMemory = prop.totalGlobalMem;  // Approximate
                devices[count].computeUnits = prop.multiProcessorCount;
                devices[count].maxThreads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
                devices[count].available = true;
                devices[count].isCuda = true;
                count++;
            }
        }
    }
#endif

#ifdef WITH_OPENCL
    cl_platform_id platforms[8];
    cl_uint numPlatforms;
    if (clGetPlatformIDs(8, platforms, &numPlatforms) == CL_SUCCESS) {
        for (cl_uint p = 0; p < numPlatforms && count < maxDevices; p++) {
            cl_device_id clDevices[8];
            cl_uint numDevices;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, clDevices, &numDevices) == CL_SUCCESS) {
                for (cl_uint d = 0; d < numDevices && count < maxDevices; d++) {
                    char name[256];
                    cl_ulong mem;
                    cl_uint units;

                    clGetDeviceInfo(clDevices[d], CL_DEVICE_NAME, sizeof(name), name, NULL);
                    clGetDeviceInfo(clDevices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
                    clGetDeviceInfo(clDevices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, NULL);

                    devices[count].id = count;
                    strncpy(devices[count].name, name, sizeof(devices[count].name) - 1);
                    devices[count].memory = mem;
                    devices[count].freeMemory = mem;  // Approximate
                    devices[count].computeUnits = units;
                    devices[count].maxThreads = units * 256;  // Approximate
                    devices[count].available = true;
                    devices[count].isCuda = false;
                    count++;
                }
            }
        }
    }
#endif

    return count;
}

MinerContext* miner_init(int deviceId, uint32_t epoch)
{
    MinerContext* ctx = (MinerContext*)calloc(1, sizeof(MinerContext));
    if (!ctx) return NULL;

    ctx->deviceId = deviceId;
    ctx->epoch = epoch;
    ctx->dagSize = adaptivepow_get_dag_size(epoch);
    ctx->startTime = time(NULL);

    // Detect GPU type and initialize
    GPUDevice devices[8];
    int numDevices = enumerate_gpus(devices, 8);

    if (deviceId >= numDevices) {
        free(ctx);
        return NULL;
    }

    ctx->isCuda = devices[deviceId].isCuda;

    int result = -1;

#ifdef WITH_CUDA
    if (ctx->isCuda) {
        result = adaptivepow_cuda_init(deviceId, epoch, &ctx->gpuContext);
    }
#endif

#ifdef WITH_OPENCL
    if (!ctx->isCuda) {
        result = adaptivepow_opencl_init(deviceId, epoch, &ctx->gpuContext);
    }
#endif

    if (result != 0) {
        free(ctx);
        return NULL;
    }

    return ctx;
}

int miner_generate_dag(MinerContext *ctx)
{
    if (!ctx || ctx->dagReady) return -1;

    int result = -1;

#ifdef WITH_CUDA
    if (ctx->isCuda) {
        result = adaptivepow_cuda_generate_dag(ctx->gpuContext);
    }
#endif

#ifdef WITH_OPENCL
    if (!ctx->isCuda) {
        result = adaptivepow_opencl_generate_dag(ctx->gpuContext);
    }
#endif

    if (result == 0) {
        ctx->dagReady = true;
    }

    return result;
}

bool miner_dag_ready(MinerContext *ctx)
{
    return ctx && ctx->dagReady;
}

int miner_submit_job(MinerContext *ctx, const MiningJob *job)
{
    if (!ctx || !ctx->dagReady || !job) return -1;

    // Build header from job
    uint32_t header[20];
    memset(header, 0, sizeof(header));

    // Copy prevHash and merkleRoot
    memcpy(&header[0], job->prevHash, 32);
    memcpy(&header[8], job->merkleRoot, 32);
    header[16] = job->nTime;
    header[17] = job->nBits;
    // header[18-19] will be the nonce, set by GPU

    uint64_t foundNonce = 0;
    uint32_t hashCount = 0;

    int result = -1;

#ifdef WITH_CUDA
    if (ctx->isCuda) {
        result = adaptivepow_cuda_search(ctx->gpuContext, header, job->target,
                                          ctx->currentNonce, &foundNonce, &hashCount);
    }
#endif

#ifdef WITH_OPENCL
    if (!ctx->isCuda) {
        result = adaptivepow_opencl_search(ctx->gpuContext, header, job->target,
                                            ctx->currentNonce, &foundNonce, &hashCount);
    }
#endif

    ctx->totalHashes += hashCount;
    ctx->currentNonce += hashCount;

    if (result > 0) {
        // Found a valid nonce
        ctx->hasResult = true;
        ctx->pendingResult.found = true;
        ctx->pendingResult.nonce = foundNonce;
        strncpy(ctx->pendingResult.jobId, job->jobId, sizeof(ctx->pendingResult.jobId) - 1);
    }

    return result;
}

int miner_get_result(MinerContext *ctx, MiningResult *result)
{
    if (!ctx || !result) return -1;

    if (ctx->hasResult) {
        memcpy(result, &ctx->pendingResult, sizeof(MiningResult));
        ctx->hasResult = false;
        return 1;
    }

    result->found = false;
    return 0;
}

void miner_get_stats(MinerContext *ctx, MinerStats *stats)
{
    if (!ctx || !stats) return;

    memset(stats, 0, sizeof(MinerStats));
    stats->totalHashes = ctx->totalHashes;
    stats->acceptedShares = ctx->acceptedShares;
    stats->rejectedShares = ctx->rejectedShares;
    stats->currentEpoch = ctx->epoch;
    stats->dagSize = ctx->dagSize;
    stats->uptime = difftime(time(NULL), ctx->startTime);

    // Calculate hashrate
    if (stats->uptime > 0) {
        stats->hashrate = stats->totalHashes / stats->uptime;
    }

    // GPU temperature would require platform-specific API calls
    stats->gpuTemp = 0;  // Not implemented
    stats->gpuPower = 0;
}

int miner_update_epoch(MinerContext *ctx, uint32_t newEpoch)
{
    if (!ctx) return -1;

    ctx->epoch = newEpoch;
    ctx->dagSize = adaptivepow_get_dag_size(newEpoch);
    ctx->dagReady = false;

    // Regenerate DAG for new epoch
    return miner_generate_dag(ctx);
}

void miner_shutdown(MinerContext *ctx)
{
    if (!ctx) return;

#ifdef WITH_CUDA
    if (ctx->isCuda && ctx->gpuContext) {
        adaptivepow_cuda_cleanup(ctx->gpuContext);
    }
#endif

#ifdef WITH_OPENCL
    if (!ctx->isCuda && ctx->gpuContext) {
        adaptivepow_opencl_cleanup(ctx->gpuContext);
    }
#endif

    free(ctx);
}

// ==================== Utility Functions ====================

uint64_t bits_to_target64(uint32_t nBits)
{
    uint32_t nSize = nBits >> 24;
    uint32_t nWord = nBits & 0x007fffff;

    if (nSize <= 3) {
        return nWord >> (8 * (3 - nSize));
    } else {
        // For higher difficulties, return max value
        return 0xFFFFFFFFFFFFFFFFULL >> ((nSize - 3) * 8);
    }
}

void bits_to_target256(uint32_t nBits, uint8_t target[32])
{
    memset(target, 0, 32);

    uint32_t nSize = nBits >> 24;
    uint32_t nWord = nBits & 0x007fffff;

    if (nSize <= 3) {
        nWord >>= 8 * (3 - nSize);
        target[0] = nWord & 0xFF;
        target[1] = (nWord >> 8) & 0xFF;
        target[2] = (nWord >> 16) & 0xFF;
    } else {
        int offset = nSize - 3;
        target[offset] = nWord & 0xFF;
        target[offset + 1] = (nWord >> 8) & 0xFF;
        target[offset + 2] = (nWord >> 16) & 0xFF;
    }
}

double target_to_difficulty(uint64_t target)
{
    if (target == 0) return 0;
    return (double)0xFFFFFFFFFFFFFFFFULL / (double)target;
}

bool verify_solution(const MiningJob *job, const MiningResult *result)
{
    // TODO: Implement full verification
    // This would recompute the hash and compare against target
    return result->found;
}
