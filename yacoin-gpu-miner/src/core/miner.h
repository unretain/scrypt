/*
 * Scrypt Coin GPU Miner - Core Header
 * AdaptivePow Algorithm Implementation
 */

#ifndef SCRYPT_MINER_H
#define SCRYPT_MINER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// AdaptivePow constants
#define ADAPTIVEPOW_DAG_BASE_SIZE  (1ULL << 30)  // 1 GB
#define ADAPTIVEPOW_EPOCH_LENGTH   (180 * 24 * 60 * 60)  // 180 days
#define ADAPTIVEPOW_GROWTH_RATE    4  // DAG doubles every 4 epochs
#define ADAPTIVEPOW_DAG_LOADS      64
#define ADAPTIVEPOW_MATH_OPS       16

// Scrypt Coin block header (112 bytes for AdaptivePow)
typedef struct {
    int32_t  nVersion;          // 4 bytes
    uint8_t  hashPrevBlock[32]; // 32 bytes
    uint8_t  hashMerkleRoot[32];// 32 bytes
    uint32_t nTime;             // 4 bytes
    uint32_t nBits;             // 4 bytes
    uint64_t nNonce64;          // 8 bytes (64-bit for GPU mining)
    uint8_t  hashMix[32];       // 32 bytes (mix hash for quick verification)
} AdaptivePowHeader;

// Mining job from stratum or solo RPC
typedef struct {
    char     jobId[64];
    uint8_t  prevHash[32];
    uint8_t  merkleRoot[32];
    uint32_t nTime;
    uint32_t nBits;
    uint32_t epoch;            // Current DAG epoch
    uint64_t dagSize;          // Current DAG size in bytes
    uint64_t target;           // 64-bit target for GPU comparison
    uint8_t  target256[32];    // Full 256-bit target
    bool     cleanJobs;
} MiningJob;

// Mining result
typedef struct {
    char     jobId[64];
    uint64_t nonce;            // 64-bit nonce
    uint8_t  mixHash[32];      // Mix hash for verification
    uint8_t  hash[32];         // Final hash
    bool     found;
} MiningResult;

// GPU device info
typedef struct {
    int      id;
    char     name[256];
    size_t   memory;          // Total VRAM in bytes
    size_t   freeMemory;      // Available VRAM
    int      computeUnits;
    int      maxThreads;
    bool     available;
    bool     isCuda;          // true = NVIDIA CUDA, false = AMD OpenCL
} GPUDevice;

// Miner statistics
typedef struct {
    double   hashrate;         // Hashes per second
    uint64_t totalHashes;
    uint64_t acceptedShares;
    uint64_t rejectedShares;
    uint32_t currentEpoch;
    uint64_t dagSize;
    double   uptime;
    double   gpuTemp;          // GPU temperature in Celsius
    double   gpuPower;         // GPU power usage in Watts
} MinerStats;

// Miner context (opaque)
typedef struct MinerContext MinerContext;

// ==================== AdaptivePow Functions ====================

// Calculate epoch from timestamp
uint32_t adaptivepow_get_epoch(uint64_t timestamp, uint64_t genesisTime);

// Calculate DAG size for epoch
uint64_t adaptivepow_get_dag_size(uint32_t epoch);

// Generate DAG seed for epoch
void adaptivepow_get_seed(uint32_t epoch, uint8_t seed[32]);

// ==================== GPU Functions ====================

// Enumerate available GPUs
int enumerate_gpus(GPUDevice *devices, int maxDevices);

// Initialize miner on specific GPU
MinerContext* miner_init(int deviceId, uint32_t epoch);

// Generate DAG for current epoch (call after miner_init)
int miner_generate_dag(MinerContext *ctx);

// Check if DAG is ready
bool miner_dag_ready(MinerContext *ctx);

// Process a mining job (non-blocking, returns immediately)
int miner_submit_job(MinerContext *ctx, const MiningJob *job);

// Check for mining results
int miner_get_result(MinerContext *ctx, MiningResult *result);

// Get mining statistics
void miner_get_stats(MinerContext *ctx, MinerStats *stats);

// Update to new epoch (regenerates DAG)
int miner_update_epoch(MinerContext *ctx, uint32_t newEpoch);

// Cleanup and free resources
void miner_shutdown(MinerContext *ctx);

// ==================== Utility Functions ====================

// Convert compact target (nBits) to 64-bit target
uint64_t bits_to_target64(uint32_t nBits);

// Convert compact target (nBits) to 256-bit target
void bits_to_target256(uint32_t nBits, uint8_t target[32]);

// Calculate difficulty from target
double target_to_difficulty(uint64_t target);

// Verify a solution
bool verify_solution(const MiningJob *job, const MiningResult *result);

#ifdef __cplusplus
}
#endif

#endif // SCRYPT_MINER_H
