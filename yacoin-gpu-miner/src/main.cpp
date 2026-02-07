/*
 * Scrypt Coin GPU Miner - Main Entry Point
 *
 * AdaptivePow algorithm - GPU mining for Scrypt Coin
 * Supports CUDA (NVIDIA) and OpenCL (AMD)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
    #include <windows.h>
    #define sleep(x) Sleep((x) * 1000)
    #define msleep(x) Sleep(x)
#else
    #include <unistd.h>
    #define msleep(x) usleep((x) * 1000)
#endif

#include "core/miner.h"
#include "core/stratum.h"

// Version
#define SCRYPT_MINER_VERSION "1.0.0"

// Genesis time for Scrypt Coin (Feb 7, 2026 00:00:00 UTC)
#define SCRYPT_GENESIS_TIME 1738886400ULL

// Global state
static volatile bool g_running = true;
static MinerContext* g_miner = NULL;
static StratumClient g_stratum;

// Configuration
typedef struct {
    char url[256];
    char user[256];
    char pass[256];
    char address[128];     // Payout address for solo mining
    int  deviceId;
    int  intensity;
    bool benchmark;
    bool solo;
    bool listDevices;
} Config;

// Signal handler
void signal_handler(int sig)
{
    printf("\nShutting down...\n");
    g_running = false;
}

// Print usage
void print_usage(const char *program)
{
    printf("Scrypt Coin GPU Miner v%s\n", SCRYPT_MINER_VERSION);
    printf("AdaptivePow Algorithm - ASIC Resistant GPU Mining\n\n");
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  -o, --url <url>       Node/pool URL\n");
    printf("  -u, --user <user>     RPC username or wallet address\n");
    printf("  -p, --pass <pass>     RPC password (default: x)\n");
    printf("  -d, --device <id>     GPU device ID (default: 0)\n");
    printf("  -i, --intensity <n>   Mining intensity (8-25, default: auto)\n");
    printf("  --address <addr>      Payout address for solo mining\n");
    printf("  --solo                Solo mining mode (direct to node)\n");
    printf("  --benchmark           Run hashrate benchmark\n");
    printf("  --list-devices        List available GPUs\n");
    printf("  -h, --help            Show this help\n");
    printf("\nExamples:\n");
    printf("  Solo mining (to your node):\n");
    printf("    %s --solo -o http://127.0.0.1:9332 -u rpcuser -p rpcpass --address SYourAddress\n", program);
    printf("  Pool mining:\n");
    printf("    %s -o stratum+tcp://pool.scrypt.org:3333 -u SYourAddress -p x\n", program);
    printf("  Benchmark:\n");
    printf("    %s --benchmark\n", program);
}

// Parse command line arguments
int parse_args(int argc, char **argv, Config *config)
{
    memset(config, 0, sizeof(Config));
    strcpy(config->pass, "x");
    config->deviceId = 0;
    config->intensity = 0;  // Auto

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--url") == 0) {
            if (++i >= argc) return -1;
            strncpy(config->url, argv[i], sizeof(config->url) - 1);
        }
        else if (strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--user") == 0) {
            if (++i >= argc) return -1;
            strncpy(config->user, argv[i], sizeof(config->user) - 1);
        }
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--pass") == 0) {
            if (++i >= argc) return -1;
            strncpy(config->pass, argv[i], sizeof(config->pass) - 1);
        }
        else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
            if (++i >= argc) return -1;
            config->deviceId = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--intensity") == 0) {
            if (++i >= argc) return -1;
            config->intensity = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--address") == 0) {
            if (++i >= argc) return -1;
            strncpy(config->address, argv[i], sizeof(config->address) - 1);
        }
        else if (strcmp(argv[i], "--solo") == 0) {
            config->solo = true;
        }
        else if (strcmp(argv[i], "--benchmark") == 0) {
            config->benchmark = true;
        }
        else if (strcmp(argv[i], "--list-devices") == 0) {
            config->listDevices = true;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    return 0;
}

// List available GPU devices
int list_devices()
{
    printf("\n=== Available GPU Devices ===\n\n");

    GPUDevice devices[8];
    int numDevices = enumerate_gpus(devices, 8);

    if (numDevices <= 0) {
        printf("No GPU devices found!\n");
        printf("\nMake sure you have:\n");
        printf("  NVIDIA: CUDA drivers installed\n");
        printf("  AMD: ROCm or AMDGPU-PRO drivers installed\n");
        return -1;
    }

    for (int i = 0; i < numDevices; i++) {
        printf("[%d] %s\n", i, devices[i].name);
        printf("    Memory: %.2f GB (%.2f GB free)\n",
               devices[i].memory / (1024.0 * 1024.0 * 1024.0),
               devices[i].freeMemory / (1024.0 * 1024.0 * 1024.0));
        printf("    Compute Units: %d\n", devices[i].computeUnits);
        printf("    Type: %s\n", devices[i].isCuda ? "NVIDIA CUDA" : "AMD OpenCL");
        printf("\n");
    }

    // Calculate current epoch and DAG size
    uint64_t now = (uint64_t)time(NULL);
    uint32_t epoch = adaptivepow_get_epoch(now, SCRYPT_GENESIS_TIME);
    uint64_t dagSize = adaptivepow_get_dag_size(epoch);

    printf("=== AdaptivePow Info ===\n\n");
    printf("Current Epoch: %u\n", epoch);
    printf("Current DAG Size: %.2f GB\n", dagSize / (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    // Check which GPUs can mine
    printf("=== Mining Compatibility ===\n\n");
    for (int i = 0; i < numDevices; i++) {
        bool canMine = devices[i].freeMemory > dagSize + (256 * 1024 * 1024);  // DAG + 256MB headroom
        printf("[%d] %s: %s\n", i, devices[i].name,
               canMine ? "OK" : "INSUFFICIENT VRAM");
    }
    printf("\n");

    return 0;
}

// Run benchmark
int run_benchmark(Config *config)
{
    printf("\n");
    printf("+=============================================+\n");
    printf("|     SCRYPT COIN GPU MINER BENCHMARK        |\n");
    printf("|         AdaptivePow Algorithm              |\n");
    printf("+=============================================+\n\n");

    // Enumerate GPUs
    GPUDevice devices[8];
    int numDevices = enumerate_gpus(devices, 8);

    if (numDevices <= 0) {
        fprintf(stderr, "No GPU devices found!\n");
        return -1;
    }

    if (config->deviceId >= numDevices) {
        fprintf(stderr, "Invalid device ID: %d (only %d devices available)\n",
                config->deviceId, numDevices);
        return -1;
    }

    GPUDevice* dev = &devices[config->deviceId];
    printf("Testing GPU: %s\n", dev->name);
    printf("VRAM: %.2f GB\n", dev->memory / (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    // Calculate current epoch
    uint64_t now = (uint64_t)time(NULL);
    uint32_t epoch = adaptivepow_get_epoch(now, SCRYPT_GENESIS_TIME);
    uint64_t dagSize = adaptivepow_get_dag_size(epoch);

    printf("Epoch: %u\n", epoch);
    printf("DAG Size: %.2f GB\n", dagSize / (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    // Check if GPU has enough memory
    if (dev->memory < dagSize + (256 * 1024 * 1024)) {
        fprintf(stderr, "ERROR: GPU does not have enough VRAM for current DAG!\n");
        fprintf(stderr, "Required: %.2f GB, Available: %.2f GB\n",
                dagSize / (1024.0 * 1024.0 * 1024.0),
                dev->memory / (1024.0 * 1024.0 * 1024.0));
        return -1;
    }

    // Initialize miner
    printf("Initializing miner...\n");
    g_miner = miner_init(config->deviceId, epoch);
    if (!g_miner) {
        fprintf(stderr, "Failed to initialize miner!\n");
        return -1;
    }

    // Generate DAG
    printf("Generating DAG (this may take a few minutes)...\n");
    time_t dagStart = time(NULL);

    if (miner_generate_dag(g_miner) != 0) {
        fprintf(stderr, "Failed to generate DAG!\n");
        miner_shutdown(g_miner);
        return -1;
    }

    time_t dagEnd = time(NULL);
    printf("DAG generated in %ld seconds\n\n", (long)(dagEnd - dagStart));

    // Create test job
    MiningJob job = {0};
    strcpy(job.jobId, "benchmark");
    job.nTime = (uint32_t)now;
    job.nBits = 0x1e0fffff;  // Easy difficulty for benchmarking
    job.epoch = epoch;
    job.dagSize = dagSize;
    job.target = 0xFFFFFFFFFFFFFFFFULL;  // Accept all hashes

    // Run benchmark for 60 seconds
    printf("Running benchmark for 60 seconds...\n\n");

    time_t startTime = time(NULL);
    uint64_t totalHashes = 0;
    int iterations = 0;

    while (g_running && (time(NULL) - startTime < 60)) {
        miner_submit_job(g_miner, &job);

        // Wait for batch to complete
        MiningResult result;
        while (miner_get_result(g_miner, &result) == 0) {
            msleep(1);
        }

        MinerStats stats;
        miner_get_stats(g_miner, &stats);

        iterations++;
        totalHashes = stats.totalHashes;

        // Print progress every 10 iterations
        if (iterations % 10 == 0) {
            double elapsed = (double)(time(NULL) - startTime);
            double hashrate = totalHashes / elapsed;
            printf("  Hashes: %llu, Rate: %.2f MH/s\n",
                   (unsigned long long)totalHashes, hashrate / 1000000.0);
        }
    }

    double elapsed = (double)(time(NULL) - startTime);
    double hashrate = totalHashes / elapsed;

    printf("\n");
    printf("+=============================================+\n");
    printf("|           BENCHMARK RESULTS                |\n");
    printf("+=============================================+\n");
    printf("| GPU: %-37s |\n", dev->name);
    printf("| Total Hashes: %-28llu |\n", (unsigned long long)totalHashes);
    printf("| Duration: %-32.0f s |\n", elapsed);
    printf("| Hashrate: %-28.2f MH/s |\n", hashrate / 1000000.0);
    printf("+=============================================+\n\n");

    miner_shutdown(g_miner);
    g_miner = NULL;

    return 0;
}

// Main mining loop
int mining_loop(Config *config)
{
    printf("\n");
    printf("+=============================================+\n");
    printf("|        SCRYPT COIN GPU MINER               |\n");
    printf("|         AdaptivePow Algorithm              |\n");
    printf("+=============================================+\n\n");

    // Get GPU info
    GPUDevice devices[8];
    int numDevices = enumerate_gpus(devices, 8);

    if (numDevices <= 0 || config->deviceId >= numDevices) {
        fprintf(stderr, "Invalid GPU device!\n");
        return -1;
    }

    GPUDevice* dev = &devices[config->deviceId];
    printf("GPU: %s (%.2f GB)\n", dev->name, dev->memory / (1024.0 * 1024.0 * 1024.0));

    // Calculate current epoch
    uint64_t now = (uint64_t)time(NULL);
    uint32_t epoch = adaptivepow_get_epoch(now, SCRYPT_GENESIS_TIME);
    uint64_t dagSize = adaptivepow_get_dag_size(epoch);

    printf("Epoch: %u, DAG: %.2f GB\n", epoch, dagSize / (1024.0 * 1024.0 * 1024.0));
    printf("Mode: %s\n", config->solo ? "Solo Mining" : "Pool Mining");
    printf("Target: %s\n\n", config->url);

    // Initialize stratum/RPC connection
    StratumConfig stratumConfig;
    strncpy(stratumConfig.url, config->url, sizeof(stratumConfig.url));
    strncpy(stratumConfig.user, config->user, sizeof(stratumConfig.user));
    strncpy(stratumConfig.pass, config->pass, sizeof(stratumConfig.pass));

    if (stratum_init(&g_stratum, &stratumConfig) != 0) {
        fprintf(stderr, "Failed to initialize connection!\n");
        return -1;
    }

    printf("Connecting to %s...\n", config->url);
    if (stratum_connect(&g_stratum) != 0) {
        fprintf(stderr, "Failed to connect!\n");
        return -1;
    }
    printf("Connected!\n\n");

    // Subscribe and authorize
    if (stratum_subscribe(&g_stratum) != 0 || stratum_authorize(&g_stratum) != 0) {
        fprintf(stderr, "Failed to authenticate!\n");
        return -1;
    }

    // Initialize miner
    printf("Initializing GPU miner...\n");
    g_miner = miner_init(config->deviceId, epoch);
    if (!g_miner) {
        fprintf(stderr, "Failed to initialize miner!\n");
        return -1;
    }

    // Generate DAG
    printf("Generating DAG...\n");
    if (miner_generate_dag(g_miner) != 0) {
        fprintf(stderr, "Failed to generate DAG!\n");
        miner_shutdown(g_miner);
        return -1;
    }
    printf("DAG ready!\n\n");

    // Wait for first job
    printf("Waiting for work...\n");
    while (g_running && !g_stratum.hasJob) {
        stratum_poll(&g_stratum);
        msleep(100);
    }

    if (!g_stratum.hasJob) {
        fprintf(stderr, "No job received!\n");
        miner_shutdown(g_miner);
        return -1;
    }

    printf("Mining started!\n\n");

    // Main mining loop
    time_t lastStatTime = time(NULL);
    uint64_t lastHashes = 0;
    uint32_t currentEpoch = epoch;

    while (g_running) {
        // Check for new jobs
        stratum_poll(&g_stratum);

        // Submit job to GPU
        miner_submit_job(g_miner, &g_stratum.currentJob);

        // Check for results
        MiningResult result;
        if (miner_get_result(g_miner, &result) > 0 && result.found) {
            printf("\n*** SHARE FOUND! Nonce: %016llx ***\n\n",
                   (unsigned long long)result.nonce);
            stratum_submit(&g_stratum, &result);
        }

        // Check for epoch change
        now = (uint64_t)time(NULL);
        uint32_t newEpoch = adaptivepow_get_epoch(now, SCRYPT_GENESIS_TIME);
        if (newEpoch != currentEpoch) {
            printf("\nEpoch changed! Regenerating DAG...\n");
            miner_update_epoch(g_miner, newEpoch);
            currentEpoch = newEpoch;
            printf("New DAG ready!\n\n");
        }

        // Print stats every 30 seconds
        time_t statNow = time(NULL);
        if (statNow - lastStatTime >= 30) {
            MinerStats stats;
            miner_get_stats(g_miner, &stats);

            double hashrate = (stats.totalHashes - lastHashes) / (double)(statNow - lastStatTime);

            printf("[%s] %.2f MH/s | Shares: %llu/%llu | GPU: %.0fC\n",
                   dev->name,
                   hashrate / 1000000.0,
                   (unsigned long long)stats.acceptedShares,
                   (unsigned long long)(stats.acceptedShares + stats.rejectedShares),
                   stats.gpuTemp);

            lastStatTime = statNow;
            lastHashes = stats.totalHashes;
        }

        msleep(10);
    }

    // Cleanup
    miner_shutdown(g_miner);
    stratum_cleanup(&g_stratum);
    g_miner = NULL;

    return 0;
}

int main(int argc, char **argv)
{
    printf("\n");
    printf("  ____                       _      ____      _       \n");
    printf(" / ___|  ___ _ __ _   _ _ __ | |_   / ___|___ (_)_ __  \n");
    printf(" \\___ \\ / __| '__| | | | '_ \\| __| | |   / _ \\| | '_ \\ \n");
    printf("  ___) | (__| |  | |_| | |_) | |_  | |__| (_) | | | | |\n");
    printf(" |____/ \\___|_|   \\__, | .__/ \\__|  \\____\\___/|_|_| |_|\n");
    printf("                  |___/|_|                             \n");
    printf("\n");
    printf("GPU Miner v%s - AdaptivePow Algorithm\n", SCRYPT_MINER_VERSION);
    printf("\n");

    // Parse arguments
    Config config;
    if (parse_args(argc, argv, &config) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    int result = 0;

    if (config.listDevices) {
        result = list_devices();
    }
    else if (config.benchmark) {
        result = run_benchmark(&config);
    }
    else if (strlen(config.url) == 0) {
        fprintf(stderr, "Error: URL required (-o)\n");
        fprintf(stderr, "Use --help for usage information\n");
        result = 1;
    }
    else if (strlen(config.user) == 0) {
        fprintf(stderr, "Error: Username/wallet required (-u)\n");
        fprintf(stderr, "Use --help for usage information\n");
        result = 1;
    }
    else if (config.solo && strlen(config.address) == 0) {
        fprintf(stderr, "Error: Payout address required for solo mining (--address)\n");
        fprintf(stderr, "Use --help for usage information\n");
        result = 1;
    }
    else {
        result = mining_loop(&config);
    }

    // Final cleanup
    if (g_miner) {
        miner_shutdown(g_miner);
    }

    printf("\nGoodbye!\n");
    return result;
}
