#pragma once

#include <vector>
#include <cstdint>

// Forward declaration for NVIDIA NCCL (Nickel)
struct ncclComm;

namespace hpc_fhe {

/**
 * @brief Phase 3 & 4: Multi-GPU RNS Tower Distributor
 * Maps independent FHE Residue Number System (RNS) towers across 
 * physical DGX/NVLink nodes to achieve linear scaling.
 */
class GPUClusterManager {
private:
    int num_gpus;
    std::vector<ncclComm*> communicators;

public:
    GPUClusterManager();
    ~GPUClusterManager();

    // Initializes peer-to-peer memory access via NVLink
    void InitNVLink();

    // Distributes a 16-tower polynomial across 4 GPUs (4 towers each)
    void DistributeRNSTowers(uint64_t* host_poly, int total_towers);

    // Phase 4: Triggers shared-memory optimized NTTs across the cluster
    void ExecuteDistributedNTT();
};

} // namespace hpc_fhe
