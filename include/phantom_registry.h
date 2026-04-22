#pragma once
#include <unordered_map>
#include <cuda_runtime.h>
#include <iostream>

// An extension to ShadowRegistry for DAG variables
class PhantomRegistry {
private:
    // Maps a dummy CPU pointer to actual VRAM
    std::unordered_map<void*, void*> phantom_map;

public:
    // Allocates memory ONLY on the GPU, returning a dummy pointer for OpenFHE
    void* AllocatePhantom(size_t size_bytes) {
        void* vram_ptr = nullptr;
        cudaMalloc(&vram_ptr, size_bytes);
        
        // Generate a dummy host pointer (using VRAM address space but just as a key)
        void* dummy_host_key = vram_ptr; 
        
        phantom_map[dummy_host_key] = vram_ptr;
        return dummy_host_key;
    }

    void* GetVramPointer(void* dummy_host_key) {
        if (phantom_map.find(dummy_host_key) != phantom_map.end()) {
            return phantom_map[dummy_host_key];
        }
        return nullptr;
    }

    void FreePhantom(void* dummy_host_key) {
        if (phantom_map.find(dummy_host_key) != phantom_map.end()) {
            cudaFree(phantom_map[dummy_host_key]);
            phantom_map.erase(dummy_host_key);
        }
    }
};
