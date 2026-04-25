// gpu_keyswitch.cpp — GPU hybrid key-switch host-side orchestration.
#include "cuda_hal.h"
#include "stream_pool.h"
#include "shadow_registry.h"
#include <cstdint>
#include <cstdio>

extern "C" void gpu_keyswitch_sync() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}
