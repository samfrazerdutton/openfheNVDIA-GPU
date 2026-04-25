// vram_cache.cpp — Phase 3 persistent VRAM cache implementation.
// Actual caching logic lives in ShadowRegistry (include/shadow_registry.h).
// This file satisfies the CMakeLists.txt target_sources reference.
#include "shadow_registry.h"

namespace {
    struct VramCacheInit {
        VramCacheInit() { ShadowRegistry::Instance(); }
    };
    static VramCacheInit g_init;
}
