#pragma once
// VRAMPool has been superseded by ShadowRegistry (include/shadow_registry.h).
//
// ShadowRegistry provides the same VRAM lifetime management with better
// semantics: it ties device allocations to host pointer identities, which
// means the registry automatically handles the case where OpenFHE reuses
// the same polynomial backing buffer across multiple EvalMult calls without
// redundant cudaMalloc / cudaFree overhead.
//
// stream_pool.h (openfhe_cuda::StreamPool) is still used for stream management.
//
// This header is kept as a tombstone so any external code that includes it
// gets a clear error message rather than a silent compile failure.
#error "vram_pool.h is deprecated. Include shadow_registry.h instead."
