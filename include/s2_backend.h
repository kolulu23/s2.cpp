#pragma once
// s2_backend.h — GPU/CPU backend initialization utilities
//
// Provides centralized backend management for Vulkan, CUDA, and CPU backends,
// reducing code duplication and improving memory safety.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <string>

namespace s2 {

class BackendManager {
public:
    // Initialize or reuse a backend.
    // If 'existing' is not null and matches requested type, it may be reused.
    // Returns nullptr on failure.
    // gpu_device: -1 for CPU only, >=0 for GPU device index
    // backend_type: 0 = Vulkan, 1 = CUDA, -1 = auto (try Vulkan, then CUDA, then CPU)
    static ggml_backend_t init_backend(int32_t gpu_device, int32_t backend_type,
                                       ggml_backend_t existing = nullptr);
    
    // Check if a backend is a GPU backend (Vulkan or CUDA)
    static bool is_gpu_backend(ggml_backend_t backend);
    
    // Get backend type as string for logging
    static std::string get_backend_name(ggml_backend_t backend);
    
    // Safely free a backend if non-null
    static void free_backend(ggml_backend_t backend);
    
    // Create allocators for a backend
    static ggml_gallocr_t create_allocator(ggml_backend_t backend);
    
    // Check if backend supports a specific device type
    static bool supports_backend_type(int32_t backend_type);
    
private:
    BackendManager() = delete;
    ~BackendManager() = delete;
    
    // Internal initialization helpers
    static ggml_backend_t init_vulkan(int32_t gpu_device);
    static ggml_backend_t init_cuda(int32_t gpu_device);
    static ggml_backend_t init_cpu();
};

} // namespace s2