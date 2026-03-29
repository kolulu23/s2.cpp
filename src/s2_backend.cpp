#include "../include/s2_backend.h"
#include <iostream>
#include <string>

namespace s2 {

ggml_backend_t BackendManager::init_backend(int32_t gpu_device, int32_t backend_type,
                                           ggml_backend_t existing) {
    // If existing backend is provided and matches requested type, reuse it
    if (existing) {
        // For now, we don't check type compatibility - caller should know
        // This matches the pattern in load() functions: if (!backend_) init
        return existing;
    }
    
    ggml_backend_t backend = nullptr;
    
    if (gpu_device >= 0) {
        // Try GPU backends based on backend_type
#ifdef GGML_USE_VULKAN
        if (backend_type == 0 || backend_type == -1) {
            backend = ggml_backend_vk_init(static_cast<size_t>(gpu_device));
            if (backend) {
                std::cout << "[Backend] Using Vulkan (device " << gpu_device << ")" << std::endl;
                return backend;
            } else if (backend_type == 0) {
                std::cerr << "[Backend] Vulkan init failed." << std::endl;
            }
        }
#endif
#ifdef GGML_USE_CUDA
        if (backend_type == 1 || backend_type == -1) {
            backend = ggml_backend_cuda_init(static_cast<size_t>(gpu_device));
            if (backend) {
                std::cout << "[Backend] Using CUDA (device " << gpu_device << ")" << std::endl;
                return backend;
            } else if (backend_type == 1) {
                std::cerr << "[Backend] CUDA init failed." << std::endl;
            }
        }
#endif
        if (!backend && backend_type != -1) {
            std::cerr << "[Backend] Requested GPU backend not available, falling back to CPU." << std::endl;
        }
    }
    
    // Fall back to CPU
    backend = ggml_backend_cpu_init();
    if (backend) {
        std::cout << "[Backend] Using CPU" << std::endl;
    } else {
        std::cerr << "[Backend] Failed to init CPU backend." << std::endl;
    }
    
    return backend;
}

bool BackendManager::is_gpu_backend(ggml_backend_t backend) {
    if (!backend) return false;
    std::string name = get_backend_name(backend);
    return name.find("Vulkan") != std::string::npos ||
           name.find("CUDA") != std::string::npos;
}

std::string BackendManager::get_backend_name(ggml_backend_t backend) {
    if (!backend) return "null";
    const char* name = ggml_backend_name(backend);
    return name ? std::string(name) : "unknown";
}

void BackendManager::free_backend(ggml_backend_t backend) {
    if (backend) {
        ggml_backend_free(backend);
    }
}

ggml_gallocr_t BackendManager::create_allocator(ggml_backend_t backend) {
    if (!backend) return nullptr;
    return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
}

bool BackendManager::supports_backend_type(int32_t backend_type) {
    switch (backend_type) {
        case 0: // Vulkan
#ifdef GGML_USE_VULKAN
            return true;
#else
            return false;
#endif
        case 1: // CUDA
#ifdef GGML_USE_CUDA
            return true;
#else
            return false;
#endif
        case -1: // CPU (always supported)
            return true;
        default:
            return false;
    }
}

// Internal implementation helpers (not exposed in header)
ggml_backend_t BackendManager::init_vulkan(int32_t gpu_device) {
#ifdef GGML_USE_VULKAN
    return ggml_backend_vk_init(static_cast<size_t>(gpu_device));
#else
    (void)gpu_device;
    return nullptr;
#endif
}

ggml_backend_t BackendManager::init_cuda(int32_t gpu_device) {
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_init(static_cast<size_t>(gpu_device));
#else
    (void)gpu_device;
    return nullptr;
#endif
}

ggml_backend_t BackendManager::init_cpu() {
    return ggml_backend_cpu_init();
}

} // namespace s2