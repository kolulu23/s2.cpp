#pragma once
// s2_tensor_loader.h — Tensor loading utilities
//
// Provides RAII wrappers for tensor contexts and buffers, simplifying
// memory management and error handling.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "s2_gguf.h"
#include <string>
#include <memory>

namespace s2 {

// RAII wrapper for a tensor context + buffer pair
class TensorContext {
public:
    TensorContext();
    ~TensorContext();
    
    // Move-only
    TensorContext(TensorContext&& other) noexcept;
    TensorContext& operator=(TensorContext&& other) noexcept;
    TensorContext(const TensorContext&) = delete;
    TensorContext& operator=(const TensorContext&) = delete;
    
    // Initialize from existing resources (takes ownership)
    void init(ggml_context* ctx, ggml_backend_buffer_t buf);
    
    // Check if valid
    bool valid() const { return ctx_ != nullptr; }
    
    // Get raw pointers (caller does NOT own them)
    ggml_context* context() const { return ctx_; }
    ggml_backend_buffer_t buffer() const { return buf_; }
    
    // Release ownership (caller must free)
    ggml_context* release_context();
    ggml_backend_buffer_t release_buffer();
    
    // Free resources
    void clear();
    
private:
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
};

// High-level tensor loader
class TensorLoader {
public:
    // Load tensors from GGUF file
    // prefix: filter tensors by name prefix (e.g., "c." for codec)
    // returns TensorContext on success, empty on failure
    static TensorContext load_from_file(const std::string& gguf_path,
                                        ggml_backend_t backend,
                                        const std::string& prefix = "");
    
    // Load tensors from GGUF file, excluding those with given prefix
    static TensorContext load_excluding_prefix(const std::string& gguf_path,
                                               ggml_backend_t backend,
                                               const std::string& exclude_prefix);
    
    // Load tensors from GGUFLoader (shared loader)
    static TensorContext load_from_loader(GGUFLoader& loader,
                                          ggml_backend_t backend,
                                          const std::string& prefix = "");
    
    // Load tensors from GGUFLoader, excluding those with given prefix
    static TensorContext load_excluding_prefix(GGUFLoader& loader,
                                               ggml_backend_t backend,
                                               const std::string& exclude_prefix);
    
    // Log tensor memory usage
    static void log_tensor_memory(ggml_context* ctx, const std::string& tag);
    
    // Get total tensor bytes in context
    static size_t get_total_tensor_bytes(ggml_context* ctx);
    
    // Get number of tensors in context
    static int64_t get_tensor_count(ggml_context* ctx);
};

} // namespace s2