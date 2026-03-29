#pragma once
// s2_gguf.h — Shared GGUF loader for model and codec tensors
//
// Loads a GGUF file once and allows creating filtered ggml contexts
// containing only tensors with a given prefix, reducing memory duplication.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <string>
#include <memory>
#include <vector>

namespace s2 {

class GGUFLoader {
public:
    // Load a GGUF file. Returns nullptr on failure.
    static std::unique_ptr<GGUFLoader> load(const std::string & gguf_path);
    
    ~GGUFLoader();
    
    // Get metadata value (similar to gguf_get_val_*)
    int32_t get_i32(const std::string & key, int32_t def = 0) const;
    uint32_t get_u32(const std::string & key, uint32_t def = 0) const;
    float get_f32(const std::string & key, float def = 0.0f) const;
    bool get_bool(const std::string & key, bool def = false) const;
    std::string get_string(const std::string & key, const std::string & def = "") const;
    std::vector<uint32_t> get_u32_array(const std::string & key) const;
    
    // Create a ggml context containing only tensors whose names start with prefix.
    // If prefix is empty, includes all tensors.
    // The context is initialized with no_alloc = true.
    // Caller is responsible for freeing the context (ggml_free) and the returned buffer.
    struct FilteredContext {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr; // allocated backend buffer (if backend provided)
        std::vector<ggml_tensor*> tensors; // pointers to tensors in ctx
    };
    
    // Create filtered context and allocate backend buffer for its tensors.
    // If backend is nullptr, buf will be nullptr (no allocation).
    FilteredContext create_filtered_context(const std::string & prefix, ggml_backend_t backend) const;
    
    // Load tensor data into an already allocated backend buffer.
    // The buffer must have been allocated for the tensors in the context.
    bool load_tensor_data(ggml_backend_buffer_t buf, ggml_context * ctx) const;
    
    // Get the original gguf_context (for advanced use)
    gguf_context * get_gguf_context() const { return ctx_gguf_; }
    
    // Get the shared ggml_context containing all tensor metadata (no data allocated)
    ggml_context * get_ggml_context() const { return ctx_all_tensors_; }
    
private:
    GGUFLoader(gguf_context * ctx_gguf, ggml_context * ctx_all_tensors, const std::string & path);
    
    gguf_context * ctx_gguf_;
    ggml_context * ctx_all_tensors_;
    std::string gguf_path_;
};

} // namespace s2