#include "../include/s2_tensor_loader.h"
#include <iostream>

namespace s2 {

// ---------------------------------------------------------------------------
// TensorContext implementation
// ---------------------------------------------------------------------------

TensorContext::TensorContext() = default;

TensorContext::~TensorContext() {
    clear();
}

TensorContext::TensorContext(TensorContext&& other) noexcept
    : ctx_(other.ctx_), buf_(other.buf_) {
    other.ctx_ = nullptr;
    other.buf_ = nullptr;
}

TensorContext& TensorContext::operator=(TensorContext&& other) noexcept {
    if (this != &other) {
        clear();
        ctx_ = other.ctx_;
        buf_ = other.buf_;
        other.ctx_ = nullptr;
        other.buf_ = nullptr;
    }
    return *this;
}

void TensorContext::init(ggml_context* ctx, ggml_backend_buffer_t buf) {
    clear();
    ctx_ = ctx;
    buf_ = buf;
}

ggml_context* TensorContext::release_context() {
    ggml_context* ctx = ctx_;
    ctx_ = nullptr;
    return ctx;
}

ggml_backend_buffer_t TensorContext::release_buffer() {
    ggml_backend_buffer_t buf = buf_;
    buf_ = nullptr;
    return buf;
}

void TensorContext::clear() {
    if (buf_) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// TensorLoader implementation
// ---------------------------------------------------------------------------

TensorContext TensorLoader::load_from_file(const std::string& gguf_path,
                                           ggml_backend_t backend,
                                           const std::string& prefix) {
    // Load GGUF file
    auto loader = GGUFLoader::load(gguf_path);
    if (!loader) {
        std::cerr << "[TensorLoader] Failed to load GGUF from " << gguf_path << std::endl;
        return TensorContext();
    }
    
    // Create filtered context
    GGUFLoader::FilteredContext fc = loader->create_filtered_context(prefix, backend);
    if (!fc.ctx || !fc.buf) {
        std::cerr << "[TensorLoader] Failed to create filtered context for prefix '" << prefix << "'" << std::endl;
        return TensorContext();
    }
    
    // Load tensor data
    if (!loader->load_tensor_data(fc.buf, fc.ctx)) {
        std::cerr << "[TensorLoader] Failed to load tensor data" << std::endl;
        // Clean up
        if (fc.buf) ggml_backend_buffer_free(fc.buf);
        if (fc.ctx) ggml_free(fc.ctx);
        return TensorContext();
    }
    
    // Create TensorContext
    TensorContext result;
    result.init(fc.ctx, fc.buf);
    
    // Log memory usage
    log_tensor_memory(fc.ctx, "Loaded from file");
    
    return result;
}

TensorContext TensorLoader::load_excluding_prefix(const std::string& gguf_path,
                                                  ggml_backend_t backend,
                                                  const std::string& exclude_prefix) {
    // Load GGUF file
    auto loader = GGUFLoader::load(gguf_path);
    if (!loader) {
        std::cerr << "[TensorLoader] Failed to load GGUF from " << gguf_path << std::endl;
        return TensorContext();
    }
    return load_excluding_prefix(*loader, backend, exclude_prefix);
}

TensorContext TensorLoader::load_from_loader(GGUFLoader& loader,
                                             ggml_backend_t backend,
                                             const std::string& prefix) {
    // Create filtered context
    GGUFLoader::FilteredContext fc = loader.create_filtered_context(prefix, backend);
    if (!fc.ctx || !fc.buf) {
        std::cerr << "[TensorLoader] Failed to create filtered context for prefix '" << prefix << "'" << std::endl;
        return TensorContext();
    }
    
    // Load tensor data
    if (!loader.load_tensor_data(fc.buf, fc.ctx)) {
        std::cerr << "[TensorLoader] Failed to load tensor data from loader" << std::endl;
        // Clean up
        if (fc.buf) ggml_backend_buffer_free(fc.buf);
        if (fc.ctx) ggml_free(fc.ctx);
        return TensorContext();
    }
    
    // Create TensorContext
    TensorContext result;
    result.init(fc.ctx, fc.buf);
    
    // Log memory usage
    log_tensor_memory(fc.ctx, "Loaded from loader");
    
    return result;
}

TensorContext TensorLoader::load_excluding_prefix(GGUFLoader& loader,
                                                  ggml_backend_t backend,
                                                  const std::string& exclude_prefix) {
    ggml_context* src_ctx = loader.get_ggml_context();
    if (!src_ctx) {
        std::cerr << "[TensorLoader] GGUFLoader has no shared ggml context." << std::endl;
        return TensorContext();
    }
    
    // Count tensors not starting with exclude_prefix and compute context size
    size_t ctx_size = 0;
    int64_t filtered_count = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(src_ctx); t != nullptr; t = ggml_get_next_tensor(src_ctx, t)) {
        const char* name = ggml_get_name(t);
        if (!name) continue;
        if (!exclude_prefix.empty() && std::strncmp(name, exclude_prefix.c_str(), exclude_prefix.length()) == 0) {
            continue; // skip excluded tensors
        }
        filtered_count++;
        ctx_size += ggml_tensor_overhead();
    }
    // Add extra for context internal structures
    ctx_size += (1ull << 20); // 1 MB
    
    // Create new context with no_alloc = true
    ggml_init_params params = {
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ggml_context* dst_ctx = ggml_init(params);
    if (!dst_ctx) {
        std::cerr << "[TensorLoader] Failed to init ggml context for exclusion prefix '" << exclude_prefix << "'" << std::endl;
        return TensorContext();
    }
    
    // Duplicate matching tensors
    for (ggml_tensor* src = ggml_get_first_tensor(src_ctx); src != nullptr; src = ggml_get_next_tensor(src_ctx, src)) {
        const char* name = ggml_get_name(src);
        if (!name) continue;
        if (!exclude_prefix.empty() && std::strncmp(name, exclude_prefix.c_str(), exclude_prefix.length()) == 0) {
            continue; // skip excluded tensors
        }
        ggml_tensor* dst = ggml_dup_tensor(dst_ctx, src);
        if (!dst) {
            std::cerr << "[TensorLoader] Failed to duplicate tensor: " << name << std::endl;
            ggml_free(dst_ctx);
            return TensorContext();
        }
        ggml_set_name(dst, name);
    }
    
    // Allocate backend buffer
    ggml_backend_buffer_t buf = nullptr;
    if (backend) {
        buf = ggml_backend_alloc_ctx_tensors(dst_ctx, backend);
        if (!buf) {
            std::cerr << "[TensorLoader] Failed to allocate backend buffer for exclusion prefix '" << exclude_prefix << "'" << std::endl;
            ggml_free(dst_ctx);
            return TensorContext();
        }
    }
    
    std::cout << "[TensorLoader] Created filtered context with " << filtered_count
              << " tensors (excluding '" << exclude_prefix << "')" << std::endl;
    
    // Load tensor data
    if (!loader.load_tensor_data(buf, dst_ctx)) {
        std::cerr << "[TensorLoader] Failed to load tensor data from loader (exclusion)" << std::endl;
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(dst_ctx);
        return TensorContext();
    }
    
    // Create TensorContext
    TensorContext result;
    result.init(dst_ctx, buf);
    
    // Log memory usage
    log_tensor_memory(dst_ctx, "Loaded with exclusion");
    
    return result;
}

void TensorLoader::log_tensor_memory(ggml_context* ctx, const std::string& tag) {
    if (!ctx) return;
    
    size_t total_bytes = 0;
    int64_t num_tensors = 0;
    
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        total_bytes += ggml_nbytes(t);
        num_tensors++;
    }
    
    std::cout << "[TensorLoader] " << tag << ": " << num_tensors 
              << " tensors, " << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
}

size_t TensorLoader::get_total_tensor_bytes(ggml_context* ctx) {
    if (!ctx) return 0;
    
    size_t total_bytes = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        total_bytes += ggml_nbytes(t);
    }
    return total_bytes;
}

int64_t TensorLoader::get_tensor_count(ggml_context* ctx) {
    if (!ctx) return 0;
    
    int64_t count = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        count++;
    }
    return count;
}

} // namespace s2