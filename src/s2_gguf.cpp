#include "../include/s2_gguf.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <stdexcept>
#ifdef __linux__
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace s2 {

std::unique_ptr<GGUFLoader> GGUFLoader::load(const std::string & gguf_path) {
    ggml_context * ctx_all_tensors = nullptr;
    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&ctx_all_tensors };
    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "[GGUFLoader] Failed to load GGUF from " << gguf_path << std::endl;
        return nullptr;
    }
    if (!ctx_all_tensors) {
        std::cerr << "[GGUFLoader] GGUF did not create ggml context" << std::endl;
        gguf_free(ctx_gguf);
        return nullptr;
    }
    return std::unique_ptr<GGUFLoader>(new GGUFLoader(ctx_gguf, ctx_all_tensors, gguf_path));
}

GGUFLoader::GGUFLoader(gguf_context * ctx_gguf, ggml_context * ctx_all_tensors, const std::string & path)
    : ctx_gguf_(ctx_gguf), ctx_all_tensors_(ctx_all_tensors), gguf_path_(path) {}

GGUFLoader::~GGUFLoader() {
    if (ctx_gguf_) {
        gguf_free(ctx_gguf_);
    }
    if (ctx_all_tensors_) {
        ggml_free(ctx_all_tensors_);
    }
}

int32_t GGUFLoader::get_i32(const std::string & key, int32_t def) const {
    int id = static_cast<int>(gguf_find_key(ctx_gguf_, key.c_str()));
    if (id < 0) return def;
    return static_cast<int32_t>(gguf_get_val_u32(ctx_gguf_, id));
}

uint32_t GGUFLoader::get_u32(const std::string & key, uint32_t def) const {
    int id = static_cast<int>(gguf_find_key(ctx_gguf_, key.c_str()));
    if (id < 0) return def;
    return gguf_get_val_u32(ctx_gguf_, id);
}

float GGUFLoader::get_f32(const std::string & key, float def) const {
    int id = static_cast<int>(gguf_find_key(ctx_gguf_, key.c_str()));
    if (id < 0) return def;
    return gguf_get_val_f32(ctx_gguf_, id);
}

bool GGUFLoader::get_bool(const std::string & key, bool def) const {
    int id = static_cast<int>(gguf_find_key(ctx_gguf_, key.c_str()));
    if (id < 0) return def;
    return gguf_get_val_bool(ctx_gguf_, id);
}

std::string GGUFLoader::get_string(const std::string & key, const std::string & def) const {
    int id = static_cast<int>(gguf_find_key(ctx_gguf_, key.c_str()));
    if (id < 0) return def;
    const char * str = gguf_get_val_str(ctx_gguf_, id);
    return str ? std::string(str) : def;
}

std::vector<uint32_t> GGUFLoader::get_u32_array(const std::string & key) const {
    std::vector<uint32_t> result;
    int id = static_cast<int>(gguf_find_key(ctx_gguf_, key.c_str()));
    if (id < 0) return result;
    const uint32_t * arr = static_cast<const uint32_t*>(gguf_get_arr_data(ctx_gguf_, id));
    const uint32_t len = static_cast<uint32_t>(gguf_get_arr_n(ctx_gguf_, id));
    result.assign(arr, arr + len);
    return result;
}

GGUFLoader::FilteredContext GGUFLoader::create_filtered_context(
        const std::string & prefix, ggml_backend_t backend) const {
    FilteredContext fc;
    
    // Count tensors matching prefix and compute context size
    size_t ctx_size = 0;
    int64_t filtered_count = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_all_tensors_); t != nullptr; t = ggml_get_next_tensor(ctx_all_tensors_, t)) {
        const char * name = ggml_get_name(t);
        if (!name) continue;
        if (!prefix.empty() && std::strncmp(name, prefix.c_str(), prefix.length()) != 0) {
            continue;
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
    fc.ctx = ggml_init(params);
    if (!fc.ctx) {
        std::cerr << "[GGUFLoader] Failed to init ggml context for prefix '" << prefix << "'" << std::endl;
        return fc;
    }
    
    // Duplicate matching tensors
    for (ggml_tensor * src = ggml_get_first_tensor(ctx_all_tensors_); src != nullptr; src = ggml_get_next_tensor(ctx_all_tensors_, src)) {
        const char * name = ggml_get_name(src);
        if (!name) continue;
        if (!prefix.empty() && std::strncmp(name, prefix.c_str(), prefix.length()) != 0) {
            continue;
        }
        ggml_tensor * dst = ggml_dup_tensor(fc.ctx, src);
        if (!dst) {
            std::cerr << "[GGUFLoader] Failed to duplicate tensor: " << name << std::endl;
            fc.tensors.clear();
            ggml_free(fc.ctx);
            fc.ctx = nullptr;
            return fc;
        }
        ggml_set_name(dst, name);
        fc.tensors.push_back(dst);
    }
    
    if (backend) {
        fc.buf = ggml_backend_alloc_ctx_tensors(fc.ctx, backend);
        if (!fc.buf) {
            std::cerr << "[GGUFLoader] Failed to allocate backend buffer for prefix '" << prefix << "'" << std::endl;
            fc.tensors.clear();
            ggml_free(fc.ctx);
            fc.ctx = nullptr;
            return fc;
        }
    }
    
    std::cout << "[GGUFLoader] Created filtered context with " << filtered_count
              << " tensors for prefix '" << prefix << "'" << std::endl;
    return fc;
}

bool GGUFLoader::load_tensor_data(ggml_backend_buffer_t buf, ggml_context * ctx) const {
    if (!buf || !ctx) return false;
    
    const size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf_);
    
    std::FILE* f = nullptr;
#ifdef _WIN32
    errno_t err = fopen_s(&f, gguf_path_.c_str(), "rb");
    if (err != 0) f = nullptr;
#else
    f = std::fopen(gguf_path_.c_str(), "rb");
#endif
    if (!f) {
        std::cerr << "[GGUFLoader] Cannot reopen " << gguf_path_ << " for data loading." << std::endl;
        return false;
    }
    
    std::vector<uint8_t> tmp;
    bool success = true;
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(ctx_gguf_, ti);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) continue; // tensor not in this context
        const size_t off   = data_offset + gguf_get_tensor_offset(ctx_gguf_, ti);
        const size_t nbytes = ggml_nbytes(t);
        if (tmp.size() < nbytes) tmp.resize(nbytes);
#ifdef _WIN32
        _fseeki64(f, (int64_t)off, SEEK_SET);
#else
        fseeko(f, (off_t)off, SEEK_SET);
#endif
        if (std::fread(tmp.data(), 1, nbytes, f) != nbytes) {
            std::cerr << "[GGUFLoader] Failed to read tensor: " << name << std::endl;
            success = false;
            break;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
    }
    tmp.clear();
    tmp.shrink_to_fit();
    std::fclose(f);
    
    // Advise kernel to drop file pages from page cache (Linux only)
#ifdef __linux__
    {
        int fd = ::open(gguf_path_.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }
#endif
    
    return success;
}

} // namespace s2