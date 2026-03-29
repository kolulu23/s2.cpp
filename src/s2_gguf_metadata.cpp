#include "../include/s2_gguf_metadata.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace s2 {

// ---------------------------------------------------------------------------
// ModelMetadata reading
// ---------------------------------------------------------------------------

ModelMetadata GGUFMetadata::read_model_metadata(gguf_context* ctx) {
    ModelMetadata meta;
    
    if (!ctx) {
        throw std::runtime_error("GGUFMetadata: null context");
    }
    
    // Get architecture
    meta.architecture = get_string(ctx, "general.architecture", "fish-speech");
    meta.arch_prefix = meta.architecture + ".";
    meta.has_fast_decoder = (meta.architecture == "fish-speech");
    
    // Main model hyperparameters (with architecture prefix)
    meta.context_length = static_cast<int32_t>(get_u32(ctx, (meta.arch_prefix + "context_length").c_str(), 32768));
    meta.vocab_size = static_cast<int32_t>(get_u32(ctx, (meta.arch_prefix + "vocab_size").c_str(), 155776));
    meta.embedding_length = static_cast<int32_t>(get_u32(ctx, (meta.arch_prefix + "embedding_length").c_str(), 2560));
    meta.feed_forward_length = static_cast<int32_t>(get_u32(ctx, (meta.arch_prefix + "feed_forward_length").c_str(), 9728));
    meta.block_count = static_cast<int32_t>(get_u32(ctx, (meta.arch_prefix + "block_count").c_str(), 36));
    meta.head_count = static_cast<int32_t>(get_u32(ctx, (meta.arch_prefix + "attention.head_count").c_str(), 32));
    meta.head_count_kv = static_cast<int32_t>(get_u32(ctx, (meta.arch_prefix + "attention.head_count_kv").c_str(), 8));
    meta.rope_freq_base = get_f32(ctx, (meta.arch_prefix + "rope.freq_base").c_str(), 1e6f);
    meta.rms_norm_eps = get_f32(ctx, (meta.arch_prefix + "attention.layer_norm_rms_epsilon").c_str(), 1e-6f);
    
    // Fish-speech specific keys
    meta.codebook_size = static_cast<int32_t>(get_u32(ctx, "fish_speech.codebook_size", 4096));
    meta.num_codebooks = static_cast<int32_t>(get_u32(ctx, "fish_speech.num_codebooks", 10));
    meta.semantic_begin_id = static_cast<int32_t>(get_u32(ctx, "fish_speech.semantic_begin_id", 151678));
    meta.semantic_end_id = static_cast<int32_t>(get_u32(ctx, "fish_speech.semantic_end_id", 155773));
    meta.tie_word_embeddings = get_bool(ctx, "fish_speech.tie_word_embeddings", true);
    meta.attention_qk_norm = get_bool(ctx, "fish_speech.attention_qk_norm", false);
    meta.scale_codebook_embeddings = get_bool(ctx, "fish_speech.scale_codebook_embeddings", false);
    
    // Fast decoder hyperparameters
    if (meta.has_fast_decoder) {
        meta.fast_context_length = static_cast<int32_t>(get_u32(ctx, "fish_speech.fast_context_length", 11));
        meta.fast_embedding_length = static_cast<int32_t>(get_u32(ctx, "fish_speech.fast_embedding_length", 2560));
        meta.fast_feed_forward_length = static_cast<int32_t>(get_u32(ctx, "fish_speech.fast_feed_forward_length", 9728));
        meta.fast_block_count = static_cast<int32_t>(get_u32(ctx, "fish_speech.fast_block_count", 4));
        meta.fast_head_count = static_cast<int32_t>(get_u32(ctx, "fish_speech.fast_head_count", 32));
        meta.fast_head_count_kv = static_cast<int32_t>(get_u32(ctx, "fish_speech.fast_head_count_kv", 8));
        meta.fast_head_dim = static_cast<int32_t>(get_u32(ctx, "fish_speech.fast_head_dim", 128));
        meta.fast_rope_freq_base = get_f32(ctx, "fish_speech.fast_rope_freq_base", 1e6f);
        meta.fast_rms_norm_eps = get_f32(ctx, "fish_speech.fast_layer_norm_rms_eps", 1e-6f);
        meta.fast_attention_qk_norm = get_bool(ctx, "fish_speech.fast_attention_qk_norm", false);
        meta.fast_has_project_in = get_bool(ctx, "fish_speech.fast_project_in", false);
    }
    
    return meta;
}

// ---------------------------------------------------------------------------
// CodecMetadata reading
// ---------------------------------------------------------------------------

CodecMetadata GGUFMetadata::read_codec_metadata(gguf_context* ctx) {
    CodecMetadata meta;
    
    if (!ctx) {
        throw std::runtime_error("GGUFMetadata: null context");
    }
    
    // Determine architecture and tensor prefix
    std::string arch = require_string(ctx, "general.architecture");
    if (arch == "fish-speech") {
        meta.tensor_prefix = "c.";
    } else if (arch == "fish-speech-codec") {
        meta.tensor_prefix = "";
    } else {
        throw std::runtime_error("GGUFMetadata: unexpected architecture: " + arch);
    }
    
    // Basic codec parameters
    meta.sample_rate = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.sample_rate"));
    meta.hop_length = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.hop_length"));
    meta.frame_length = static_cast<int32_t>(get_u32(ctx, "fish_speech.codec.frame_length", 512));
    
    // Encoder/decoder dimensions
    meta.encoder_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.encoder_dim"));
    meta.decoder_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.decoder_dim"));
    meta.latent_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.latent_dim"));
    
    // Arrays
    meta.encoder_rates = require_i32_array(ctx, "fish_speech.codec.encoder_rates");
    meta.decoder_rates = require_i32_array(ctx, "fish_speech.codec.decoder_rates");
    meta.encoder_transformer_layers = require_i32_array(ctx, "fish_speech.codec.encoder_transformer_layers");
    
    // Quantizer parameters
    meta.quantizer_input_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.quantizer_input_dim"));
    meta.quantizer_codebook_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.quantizer_codebook_dim"));
    meta.quantizer_residual_codebooks = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.quantizer_residual_codebooks"));
    meta.quantizer_residual_codebook_size = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.quantizer_residual_codebook_size"));
    meta.quantizer_semantic_codebook_size = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.quantizer_semantic_codebook_size"));
    meta.quantizer_downsample_factor = require_i32_array(ctx, "fish_speech.codec.quantizer_downsample_factor");
    
    // Transformer parameters
    meta.transformer_block_size = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.transformer.block_size"));
    meta.transformer_n_local_heads = require_i32(ctx, "fish_speech.codec.transformer.n_local_heads");
    meta.transformer_head_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.transformer.head_dim"));
    meta.transformer_rope_base = require_f32(ctx, "fish_speech.codec.transformer.rope_freq_base");
    meta.transformer_norm_eps = require_f32(ctx, "fish_speech.codec.transformer.layer_norm_rms_eps");
    
    // RVQ transformer parameters
    meta.rvq_transformer_window_size = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.rvq_transformer.window_size"));
    meta.rvq_transformer_block_size = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.rvq_transformer.block_size"));
    meta.rvq_transformer_n_layer = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.rvq_transformer.n_layer"));
    meta.rvq_transformer_n_local_heads = require_i32(ctx, "fish_speech.codec.rvq_transformer.n_local_heads");
    meta.rvq_transformer_head_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.rvq_transformer.head_dim"));
    meta.rvq_transformer_dim = static_cast<int32_t>(require_u32(ctx, "fish_speech.codec.rvq_transformer.dim"));
    meta.rvq_transformer_rope_base = require_f32(ctx, "fish_speech.codec.rvq_transformer.rope_freq_base");
    meta.rvq_transformer_norm_eps = require_f32(ctx, "fish_speech.codec.rvq_transformer.layer_norm_rms_eps");
    
    return meta;
}

// ---------------------------------------------------------------------------
// Generic metadata access (with defaults)
// ---------------------------------------------------------------------------

uint32_t GGUFMetadata::get_u32(gguf_context* ctx, const char* key, uint32_t def) {
    if (!ctx) return def;
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) return def;
    return gguf_get_val_u32(ctx, id);
}

int32_t GGUFMetadata::get_i32(gguf_context* ctx, const char* key, int32_t def) {
    if (!ctx) return def;
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) return def;
    
    auto type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_INT32) {
        return gguf_get_val_i32(ctx, id);
    } else if (type == GGUF_TYPE_UINT32) {
        return static_cast<int32_t>(gguf_get_val_u32(ctx, id));
    } else {
        return def;
    }
}

float GGUFMetadata::get_f32(gguf_context* ctx, const char* key, float def) {
    if (!ctx) return def;
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) return def;
    return gguf_get_val_f32(ctx, id);
}

bool GGUFMetadata::get_bool(gguf_context* ctx, const char* key, bool def) {
    if (!ctx) return def;
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) return def;
    return gguf_get_val_bool(ctx, id);
}

std::string GGUFMetadata::get_string(gguf_context* ctx, const char* key, const std::string& def) {
    if (!ctx) return def;
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) return def;
    const char* str = gguf_get_val_str(ctx, id);
    return str ? std::string(str) : def;
}

std::vector<uint32_t> GGUFMetadata::get_u32_array(gguf_context* ctx, const char* key) {
    std::vector<uint32_t> result;
    if (!ctx) return result;
    
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) return result;
    
    size_t n = static_cast<size_t>(gguf_get_arr_n(ctx, id));
    const uint32_t* data = static_cast<const uint32_t*>(gguf_get_arr_data(ctx, id));
    if (!data) return result;
    
    result.assign(data, data + n);
    return result;
}

std::vector<int32_t> GGUFMetadata::get_i32_array(gguf_context* ctx, const char* key) {
    if (!ctx) return {};
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) return {};
    return gguf_array_to_i32_vector(ctx, id);
}

// ---------------------------------------------------------------------------
// Strict metadata access (throws on missing keys)
// ---------------------------------------------------------------------------

uint32_t GGUFMetadata::require_u32(gguf_context* ctx, const char* key) {
    if (!ctx) throw std::runtime_error(std::string("GGUFMetadata: null context for key: ") + key);
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) throw std::runtime_error(std::string("GGUFMetadata: missing key: ") + key);
    return gguf_get_val_u32(ctx, id);
}

int32_t GGUFMetadata::require_i32(gguf_context* ctx, const char* key) {
    if (!ctx) throw std::runtime_error(std::string("GGUFMetadata: null context for key: ") + key);
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) throw std::runtime_error(std::string("GGUFMetadata: missing key: ") + key);
    
    auto type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_INT32) {
        return gguf_get_val_i32(ctx, id);
    } else if (type == GGUF_TYPE_UINT32) {
        return static_cast<int32_t>(gguf_get_val_u32(ctx, id));
    } else {
        throw std::runtime_error(std::string("GGUFMetadata: expected INT32/UINT32 for key: ") + key);
    }
}

float GGUFMetadata::require_f32(gguf_context* ctx, const char* key) {
    if (!ctx) throw std::runtime_error(std::string("GGUFMetadata: null context for key: ") + key);
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) throw std::runtime_error(std::string("GGUFMetadata: missing key: ") + key);
    return gguf_get_val_f32(ctx, id);
}

bool GGUFMetadata::require_bool(gguf_context* ctx, const char* key) {
    if (!ctx) throw std::runtime_error(std::string("GGUFMetadata: null context for key: ") + key);
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) throw std::runtime_error(std::string("GGUFMetadata: missing key: ") + key);
    return gguf_get_val_bool(ctx, id);
}

std::string GGUFMetadata::require_string(gguf_context* ctx, const char* key) {
    if (!ctx) throw std::runtime_error(std::string("GGUFMetadata: null context for key: ") + key);
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) throw std::runtime_error(std::string("GGUFMetadata: missing key: ") + key);
    const char* str = gguf_get_val_str(ctx, id);
    if (!str) throw std::runtime_error(std::string("GGUFMetadata: null string for key: ") + key);
    return std::string(str);
}

std::vector<int32_t> GGUFMetadata::require_i32_array(gguf_context* ctx, const char* key) {
    if (!ctx) throw std::runtime_error(std::string("GGUFMetadata: null context for key: ") + key);
    int id = static_cast<int>(gguf_find_key(ctx, key));
    if (id < 0) throw std::runtime_error(std::string("GGUFMetadata: missing key: ") + key);
    return gguf_array_to_i32_vector(ctx, id);
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

std::string GGUFMetadata::get_architecture_prefix(gguf_context* ctx) {
    std::string arch = get_string(ctx, "general.architecture", "fish-speech");
    return arch + ".";
}

bool GGUFMetadata::is_unified_model(gguf_context* ctx) {
    std::string arch = get_string(ctx, "general.architecture", "");
    return arch == "fish-speech";
}

bool GGUFMetadata::is_standalone_codec(gguf_context* ctx) {
    std::string arch = get_string(ctx, "general.architecture", "");
    return arch == "fish-speech-codec";
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

std::vector<int32_t> GGUFMetadata::gguf_array_to_i32_vector(gguf_context* ctx, int id) {
    auto type = gguf_get_arr_type(ctx, id);
    size_t n = static_cast<size_t>(gguf_get_arr_n(ctx, id));
    std::vector<int32_t> v(n);
    
    if (type == GGUF_TYPE_UINT32) {
        const auto* d = static_cast<const uint32_t*>(gguf_get_arr_data(ctx, id));
        for (size_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(d[i]);
    } else if (type == GGUF_TYPE_INT32) {
        const auto* d = static_cast<const int32_t*>(gguf_get_arr_data(ctx, id));
        for (size_t i = 0; i < n; ++i) v[i] = d[i];
    } else if (type == GGUF_TYPE_UINT64) {
        const auto* d = static_cast<const uint64_t*>(gguf_get_arr_data(ctx, id));
        for (size_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(d[i]);
    } else {
        throw std::runtime_error("GGUFMetadata: unexpected array type");
    }
    
    return v;
}

} // namespace s2