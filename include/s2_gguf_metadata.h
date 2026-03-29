#pragma once
// s2_gguf_metadata.h — GGUF metadata reading utilities
//
// Provides safe, consistent access to GGUF metadata with proper error handling
// and type conversions.

#include "ggml.h"
#include "gguf.h"
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

namespace s2 {

// Model hyperparameters (SlowARModel)
struct ModelMetadata {
    int32_t context_length = 0;
    int32_t vocab_size = 0;
    int32_t embedding_length = 0;
    int32_t feed_forward_length = 0;
    int32_t block_count = 0;
    int32_t head_count = 0;
    int32_t head_count_kv = 0;
    int32_t codebook_size = 0;
    int32_t num_codebooks = 0;
    int32_t semantic_begin_id = 0;
    int32_t semantic_end_id = 0;
    float rope_freq_base = 10000.0f;
    float rms_norm_eps = 1e-5f;
    bool tie_word_embeddings = true;
    bool attention_qk_norm = false;
    bool scale_codebook_embeddings = false;
    
    // Fast decoder
    int32_t fast_context_length = 0;
    int32_t fast_embedding_length = 0;
    int32_t fast_feed_forward_length = 0;
    int32_t fast_block_count = 0;
    int32_t fast_head_count = 0;
    int32_t fast_head_count_kv = 0;
    int32_t fast_head_dim = 0;
    float fast_rope_freq_base = 10000.0f;
    float fast_rms_norm_eps = 1e-5f;
    bool fast_attention_qk_norm = false;
    bool fast_has_project_in = false;
    bool has_fast_decoder = false;
    
    // Architecture info
    std::string architecture;
    std::string arch_prefix;
};

// Codec hyperparameters (AudioCodec)
struct CodecMetadata {
    int32_t sample_rate = 44100;
    int32_t hop_length = 512;
    int32_t frame_length = 512;
    int32_t encoder_dim = 0;
    int32_t decoder_dim = 0;
    int32_t latent_dim = 0;
    std::vector<int32_t> encoder_rates;
    std::vector<int32_t> decoder_rates;
    std::vector<int32_t> encoder_transformer_layers;
    
    int32_t quantizer_input_dim = 0;
    int32_t quantizer_codebook_dim = 0;
    int32_t quantizer_residual_codebooks = 0;
    int32_t quantizer_residual_codebook_size = 0;
    int32_t quantizer_semantic_codebook_size = 0;
    std::vector<int32_t> quantizer_downsample_factor;
    
    int32_t transformer_block_size = 0;
    int32_t transformer_n_local_heads = 0;
    int32_t transformer_head_dim = 0;
    float transformer_rope_base = 0.0f;
    float transformer_norm_eps = 0.0f;
    
    int32_t rvq_transformer_window_size = 0;
    int32_t rvq_transformer_block_size = 0;
    int32_t rvq_transformer_n_layer = 0;
    int32_t rvq_transformer_n_local_heads = 0;
    int32_t rvq_transformer_head_dim = 0;
    int32_t rvq_transformer_dim = 0;
    float rvq_transformer_rope_base = 0.0f;
    float rvq_transformer_norm_eps = 0.0f;
    
    std::string tensor_prefix; // "c." for unified models, "" for standalone codec
};

class GGUFMetadata {
public:
    // Read model metadata from GGUF context
    static ModelMetadata read_model_metadata(gguf_context* ctx);
    
    // Read codec metadata from GGUF context
    static CodecMetadata read_codec_metadata(gguf_context* ctx);
    
    // Generic metadata access with defaults (non-throwing)
    static uint32_t get_u32(gguf_context* ctx, const char* key, uint32_t def = 0);
    static int32_t get_i32(gguf_context* ctx, const char* key, int32_t def = 0);
    static float get_f32(gguf_context* ctx, const char* key, float def = 0.0f);
    static bool get_bool(gguf_context* ctx, const char* key, bool def = false);
    static std::string get_string(gguf_context* ctx, const char* key, const std::string& def = "");
    static std::vector<uint32_t> get_u32_array(gguf_context* ctx, const char* key);
    static std::vector<int32_t> get_i32_array(gguf_context* ctx, const char* key);
    
    // Strict metadata access (throws on missing keys)
    static uint32_t require_u32(gguf_context* ctx, const char* key);
    static int32_t require_i32(gguf_context* ctx, const char* key);
    static float require_f32(gguf_context* ctx, const char* key);
    static bool require_bool(gguf_context* ctx, const char* key);
    static std::string require_string(gguf_context* ctx, const char* key);
    static std::vector<int32_t> require_i32_array(gguf_context* ctx, const char* key);
    
    // Helper to get architecture prefix (e.g., "fish-speech.")
    static std::string get_architecture_prefix(gguf_context* ctx);
    
    // Check if architecture is unified model or standalone codec
    static bool is_unified_model(gguf_context* ctx);
    static bool is_standalone_codec(gguf_context* ctx);
    
private:
    GGUFMetadata() = delete;
    ~GGUFMetadata() = delete;
    
    // Internal helper to convert GGUF array to vector<int32_t>
    static std::vector<int32_t> gguf_array_to_i32_vector(gguf_context* ctx, int id);
};

} // namespace s2