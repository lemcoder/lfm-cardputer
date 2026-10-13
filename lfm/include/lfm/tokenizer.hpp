#pragma once
// Text -> token ids for LFM2.5 (byte-level BPE), small enough for the
// device: the merge table (tools/export_tokenizer.py, ~500 KB) stays in a
// file and is read one 512-byte block per lookup; RAM holds only the first
// key of every block (~4 KB).
//
// Two steps, as in the HF tokenizer:
//   1. split the text with the model's pre-tokenizer regex
//        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
//        | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//      written out by hand. Letters are ASCII letters and every byte >= 0x80
//      (exact for ASCII text, an approximation for other scripts);
//   2. BPE inside each piece: start from one token per byte, repeatedly
//      merge the adjacent pair with the lowest merge rank.
//
// Non-special added tokens ("python", "<think>", ...) are matched literally
// first, as HF does. Special tokens (<|im_start|> ...) are never produced
// from text; callers add their ids directly.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "lfm/source.hpp"

namespace lfm {

struct TokenizerHeader {
    char magic[8];  // "LFMTOK\0\0"
    uint32_t version;
    uint32_t n_merges;
    uint16_t byte_id[256];
};
struct TokenizerMerge {
    uint32_t key;  // (left << 16) | right
    uint16_t result;
    uint16_t rank;
};
static_assert(sizeof(TokenizerHeader) == 528 && sizeof(TokenizerMerge) == 8);

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Reads the header and builds the block index (one pass over the file).
    bool open(Source& src);

    // Appends the tokens of `text` to out. Returns how many, or -1 if they
    // don't fit in cap or the file can't be read.
    int encode(std::string_view text, int32_t* out, int cap);

    // Pre-tokenizer only: length of the piece starting at text[0] (> 0 for non-empty text).
    static std::size_t piece_length(std::string_view text);

private:
    static constexpr std::size_t kBlockMerges = 64;  // 512 bytes per read

    // Rank of merging (left, right), or -1; *result = the merged token.
    int lookup(uint32_t left, uint32_t right, uint16_t* result);
    int encode_piece(std::string_view piece, int32_t* out, int cap);
    int encode_split(std::string_view text, int32_t* out, int cap);

    static constexpr int kMaxAdded = 8;
    static constexpr std::size_t kMaxAddedLen = 24;
    struct Added {
        int32_t id;
        uint8_t len;
        char text[kMaxAddedLen];
    };
    Added added_[kMaxAdded];
    int n_added_ = 0;

    Source* src_ = nullptr;
    TokenizerHeader h_{};
    uint32_t* first_key_ = nullptr;  // per block
    uint32_t n_blocks_ = 0;
    TokenizerMerge block_[kBlockMerges];
    int cached_ = -1;  // block_ holds this block
};

}  // namespace lfm
