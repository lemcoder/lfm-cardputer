#pragma once
// Cheap drafts for speculative decoding, and the decoding step itself.
//
// Draft sources, tried in order:
//   1. prompt lookup: find the last 2 tokens earlier in the history and
//      propose what followed them there (repeats, lists, code);
//   2. n-gram table: trigram (a, b) -> most frequent next token, bigram
//      fallback, built from a corpus by tools/spec_eval.py --export. On the
//      device it lives in the flash "weights" partition, read through mmap.
//
// Greedy verification makes the result identical to plain greedy decoding;
// drafts only change how many tokens one pass over the weights produces.

#include <cstddef>
#include <cstdint>
#include <span>

namespace lfm {

class Engine;

// N-gram table file layout (little-endian), written by tools/spec_eval.py:
//   NgramHeader
//   uint16 bigram[vocab]            next token after b; 0xFFFF = none
//   NgramEntry tri[tri_slots]       open addressing, linear probing
struct NgramHeader {
    char magic[8];         // "LFMNGRM\0"
    uint32_t version;      // 1
    uint32_t vocab;
    uint32_t tri_slots;    // power of two
    uint32_t reserved;
};
struct NgramEntry {
    uint32_t key;   // (a << 16) | b
    uint16_t next;  // 0xFFFF = empty slot
    uint16_t pad;
};
static_assert(sizeof(NgramHeader) == 24 && sizeof(NgramEntry) == 8);

inline constexpr char kNgramMagic[8] = {'L', 'F', 'M', 'N', 'G', 'R', 'M', '\0'};
inline constexpr uint16_t kNgramNone = 0xFFFF;

// Slot for a trigram key: Fibonacci hashing (same in spec_eval.py).
inline uint32_t ngram_slot(uint32_t key, uint32_t slots) {
    uint32_t bits = 0;
    while ((1u << bits) < slots) ++bits;
    return bits == 0 ? 0 : (key * 0x9E3779B1u) >> (32 - bits);
}

class Drafter {
public:
    // history: caller-owned buffer for up to `cap` tokens (the context).
    Drafter(int32_t* history, int cap) : hist_(history), cap_(cap) {}

    // Optional n-gram table (whole file contents, kept by the caller).
    // False if it isn't a valid table; drafting then uses lookup only.
    bool set_ngram(const void* table, std::size_t size);

    void reset() { n_ = 0; }
    void push(int32_t token);
    int size() const { return n_; }

    // Up to k draft tokens continuing the history. Returns how many.
    int draft(int32_t* out, int k) const;

private:
    int lookup(int32_t* out, int k) const;
    int ngram(int32_t* out, int k) const;
    int32_t ngram_next(int32_t a, int32_t b) const;  // -1 if unknown

    int32_t* hist_;
    int cap_;
    int n_ = 0;
    const uint16_t* bigram_ = nullptr;
    const NgramEntry* tri_ = nullptr;
    uint32_t vocab_ = 0;
    uint32_t tri_slots_ = 0;
};

// Speculative decoding statistics.
struct SpecStats {
    int passes = 0;   // verify() calls (one weight read each)
    int tokens = 0;   // tokens produced
    int drafted = 0;  // drafts proposed
    int accepted = 0; // drafts accepted
};

// One step. `fed` is the newest token: already produced (and pushed to the
// drafter) but not yet run through the model. Drafts up to `max_draft`
// tokens, verifies them in one pass, and writes the new tokens (accepted
// drafts, then the model's own next token) to out. Pushes them to the
// drafter. Returns how many (>= 1), or a negative Engine code. The last one
// is the next step's `fed`.
int speculative_step(Engine& eng, Drafter& drafter, int32_t fed, int max_draft, int32_t* out, SpecStats* stats);

// The prompt, max_batch tokens per pass, with drafts for the first answer
// tokens riding along in the spare room of the last batch (that pass happens
// anyway). The drafter must already hold the prompt. Writes the produced
// tokens to out (accepted drafts, then the model's next token; the last is
// the next step's `fed`) and returns how many, or a negative Engine code.
int speculative_prefill(Engine& eng, Drafter& drafter, std::span<const int32_t> prompt, int max_batch, int max_draft,
                        int32_t* out, SpecStats* stats);

}  // namespace lfm
