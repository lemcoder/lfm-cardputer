#pragma once
// Cheap drafts for speculative decoding, and the decoding step itself.
//
// Draft sources, tried in order:
//   1. prompt lookup: find the last 2 tokens earlier in the history and
//      propose what followed them there (repeats, lists, code);
//   2. a draft head (if set): a small network trained on the model's own
//      text (tools/draft_heads.py) that guesses the next tokens from the
//      model's final hidden state, one shared step per draft:
//        h' = h + up(silu(down([h, rmsnorm(embedding(token))])))
//        draft = argmax(LM head(final_norm(h')))
//      with its top guesses forming a tree of drafts (DraftShape).
// On the device the head lives in the flash "draft" partition, read through
// mmap.
//
// Greedy verification makes the result identical to plain greedy decoding;
// drafts only change how many tokens one pass over the weights produces.

#include <cstddef>
#include <cstdint>
#include <span>

#include "lfm/format.hpp"

namespace lfm {

class Engine;

// Draft head file layout (little-endian), written by tools/draft_heads.py export:
//   DraftHeadHeader, then the two matrices at their offsets (from the header start).
struct DraftHeadHeader {
    char magic[8];         // "LFMDHED\0"
    uint32_t version;      // 1
    uint32_t dim;
    uint32_t rank;
    Dtype dtype;           // of both matrices
    uint64_t down_offset;  // [rank][2 * dim], input [h, rmsnorm(embedding(token))]
    uint64_t up_offset;    // [dim][rank]
};
static_assert(sizeof(DraftHeadHeader) == 40);
inline constexpr char kDraftHeadMagic[8] = {'L', 'F', 'M', 'D', 'H', 'E', 'D', '\0'};

// A draft tree's shape: node i (1 .. n-1) is the rank[i]-th guess (0 = best,
// at most 2) of the head after node parent[i]; node 0 is the fed token.
// Parents come before children.
struct DraftShape {
    int n;
    int8_t parent[8];
    int8_t rank[8];
};
// Chain of 2 (the head's best guess, then its best guess after that).
inline constexpr DraftShape kChain2 = {3, {-1, 0, 1}, {0, 0, 0}};
// Top 3 first guesses, one more under each of the best two, one more under
// the best path (tools/tree_sim.py "3x1+": 1.95 tokens per pass on chat).
inline constexpr DraftShape kTree3x1 = {7, {-1, 0, 0, 0, 1, 2, 4}, {0, 0, 1, 2, 0, 0, 0}};

class DraftHead {
public:
    // The whole file contents, kept by the caller (memory-mapped flash).
    // False if it isn't a head for this model.
    bool set(const void* data, std::size_t size, const Engine& eng);
    bool ok() const { return down_ != nullptr; }

    // Up to k drafts for the tokens after `fed`, from eng.last_hidden() (the
    // state that produced fed). Uses the engine's scratch between passes.
    // Returns how many (0 if there's no hidden state), or a negative Engine code.
    int draft(Engine& eng, int32_t fed, int k, int32_t* out) const;

    // A tree of drafts: tokens[0] = fed, tokens[i] for the shape's nodes,
    // parents[i] = shape.parent[i]. One head step per node with children.
    // Returns the node count (shape.n), 0 without a hidden state, or a
    // negative Engine code. Needs max_batch >= 5 (scratch for the states).
    int draft_tree(Engine& eng, int32_t fed, const DraftShape& shape, int32_t* tokens, int8_t* parents) const;

private:
    // One step: h' = h + up(silu(down([h, rmsnorm(embedding(tok))]))) into out.
    bool step(Engine& eng, const float* h, int32_t tok, float* out) const;

    DraftHeadHeader h_{};
    const std::byte* down_ = nullptr;
    const std::byte* up_ = nullptr;
};

class Drafter {
public:
    // history: caller-owned buffer for up to `cap` tokens (the context).
    Drafter(int32_t* history, int cap) : hist_(history), cap_(cap) {}

    void reset() { n_ = 0; }
    void push(int32_t token);
    int size() const { return n_; }

    // Prompt lookup: up to k tokens that followed the last 2 tokens the most
    // recent time they appeared earlier. Returns how many.
    int lookup(int32_t* out, int k) const;

    // Optional draft head, used when lookup finds nothing. Kept by the caller.
    void set_head(const DraftHead* head) { head_ = head; }
    const DraftHead* head() const { return head_; }
    // Tree shape for the head's drafts (nullptr: a chain of max_draft).
    void set_shape(const DraftShape* shape) { shape_ = shape; }
    const DraftShape* shape() const { return shape_; }

private:
    int32_t* hist_;
    int cap_;
    int n_ = 0;
    const DraftHead* head_ = nullptr;
    const DraftShape* shape_ = nullptr;
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

}  // namespace lfm
