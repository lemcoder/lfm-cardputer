#include "lfm/draft.hpp"

#include <cmath>
#include <cstring>

#include "lfm/engine.hpp"
#include "lfm/math/ops.hpp"

namespace lfm {

void Drafter::push(int32_t token) {
    if (n_ == cap_) {  // full: drop the oldest half (lookup only needs recent context)
        const int keep = cap_ / 2;
        std::memmove(hist_, hist_ + (cap_ - keep), static_cast<std::size_t>(keep) * sizeof(int32_t));
        n_ = keep;
    }
    hist_[n_++] = token;
}

// Most recent earlier occurrence of the last 2 tokens; propose what followed.
int Drafter::lookup(int32_t* out, int k) const {
    constexpr int kMatch = 2;
    if (n_ <= kMatch) return 0;
    const int32_t* tail = hist_ + n_ - kMatch;
    for (int start = n_ - kMatch - 1; start >= 0; --start) {
        if (hist_[start] == tail[0] && hist_[start + 1] == tail[1]) {
            int m = 0;
            for (int i = start + kMatch; i < n_ && m < k; ++i) out[m++] = hist_[i];
            return m;
        }
    }
    return 0;
}

bool DraftHead::set(const void* data, std::size_t size, const Engine& eng) {
    down_ = up_ = nullptr;
    if (data == nullptr || size < sizeof(DraftHeadHeader)) return false;
    std::memcpy(&h_, data, sizeof(h_));
    const uint32_t dim = eng.header().dim;
    const bool ok = std::memcmp(h_.magic, kDraftHeadMagic, sizeof(kDraftHeadMagic)) == 0 && h_.version == 1 &&
                    h_.dim == dim && h_.rank % kQ8BlockSize == 0 && h_.rank > 0 &&
                    (h_.dtype == Dtype::Q8_0 || h_.dtype == Dtype::Q4_0 || h_.dtype == Dtype::BF16) &&
                    h_.down_offset + h_.rank * row_bytes(h_.dtype, 2 * dim) <= size &&
                    h_.up_offset + dim * row_bytes(h_.dtype, h_.rank) <= size;
    if (!ok) return false;
    down_ = static_cast<const std::byte*>(data) + h_.down_offset;
    up_ = static_cast<const std::byte*>(data) + h_.up_offset;
    return true;
}

bool DraftHead::step(Engine& eng, const float* h, int32_t tok, float* out) const {
    const std::size_t dim = h_.dim, rank = h_.rank;
    float* z = eng.scratch(0);   // [h | rmsnorm(embedding)]
    float* u = eng.scratch(1);   // rank
    float* work = u + rank;      // dim
    std::memcpy(z, h, dim * sizeof(float));
    float* e = z + dim;
    if (!eng.embedding(tok, e)) return false;
    float ss = 0;
    for (std::size_t j = 0; j < dim; ++j) ss += e[j] * e[j];
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + 1e-5f);
    for (std::size_t j = 0; j < dim; ++j) e[j] *= inv;
    eng.matvec_mem(down_, h_.dtype, rank, 2 * dim, z, u);
    math::silu({u, rank});
    eng.matvec_mem(up_, h_.dtype, dim, rank, u, work);
    for (std::size_t j = 0; j < dim; ++j) out[j] = h[j] + work[j];
    return true;
}

int DraftHead::draft_tree(Engine& eng, int32_t fed, const DraftShape& shape, int32_t* tokens, int8_t* parents) const {
    const float* h0 = eng.last_hidden();
    const std::size_t dim = h_.dim;
    // States after each expansion: two per scratch slot, slots 2 and 3.
    float* s2 = eng.scratch(2);
    float* s3 = eng.scratch(3);
    float* work = eng.scratch(1) + h_.rank;
    if (!ok() || h0 == nullptr || s2 == nullptr || s3 == nullptr || eng.scratch_floats() < 2 * dim ||
        eng.scratch_floats() < h_.rank + dim || shape.n < 1 || shape.n > Engine::kMaxBatch) {
        return 0;
    }
    float* store[4] = {s2, s2 + dim, s3, s3 + dim};
    const float* state[Engine::kMaxBatch] = {};  // the state each node's guess came from
    const float* out_of[Engine::kMaxBatch] = {};  // state after expanding node i
    int32_t cand[Engine::kMaxBatch][3];
    int n_store = 0;
    tokens[0] = fed;
    parents[0] = -1;
    state[0] = h0;
    for (int i = 1; i < shape.n; ++i) {
        const int p = shape.parent[i];
        if (p < 0 || p >= i || shape.rank[i] < 0 || shape.rank[i] > 2) return Engine::kInvalid;
        if (out_of[p] == nullptr) {  // expand p: one head step, its top 3 guesses
            if (n_store == 4) return Engine::kInvalid;
            float* h = store[n_store++];
            if (!step(eng, state[p], tokens[p], h)) return Engine::kIoError;
            cand[p][0] = cand[p][1] = cand[p][2] = Engine::kNoToken;
            const int32_t t = eng.draft_token(h, work, cand[p], 3);
            if (t < 0) return t;
            out_of[p] = h;
        }
        tokens[i] = cand[p][shape.rank[i]];
        if (tokens[i] < 0) tokens[i] = cand[p][0];  // fewer candidates than asked: repeat the best
        parents[i] = static_cast<int8_t>(p);
        state[i] = out_of[p];
    }
    return shape.n;
}

int DraftHead::draft(Engine& eng, int32_t fed, int k, int32_t* out) const {
    const float* h0 = eng.last_hidden();
    const std::size_t dim = h_.dim, rank = h_.rank;
    // z = [h | embedding], w = [u (rank) | work (dim)], in two scratch slots.
    float* z = eng.scratch(0);
    float* w = eng.scratch(1);
    if (!ok() || h0 == nullptr || z == nullptr || w == nullptr || eng.scratch_floats() < 2 * dim ||
        eng.scratch_floats() < rank + dim) {
        return 0;
    }
    float* u = w;
    float* work = w + rank;
    std::memcpy(z, h0, dim * sizeof(float));
    int32_t tok = fed;
    for (int i = 0; i < k; ++i) {
        float* e = z + dim;
        if (!eng.embedding(tok, e)) return Engine::kIoError;
        float ss = 0;
        for (std::size_t j = 0; j < dim; ++j) ss += e[j] * e[j];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + 1e-5f);
        for (std::size_t j = 0; j < dim; ++j) e[j] *= inv;
        eng.matvec_mem(down_, h_.dtype, rank, 2 * dim, z, u);
        math::silu({u, rank});
        eng.matvec_mem(up_, h_.dtype, dim, rank, u, work);
        for (std::size_t j = 0; j < dim; ++j) z[j] += work[j];  // h' (the next step's h)
        tok = eng.draft_token(z, work);
        if (tok < 0) return tok;
        out[i] = tok;
    }
    return k;
}

int speculative_step(Engine& eng, Drafter& drafter, int32_t fed, int max_draft, int32_t* out, SpecStats* stats) {
    int32_t tokens[Engine::kMaxBatch];
    int32_t preds[Engine::kMaxBatch];
    tokens[0] = fed;
    // Never draft past the context.
    int k = max_draft;
    if (k > Engine::kMaxBatch - 1) k = Engine::kMaxBatch - 1;
    if (k > eng.max_ctx() - eng.pos() - 1) k = eng.max_ctx() - eng.pos() - 1;

    // Prompt lookup (a chain), else the draft head: a tree when a shape is
    // set and fits, else a chain.
    int n_draft = k > 0 ? drafter.lookup(tokens + 1, k) : 0;
    int n_tree = 0;
    int8_t parents[Engine::kMaxBatch];
    if (k > 0 && n_draft == 0 && drafter.head() != nullptr) {
        const DraftShape* shape = drafter.shape();
        if (shape != nullptr && shape->n - 1 <= k && shape->n <= eng.max_batch()) {
            n_tree = drafter.head()->draft_tree(eng, fed, *shape, tokens, parents);
            if (n_tree < 0) return n_tree;
        } else {
            n_draft = drafter.head()->draft(eng, fed, k, tokens + 1);
            if (n_draft < 0) return n_draft;
        }
    }

    int n_out = 0;
    int accepted = 0;
    if (n_tree > 1) {
        const int32_t r = eng.verify_tree({tokens, static_cast<std::size_t>(n_tree)}, parents, preds);
        if (r < 0) return r;
        // Longest path whose every token is what the model predicts after its parent.
        int8_t path[Engine::kMaxBatch] = {0};
        int len = 1;
        for (int node = 0;;) {
            int next = -1;
            for (int c = node + 1; c < n_tree && next < 0; ++c) {
                if (parents[c] == node && tokens[c] == preds[node]) next = c;
            }
            if (next < 0) break;
            path[len++] = static_cast<int8_t>(next);
            node = next;
        }
        if (!eng.accept_path(path, len)) return Engine::kIoError;
        for (int i = 1; i < len; ++i) out[n_out++] = tokens[path[i]];
        out[n_out++] = preds[path[len - 1]];
        accepted = len - 1;
        n_draft = n_tree - 1;
    } else {
        const int32_t r = eng.verify({tokens, static_cast<std::size_t>(1 + n_draft)}, preds);
        if (r < 0) return r;
        while (accepted < n_draft && tokens[1 + accepted] == preds[accepted]) ++accepted;
        if (!eng.accept(1 + accepted)) return Engine::kIoError;
        for (int i = 0; i < accepted; ++i) out[n_out++] = tokens[1 + i];
        out[n_out++] = preds[accepted];
    }
    for (int i = 0; i < n_out; ++i) drafter.push(out[i]);

    if (stats != nullptr) {
        ++stats->passes;
        stats->tokens += n_out;
        stats->drafted += n_draft;
        stats->accepted += accepted;
    }
    return n_out;
}

}  // namespace lfm
