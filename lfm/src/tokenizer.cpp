#include "lfm/tokenizer.hpp"

#include <cstdlib>
#include <cstring>

namespace lfm {

namespace {

constexpr char kTokMagic[8] = {'L', 'F', 'M', 'T', 'O', 'K', '\0', '\0'};
constexpr std::size_t kMaxPiece = 256;  // longer pieces are split (only runs of spaces / symbols get this long)

bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_newline(unsigned char c) { return c == '\r' || c == '\n'; }
bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
bool is_other(unsigned char c) { return !is_space(c) && !is_letter(c) && !is_digit(c); }

char lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

}  // namespace

Tokenizer::~Tokenizer() { std::free(first_key_); }

bool Tokenizer::open(Source& src) {
    src_ = &src;
    cached_ = -1;
    if (!src.read_at(0, &h_, sizeof(h_))) return false;
    if (std::memcmp(h_.magic, kTokMagic, sizeof(kTokMagic)) != 0 || h_.version != 1 || h_.n_merges == 0) return false;
    n_blocks_ = (h_.n_merges + kBlockMerges - 1) / kBlockMerges;
    std::free(first_key_);
    first_key_ = static_cast<uint32_t*>(std::malloc(n_blocks_ * sizeof(uint32_t)));
    if (first_key_ == nullptr) return false;
    for (uint32_t b = 0; b < n_blocks_; ++b) {
        const uint64_t off = sizeof(TokenizerHeader) + static_cast<uint64_t>(b) * kBlockMerges * sizeof(TokenizerMerge);
        if (!src.read_at(off, &first_key_[b], sizeof(uint32_t))) return false;
    }
    uint64_t off = sizeof(TokenizerHeader) + static_cast<uint64_t>(h_.n_merges) * sizeof(TokenizerMerge);
    uint32_t n_added = 0;
    if (!src.read_at(off, &n_added, sizeof(n_added))) return false;
    off += sizeof(n_added);
    n_added_ = 0;
    for (uint32_t i = 0; i < n_added; ++i) {
        uint16_t id_len[2];
        if (!src.read_at(off, id_len, sizeof(id_len))) return false;
        off += sizeof(id_len);
        if (id_len[1] <= kMaxAddedLen && n_added_ < kMaxAdded) {
            Added& a = added_[n_added_++];
            a.id = id_len[0];
            a.len = static_cast<uint8_t>(id_len[1]);
            if (!src.read_at(off, a.text, a.len)) return false;
        }
        off += id_len[1];
    }
    return true;
}

int Tokenizer::lookup(uint32_t left, uint32_t right, uint16_t* result) {
    const uint32_t key = (left << 16) | right;
    // Last block whose first key <= key.
    uint32_t lo = 0, hi = n_blocks_;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        (first_key_[mid] <= key ? lo : hi) = mid;
    }
    if (first_key_[lo] > key) return -1;
    if (cached_ != static_cast<int>(lo)) {
        const uint32_t n = lo + 1 < n_blocks_ ? kBlockMerges : h_.n_merges - lo * kBlockMerges;
        const uint64_t off = sizeof(TokenizerHeader) + static_cast<uint64_t>(lo) * kBlockMerges * sizeof(TokenizerMerge);
        if (!src_->read_at(off, block_, n * sizeof(TokenizerMerge))) return -2;
        for (uint32_t i = n; i < kBlockMerges; ++i) block_[i].key = 0xFFFFFFFFu;
        cached_ = static_cast<int>(lo);
    }
    std::size_t a = 0, b = kBlockMerges;
    while (a < b) {
        const std::size_t mid = (a + b) / 2;
        if (block_[mid].key < key) a = mid + 1;
        else b = mid;
    }
    if (a == kBlockMerges || block_[a].key != key) return -1;
    *result = block_[a].result;
    return block_[a].rank;
}

std::size_t Tokenizer::piece_length(std::string_view t) {
    const std::size_t n = t.size();
    if (n == 0) return 0;
    auto at = [&](std::size_t i) { return static_cast<unsigned char>(t[i]); };

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (t[0] == '\'' && n >= 2) {
        const char c1 = lower(t[1]);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
        if (n >= 3) {
            const char c2 = lower(t[2]);
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return 3;
        }
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    {
        std::size_t i = 0;
        if (!is_newline(at(0)) && !is_letter(at(0)) && !is_digit(at(0)) && n >= 2 && is_letter(at(1))) i = 1;
        if (is_letter(at(i))) {
            while (i < n && is_letter(at(i))) ++i;
            return i;
        }
    }
    // \p{N}{1,3}
    if (is_digit(at(0))) {
        std::size_t i = 1;
        while (i < n && i < 3 && is_digit(at(i))) ++i;
        return i;
    }
    // ' '?[^\s\p{L}\p{N}]+[\r\n]*
    {
        std::size_t i = (at(0) == ' ' && n >= 2 && is_other(at(1))) ? 1 : 0;
        if (is_other(at(i))) {
            while (i < n && is_other(at(i))) ++i;
            while (i < n && is_newline(at(i))) ++i;
            return i;
        }
    }
    // Whitespace from here on.
    std::size_t run = 0;
    while (run < n && is_space(at(run))) ++run;
    // \s*[\r\n]+ : up to and including the last newline of the run.
    for (std::size_t k = run; k > 0; --k) {
        if (is_newline(at(k - 1))) return k;
    }
    // \s+(?!\S) : the run, minus its last space when a non-space follows.
    if (run == n) return run;
    if (run >= 2) return run - 1;
    // \s+
    return run > 0 ? run : 1;
}

int Tokenizer::encode_piece(std::string_view piece, int32_t* out, int cap) {
    int32_t sym[kMaxPiece];
    int rank[kMaxPiece];     // rank of merging sym[i] with sym[i+1], -1 = none
    uint16_t merged[kMaxPiece];
    int n = static_cast<int>(piece.size());
    for (int i = 0; i < n; ++i) sym[i] = h_.byte_id[static_cast<unsigned char>(piece[i])];
    for (int i = 0; i + 1 < n; ++i) {
        rank[i] = lookup(static_cast<uint32_t>(sym[i]), static_cast<uint32_t>(sym[i + 1]), &merged[i]);
        if (rank[i] == -2) return -1;
    }
    while (n > 1) {
        int best = -1;
        for (int i = 0; i + 1 < n; ++i) {
            if (rank[i] >= 0 && (best < 0 || rank[i] < rank[best])) best = i;
        }
        if (best < 0) break;
        sym[best] = merged[best];
        for (int i = best + 1; i + 1 < n; ++i) {  // drop sym[best + 1]
            sym[i] = sym[i + 1];
            rank[i] = rank[i + 1];
            merged[i] = merged[i + 1];
        }
        --n;
        for (int i : {best - 1, best}) {  // the pairs around the new token
            if (i < 0 || i + 1 >= n) continue;
            rank[i] = lookup(static_cast<uint32_t>(sym[i]), static_cast<uint32_t>(sym[i + 1]), &merged[i]);
            if (rank[i] == -2) return -1;
        }
    }
    if (n > cap) return -1;
    for (int i = 0; i < n; ++i) out[i] = sym[i];
    return n;
}

int Tokenizer::encode(std::string_view text, int32_t* out, int cap) {
    if (src_ == nullptr) return -1;
    int total = 0;
    // Added tokens first: the leftmost match (the longest one there) splits the text.
    while (!text.empty()) {
        std::size_t at = std::string_view::npos;
        const Added* hit = nullptr;
        for (int i = 0; i < n_added_; ++i) {
            const std::size_t p = text.find(std::string_view(added_[i].text, added_[i].len));
            if (p != std::string_view::npos && (p < at || (p == at && added_[i].len > hit->len))) {
                at = p;
                hit = &added_[i];
            }
        }
        const int got = encode_split(text.substr(0, at), out + total, cap - total);
        if (got < 0) return -1;
        total += got;
        if (hit == nullptr) break;
        if (total >= cap) return -1;
        out[total++] = hit->id;
        text.remove_prefix(at + hit->len);
    }
    return total;
}

int Tokenizer::encode_split(std::string_view text, int32_t* out, int cap) {
    int total = 0;
    while (!text.empty()) {
        std::size_t len = piece_length(text);
        if (len > kMaxPiece) len = kMaxPiece;
        const int got = encode_piece(text.substr(0, len), out + total, cap - total);
        if (got < 0) return -1;
        total += got;
        text.remove_prefix(len);
    }
    return total;
}

}  // namespace lfm
