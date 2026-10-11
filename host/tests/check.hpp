#pragma once

// Minimal test harness: TEST(name) { CHECK(...); CHECK_NEAR(...); }
// Tests self-register; tests/main.cpp runs them all.

#include <cmath>
#include <cstdio>
#include <functional>
#include <span>
#include <vector>

namespace check {

struct Test {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Test>& registry() {
    static std::vector<Test> tests;
    return tests;
}

inline int& failures() {
    static int n = 0;
    return n;
}

struct Register {
    Register(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline void fail(const char* file, int line, const char* what) {
    std::printf("    FAIL %s:%d: %s\n", file, line, what);
    ++failures();
}

inline bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol * (1.0f + std::fabs(b)); }

// Element-wise compare; prints the first mismatch.
inline void spans_near(std::span<const float> got, std::span<const float> want, float tol, const char* file, int line) {
    if (got.size() != want.size()) {
        std::printf("    FAIL %s:%d: size %zu != %zu\n", file, line, got.size(), want.size());
        ++failures();
        return;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!near(got[i], want[i], tol)) {
            std::printf("    FAIL %s:%d: [%zu] got %.7g want %.7g\n", file, line, i, got[i], want[i]);
            ++failures();
            return;
        }
    }
}

}  // namespace check

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                                                  \
    static void name();                                                             \
    static check::Register CHECK_CAT(register_, name){#name, name};                 \
    static void name()

#define CHECK(cond) \
    do { if (!(cond)) check::fail(__FILE__, __LINE__, #cond); } while (0)

#define CHECK_NEAR(got, want, tol)                                                          \
    do {                                                                                    \
        float g_ = (got), w_ = (want);                                                      \
        if (!check::near(g_, w_, (tol))) {                                                  \
            std::printf("    FAIL %s:%d: %s = %.7g, want %.7g\n", __FILE__, __LINE__, #got, g_, w_); \
            ++check::failures();                                                            \
        }                                                                                   \
    } while (0)

#define CHECK_SPAN_NEAR(got, want, tol) check::spans_near((got), (want), (tol), __FILE__, __LINE__)
