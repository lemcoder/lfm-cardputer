#include <cstdio>
#include <cstring>

#include "check.hpp"

// Usage: lfm_tests [name-substring]
int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int run = 0, failed = 0;
    for (const auto& t : check::registry()) {
        if (filter && !std::strstr(t.name, filter)) {
            continue;
        }
        int before = check::failures();
        t.fn();
        bool ok = check::failures() == before;
        std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", t.name);
        ++run;
        failed += ok ? 0 : 1;
    }
    std::printf("\n%d/%d passed\n", run - failed, run);
    return failed == 0 ? 0 : 1;
}
