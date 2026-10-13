#pragma once
// On-device check of the PIE (SIMD) dot kernels against the C reference:
// random rows at every byte misalignment, exact equality, and timing.
// Enabled with kKernelTest in main.cpp.

using PrintFn = void (*)(const char* line);

// Returns true if every PIE result matched the reference exactly.
bool kernel_test(PrintFn print);

// Memory-mapped flash: read bandwidth vs the LM head table scan.
void flash_scan_bench(PrintFn print);
