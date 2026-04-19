// Portability compat macros for FlashInfer.
//
// Linux builds expand FI_RESTRICT to __restrict__ so the preprocessed
// token stream matches the pre-port code exactly — no perf regression
// and no behavior change relative to GCC/clang.
//
// Windows MSVC host pass doesn't accept __restrict__ (reserves the
// token at a level below the preprocessor), but does accept __restrict.
// We pick that spelling only there.
//
// nvcc's device frontend accepts both spellings, so device kernel
// compilation is unaffected on every platform.

#ifndef FLASHINFER_COMPAT_H_
#define FLASHINFER_COMPAT_H_

#if defined(_MSC_VER)
// MSVC accepts __restrict on plain variables but rejects it in
// template function parameter positions (observed via C3646 on
// flashinfer template host functions). Drop the hint entirely on
// Windows — correctness is unaffected; aliasing optimizations lose a
// hint that nvcc/cl rarely exploits aggressively anyway.
#define FI_RESTRICT
#else
#define FI_RESTRICT __restrict__
#endif

#endif  // FLASHINFER_COMPAT_H_
