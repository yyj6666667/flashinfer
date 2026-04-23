// FlashInfer cross-platform compatibility macros.
//
// This header is the single destination for portability macros shared by
// framework-agnostic kernel code in ``include/flashinfer/``. Keep it
// lightweight and side-effect-free: no Windows SDK headers, no Torch,
// no CUDA-version probing. It must be safe to include from any ``.cuh``.
//
// When adding a new MSVC / GCC / clang shim, document ``Why`` with the
// concrete symptom it prevents so later readers can judge whether the
// workaround is still needed on newer toolchains.

#ifndef FLASHINFER_COMPAT_H_
#define FLASHINFER_COMPAT_H_

// ---------------------------------------------------------------------------
// FI_RESTRICT
//
// Linux: expand to ``__restrict__`` so the preprocessed token stream matches
// the pre-port code exactly — no perf regression or behavior change on
// GCC/clang device code.
//
// MSVC: the host pass accepts ``__restrict`` on plain pointers but rejects
// ``__restrict__`` and also trips C3646 when a restrict-qualified pointer
// appears as a template parameter. Drop the hint entirely on Windows —
// correctness is unaffected; MSVC-era aliasing optimizations rarely
// exploit the hint anyway.
//
// nvcc's device frontend accepts both spellings, so device kernel
// compilation is unaffected on every platform.
#if defined(_MSC_VER)
#define FI_RESTRICT
#else
#define FI_RESTRICT __restrict__
#endif

// ---------------------------------------------------------------------------
// FI_EXPORT / FI_IMPORT
//
// Windows DLLs do not export ``extern "C"`` symbols by default; ``ctypes``
// (and ``GetProcAddress``) cannot resolve them without ``__declspec
// (dllexport)``. On Linux every visible symbol is exported already, so
// the macro is a no-op there.
//
// Use ``FI_EXPORT`` on C ABI entry points that Python code resolves via
// ctypes (e.g. callbacks registered from ``flashinfer/jit/cubin_loader.py``).
// ``FI_IMPORT`` is provided symmetrically for future consumers that need
// to declare an imported symbol on Windows.
#if defined(_WIN32)
#define FI_EXPORT __declspec(dllexport)
#define FI_IMPORT __declspec(dllimport)
#else
#define FI_EXPORT
#define FI_IMPORT
#endif

#endif  // FLASHINFER_COMPAT_H_
