// Windows-only compatibility shim, force-included by nvcc's host pass via
// -Xcompiler=/FI<this-file>. NOT used on Linux (never force-included there).
//
// MSVC does not accept __restrict__ (GCC/clang spelling). Its preprocessor
// also appears to refuse to macro-redefine __restrict__ via /D on the
// command line — hence the explicit #undef + #define inside this header,
// which runs BEFORE any other source file is preprocessed by cl.exe.
//
// Linux builds never include this file, so Linux codegen sees __restrict__
// unchanged and keeps the aliasing hint.

#pragma once

#if defined(_MSC_VER)
#  ifdef __restrict__
#    undef __restrict__
#  endif
#  define __restrict__ __restrict
#endif
