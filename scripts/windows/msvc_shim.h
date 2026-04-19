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
#  pragma message("FLASHINFER: msvc_shim.h applied")
#  ifdef __restrict__
#    pragma message("FLASHINFER: __restrict__ was already a macro, undefing")
#    undef __restrict__
#  endif
// Diagnostic: define as empty. If the host-side parse errors on
// __restrict__ disappear, then substitution IS happening and the
// problem is that __restrict (MSVC keyword) is disallowed in template
// parameter positions. If errors persist, substitution isn't reaching
// the error site.
#  define __restrict__
#  pragma message("FLASHINFER: __restrict__ now defined as EMPTY")
#endif
