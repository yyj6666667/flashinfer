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

// Windows-only helpers consumed via -Xcompiler=/FI.
//
// (Formerly attempted to redefine __restrict__; MSVC reserves the token
// at a level below the preprocessor — both -D and /FI + #define are
// silently ignored on it — so that path was abandoned. Source code uses
// the FI_RESTRICT macro instead, defined in include/flashinfer/_compat.h,
// which is Linux-token-equivalent.)

// NOTE: Do NOT add `#include <ciso646>` here. On MSVC that header drags in
// <yvals_core.h> and half of the CRT *before* nvcc's host pass emits its
// own `#line`-preserved CRT includes, which then fires dozens of C2011
// (struct/enum redefinition) errors deep inside <corecrt.h>. Use
// `-Xcompiler=/permissive-` on a per-module basis when operator keywords
// (`and`/`or`/`not`) need recognising — the flag lives in gen_*_module()
// next to the code that actually relies on it.

#pragma once
