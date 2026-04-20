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

// Enable the C++ alternative operator keywords (and/or/not/xor/compl/...).
// MSVC's default mode (without /permissive- or /Za) disables these tokens;
// TensorRT-LLM headers pulled in from csrc/nv_internal use them freely.
// <ciso646> (standard since C++98) defines them as macros. GCC/Clang
// recognise the tokens natively, so including the header has no effect.
#include <ciso646>

#pragma once
