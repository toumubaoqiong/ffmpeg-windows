/*
 * copyright (c) 2006 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * memory handling functions
 */

//****************************************************************************//
//libavutil\mem.h,libavutil\mem.c
//	在原始内存分配的函数的基础上做了一些封装
//学习的地方：
//1.在分配内存的时候进行对齐操作
//附录：
//1.
//****************************************************************************//

#ifndef AVUTIL_MEM_H
#define AVUTIL_MEM_H

#include "attributes.h"
#include "avutil.h"


#define DECLARE_ALIGNED(n,t,v)      t v
#define DECLARE_ASM_CONST(n,t,v)    static const t v

//#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1110 || defined(__SUNPRO_C)
//    #define DECLARE_ALIGNED(n,t,v)      t __attribute__ ((aligned (n))) v
//    #define DECLARE_ASM_CONST(n,t,v)    const t __attribute__ ((aligned (n))) v
//#elif defined(__TI_COMPILER_VERSION__)
//    #define DECLARE_ALIGNED(n,t,v)                      \
//        AV_PRAGMA(DATA_ALIGN(v,n))                      \
//        t __attribute__((aligned(n))) v
//    #define DECLARE_ASM_CONST(n,t,v)                    \
//        AV_PRAGMA(DATA_ALIGN(v,n))                      \
//        static const t __attribute__((aligned(n))) v
//#elif defined(__GNUC__)
//    #define DECLARE_ALIGNED(n,t,v)      t __attribute__ ((aligned (n))) v
//    #define DECLARE_ASM_CONST(n,t,v)    static const t av_used __attribute__ ((aligned (n))) v
//#elif defined(_MSC_VER)
//    #define DECLARE_ALIGNED(n,t,v)      __declspec(align(n)) t v
//    #define DECLARE_ASM_CONST(n,t,v)    __declspec(align(n)) static const t v
//#else
//    #define DECLARE_ALIGNED(n,t,v)      t v
//    #define DECLARE_ASM_CONST(n,t,v)    static const t v
//#endif

#if AV_GCC_VERSION_AT_LEAST(3,1)
#define av_malloc_attrib __attribute__((__malloc__))
#else
#define av_malloc_attrib
#endif

#if AV_GCC_VERSION_AT_LEAST(4,3)
#define av_alloc_size(n) __attribute__((alloc_size(n)))
#else
#define av_alloc_size(n)
#endif

#if LIBAVUTIL_VERSION_MAJOR < 51
#   define FF_INTERNAL_MEM_TYPE unsigned int
#   define FF_INTERNAL_MEM_TYPE_MAX_VALUE UINT_MAX
#else
#   define FF_INTERNAL_MEM_TYPE size_t
#   define FF_INTERNAL_MEM_TYPE_MAX_VALUE SIZE_MAX
#endif

FFMPEGLIB_API int av_mem_checkBy_memwatch(const int logInt);

/**
 * Allocate a block of size bytes with alignment suitable for all
 * memory accesses (including vectors if available on the CPU).
 * @param size Size in bytes for the memory block to be allocated.
 * @return Pointer to the allocated block, NULL if the block cannot
 * be allocated.
 * @see av_mallocz()
 */
FFMPEGLIB_API void *av_malloc(FF_INTERNAL_MEM_TYPE size) av_malloc_attrib av_alloc_size(1);

/**
 * Allocate or reallocate a block of memory.
 * If ptr is NULL and size > 0, allocate a new block. If
 * size is zero, free the memory block pointed to by ptr.
 * @param size Size in bytes for the memory block to be allocated or
 * reallocated.
 * @param ptr Pointer to a memory block already allocated with
 * av_malloc(z)() or av_realloc() or NULL.
 * @return Pointer to a newly reallocated block or NULL if the block
 * cannot be reallocated or the function is used to free the memory block.
 * @see av_fast_realloc()
 */
FFMPEGLIB_API void *av_realloc(void *ptr, FF_INTERNAL_MEM_TYPE size) av_alloc_size(2);

/**
 * Free a memory block which has been allocated with av_malloc(z)() or
 * av_realloc().
 * @param ptr Pointer to the memory block which should be freed.
 * @note ptr = NULL is explicitly allowed.
 * @note It is recommended that you use av_freep() instead.
 * @see av_freep()
 */
FFMPEGLIB_API void av_free(void *ptr);

/**
 * Allocate a block of size bytes with alignment suitable for all
 * memory accesses (including vectors if available on the CPU) and
 * zero all the bytes of the block.
 * @param size Size in bytes for the memory block to be allocated.
 * @return Pointer to the allocated block, NULL if it cannot be allocated.
 * @see av_malloc()
 */
FFMPEGLIB_API void *av_mallocz(FF_INTERNAL_MEM_TYPE size) av_malloc_attrib av_alloc_size(1);

/**
 * Duplicate the string s.
 * @param s string to be duplicated
 * @return Pointer to a newly allocated string containing a
 * copy of s or NULL if the string cannot be allocated.
 */
FFMPEGLIB_API char *av_strdup(const char *s) av_malloc_attrib;

/**
 * Free a memory block which has been allocated with av_malloc(z)() or
 * av_realloc() and set the pointer pointing to it to NULL.
 * @param ptr Pointer to the pointer to the memory block which should
 * be freed.
 * @see av_free()
 */
FFMPEGLIB_API void av_freep(void *ptr);

#endif /* AVUTIL_MEM_H */
