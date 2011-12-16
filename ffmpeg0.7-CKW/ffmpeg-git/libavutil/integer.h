/*
 * arbitrary precision integers
 * Copyright (c) 2004 Michael Niedermayer <michaelni@gmx.at>
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

//****************************************************************************//
//libavutil\integer.h， libavutil\integer.c
//	本内容主要是任意精度的运算
//学习的地方：
//1.任意精度的运算，以前本人也编写过，但是那是运用字符来完成的，
//学习一下这里是如何设计的
//2.阅读了一边，代码逻辑还是很清晰的，主要靠移位来实现的，这样效率也会比较高
//附录：
//1.
//****************************************************************************//

/**
 * @file
 * arbitrary precision integers
 * @author Michael Niedermayer <michaelni@gmx.at>
 */

#ifndef AVUTIL_INTEGER_H
#define AVUTIL_INTEGER_H

#include <stdint.h>
#include "common.h"

#define AV_INTEGER_SIZE 8

FFMPEGLIB_API typedef struct AVInteger
{
    uint16_t v[AV_INTEGER_SIZE];
} AVInteger;

FFMPEGLIB_API AVInteger av_add_i(AVInteger a, AVInteger b) av_const;
FFMPEGLIB_API AVInteger av_sub_i(AVInteger a, AVInteger b) av_const;

/**
 * Return the rounded-down value of the base 2 logarithm of the given
 * AVInteger. This is simply the index of the most significant bit
 * which is 1, or 0 if all bits are 0.
 */
FFMPEGLIB_API int av_log2_i(AVInteger a) av_const;
FFMPEGLIB_API AVInteger av_mul_i(AVInteger a, AVInteger b) av_const;

/**
 * Return 0 if a==b, 1 if a>b and -1 if a<b.
 */
FFMPEGLIB_API int av_cmp_i(AVInteger a, AVInteger b) av_const;

/**
 * bitwise shift
 * @param s the number of bits by which the value should be shifted right,
            may be negative for shifting left
 */
FFMPEGLIB_API AVInteger av_shr_i(AVInteger a, int s) av_const;

/**
 * Return a % b.
 * @param quot a/b will be stored here.
 */
FFMPEGLIB_API AVInteger av_mod_i(AVInteger *quot, AVInteger a, AVInteger b);

/**
 * Return a/b.
 */
FFMPEGLIB_API AVInteger av_div_i(AVInteger a, AVInteger b) av_const;

/**
 * Convert the given int64_t to an AVInteger.
 */
FFMPEGLIB_API AVInteger av_int2i(int64_t a) av_const;

/**
 * Convert the given AVInteger to an int64_t.
 * If the AVInteger is too large to fit into an int64_t,
 * then only the least significant 64 bits will be used.
 */
FFMPEGLIB_API int64_t av_i2int(AVInteger a) av_const;

#endif /* AVUTIL_INTEGER_H */
