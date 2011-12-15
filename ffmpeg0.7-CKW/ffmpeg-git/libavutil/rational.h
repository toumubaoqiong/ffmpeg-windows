/*
 * rational numbers
 * Copyright (c) 2003 Michael Niedermayer <michaelni@gmx.at>
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
 * rational numbers
 * @author Michael Niedermayer <michaelni@gmx.at>
 */

#ifndef AVUTIL_RATIONAL_H
#define AVUTIL_RATIONAL_H

#include <stdint.h>
#include <limits.h>
#include "attributes.h"
#include "_types.h"


//****************************************************************************//
//libavutil\rational.h, libavutil\rational.c:
//计算AVRational所有相关计算
//学习的地方：
//1.两个英文字幕：numerator分子简称num, denominator分母简称den
//2.两个分数的比较方式：A.num*B.den 与 B.num*A.den的比较
//3.不同类型的运算的数据转变技巧：int A; double B; double c = A/(double)B;
//4.从AVRational的所有相关计算函数中可以得到一个信息，
//大多数计算函数都是在av_reduce基础上得到的
//在第二个运算数之前添加结果的类型，第一个不用添加
//附录：
//1.扩展知识----C语言位运算详解.txt
//****************************************************************************//


/**
 * rational number numerator/denominator
 */
typedef struct AVRational
{
    int num; ///< numerator
    int den; ///< denominator
} AVRational;

/**
 * Compare two rationals.
 * @param a first rational
 * @param b second rational
 * @return 0 if a==b, 1 if a>b, -1 if a<b, and INT_MIN if one of the
 * values is of the form 0/0
 */
static inline int av_cmp_q(AVRational a, AVRational b)
{
    const int64_t tmp = 
		a.num * (int64_t)b.den - b.num * (int64_t)a.den;

    if(tmp) 
	{
		//这里仍然无法弄明白？
		return ((tmp ^ a.den ^ b.den) >> 63) | 1;
	}
    else if(b.den && a.den) 
	{
		return 0;
	}
    else if(a.num && b.num) 
	{
		//避免因为分母都是零导致相等
		return (a.num >> 31) - (b.num >> 31);
	}
    else
	{
		return INT_MIN;
	}
}

/**
 * Convert rational to double.
 * @param a rational to convert
 * @return (double) a
 */
static inline double av_q2d(AVRational a)
{
    return a.num / (double) a.den;
}

/**
 * Reduce a fraction.
 * This is useful for framerate calculations.
 * @param dst_num destination numerator
 * @param dst_den destination denominator
 * @param num source numerator
 * @param den source denominator
 * @param max the maximum allowed for dst_num & dst_den
 * @return 1 if exact, 0 otherwise
 */
//这个函数我看的太神秘了，实际上就是以最大值为限，将*dst_num = num , *dst_den = den
FFMPEGLIB_API int av_reduce(int *dst_num, int *dst_den, int64_t num, int64_t den, int64_t max);

/**
 * Multiply two rationals.
 * @param b first rational
 * @param c second rational
 * @return b*c
 */
FFMPEGLIB_API AVRational av_mul_q(AVRational b, AVRational c) av_const;

/**
 * Divide one rational by another.
 * @param b first rational
 * @param c second rational
 * @return b/c
 */
FFMPEGLIB_API AVRational av_div_q(AVRational b, AVRational c) av_const;

/**
 * Add two rationals.
 * @param b first rational
 * @param c second rational
 * @return b+c
 */
FFMPEGLIB_API AVRational av_add_q(AVRational b, AVRational c) av_const;

/**
 * Subtract one rational from another.
 * @param b first rational
 * @param c second rational
 * @return b-c
 */
FFMPEGLIB_API AVRational av_sub_q(AVRational b, AVRational c) av_const;

/**
 * Convert a double precision floating point number to a rational.
 * inf is expressed as {1,0} or {-1,0} depending on the sign.
 *
 * @param d double to convert
 * @param max the maximum allowed numerator and denominator
 * @return (AVRational) d
 */
FFMPEGLIB_API AVRational av_d2q(double d, int max) av_const;

/**
 * @return 1 if q1 is nearer to q than q2, -1 if q2 is nearer
 * than q1, 0 if they have the same distance.
 */
FFMPEGLIB_API int av_nearer_q(AVRational q, AVRational q1, AVRational q2);

/**
 * Find the nearest value in q_list to q.
 * @param q_list an array of rationals terminated by {0, 0}
 * @return the index of the nearest value found in the array
 */
FFMPEGLIB_API int av_find_nearest_q_idx(AVRational q, const AVRational *q_list);

FFMPEGLIB_API int av_rational_test_local(void);

#endif /* AVUTIL_RATIONAL_H */
