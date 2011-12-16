/*
 * copyright (c) 2007 Michael Niedermayer <michaelni@gmx.at>
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
//libavutil\aes.h,libavutil\aes.c
//	生成AES加密的一切信息
//学习的地方：
//1.
//附录：
//1.扩展知识----AES加密算法(C++实现).doc
//2.扩展知识----AES加密算法(C++实现源代码).rar
//3.扩展知识----AES算法的密码分析与快速实现.doc
//****************************************************************************//

#ifndef AVUTIL_AES_H
#define AVUTIL_AES_H

#include <stdint.h>
#include "libavutil/attributes.h"

extern const int av_aes_size;
FFMPEGLIB_API const int av_getav_aes_size(void);


FFMPEGLIB_API typedef union
{
	uint64_t u64[2];
	uint32_t u32[4];
	uint8_t u8x4[4][4];
	uint8_t u8[16];
} av_aes_block;

FFMPEGLIB_API typedef struct AVAES
{
	// Note: round_key[16] is accessed in the init code, but this only
	// overwrites state, which does not matter (see also r7471).
	av_aes_block round_key[15];
	av_aes_block state[2];
	int rounds;
} AVAES;


/**
 * Initialize an AVAES context.
 * @param key_bits 128, 192 or 256
 * @param decrypt 0 for encryption, 1 for decryption
 */
FFMPEGLIB_API int av_aes_init(struct AVAES *a, const uint8_t *key, int key_bits, int decrypt);

/**
 * Encrypt or decrypt a buffer using a previously initialized context.
 * @param count number of 16 byte blocks
 * @param dst destination array, can be equal to src
 * @param src source array, can be equal to dst
 * @param iv initialization vector for CBC mode, if NULL then ECB will be used
 * @param decrypt 0 for encryption, 1 for decryption
 */
FFMPEGLIB_API void av_aes_crypt(struct AVAES *a, uint8_t *dst, const uint8_t *src, int count, uint8_t *iv, int decrypt);

#endif /* AVUTIL_AES_H */
