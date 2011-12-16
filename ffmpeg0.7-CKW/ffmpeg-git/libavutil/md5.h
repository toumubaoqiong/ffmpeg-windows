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

//****************************************************************************//
//libavutil\md5.h,libavutil\md5.c 
//	获取md5的相关信息
//学习的地方：
//附录：
//1.扩展知识----MD5算法.txt
//****************************************************************************//

#ifndef AVUTIL_MD5_H
#define AVUTIL_MD5_H

#include <stdint.h>
#include "libavutil/attributes.h"
extern const int av_md5_size;

typedef struct AVMD5
{
	uint64_t len;
	uint8_t  block[64];
	uint32_t ABCD[4];
} AVMD5;

FFMPEGLIB_API int av_getav_md5_size(void);
FFMPEGLIB_API void av_md5_init(struct AVMD5 *ctx);
FFMPEGLIB_API void av_md5_update(struct AVMD5 *ctx, const uint8_t *src, const int len);
FFMPEGLIB_API void av_md5_final(struct AVMD5 *ctx, uint8_t *dst);
FFMPEGLIB_API void av_md5_sum(uint8_t *dst, const uint8_t *src, const int len);

#endif /* AVUTIL_MD5_H */

