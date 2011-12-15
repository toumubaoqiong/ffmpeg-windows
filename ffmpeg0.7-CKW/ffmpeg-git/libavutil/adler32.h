/*
 * copyright (c) 2006 Mans Rullgard
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

#ifndef AVUTIL_ADLER32_H
#define AVUTIL_ADLER32_H

#include <stdint.h>
#include "attributes.h"

//****************************************************************************//
//libavutil\adler32.h，libavutil\adler32.c
//  主要作用就是获取adler32校验和
//学习的地方：
//1.如何计算adler32校验和
//附录：
//1.扩展知识----Hash 函数的常用算法和应用领域.txt
//****************************************************************************//

/**
 * Calculate the Adler32 checksum of a buffer.
 *
 * Passing the return value to a subsequent av_adler32_update() call
 * allows the checksum of multiple buffers to be calculated as though
 * they were concatenated.
 *
 * @param adler initial checksum value
 * @param buf   pointer to input buffer
 * @param len   size of input buffer
 * @return      updated checksum
 */
FFMPEGLIB_API unsigned long av_adler32_update(unsigned long adler, const uint8_t *buf,
        unsigned int len) av_pure;

#endif /* AVUTIL_ADLER32_H */
