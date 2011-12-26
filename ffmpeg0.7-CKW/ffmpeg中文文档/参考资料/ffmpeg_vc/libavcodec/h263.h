/*
 * H263/MPEG4 backend for ffmpeg encoder and decoder
 * copyright (c) 2007 Aurelien Jacobs <aurel@gnuage.org>
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

#ifndef AVCODEC_H263_H
#define AVCODEC_H263_H

#include "../config.h"
#include "msmpeg4.h"

#define ENABLE_H263_DECODER 1
#define ENABLE_H263I_DECODER 1
#define ENABLE_FLV_DECODER 1
#define ENABLE_RV10_DECODER 1
#define ENABLE_RV20_DECODER 1
#define ENABLE_MPEG4_DECODER 1
//#define ENABLE_MSMPEG4_DECODER 1
//#define ENABLE_WMV_DECODER 1


#define ENABLE_H264_DECODER 1
#define ENABLE_VP3_DECODER 0
#define ENABLE_VP5_DECODER 0
#define ENABLE_VP6_DECODER 0
#define ENABLE_THEORA_DECODER 0
#define ENABLE_EATGQ_DECODER 0
#define ENABLE_MMX 0
#define ENABLE_ARMV4L 0
#define ENABLE_MLIB 0
#define ENABLE_VIS 0
#define ENABLE_ALPHA 0
#define ENABLE_POWERPC 0
#define ENABLE_MMI 0
#define ENABLE_SH4 0
#define ENABLE_BFIN 0


#define ENABLE_H263_ENCODER 0
#define ENABLE_H263P_ENCODER 0
#define ENABLE_FLV_ENCODER 0
#define ENABLE_RV10_ENCODER 0
#define ENABLE_RV20_ENCODER 0
#define ENABLE_MPEG4_ENCODER 0
//#define ENABLE_MSMPEG4_ENCODER 0
//#define ENABLE_WMV_ENCODER 0

#define ENABLE_ANY_H263_DECODER (ENABLE_H263_DECODER    || \
                                 ENABLE_H263I_DECODER   || \
                                 ENABLE_FLV_DECODER     || \
                                 ENABLE_RV10_DECODER    || \
                                 ENABLE_RV20_DECODER    || \
                                 ENABLE_MPEG4_DECODER   || \
                                 ENABLE_MSMPEG4_DECODER || \
                                 ENABLE_WMV_DECODER)
#define ENABLE_ANY_H263_ENCODER 1/*(ENABLE_H263_ENCODER    || \
                                 ENABLE_H263P_ENCODER   || \
                                 ENABLE_FLV_ENCODER     || \
                                 ENABLE_RV10_ENCODER    || \
                                 ENABLE_RV20_ENCODER    || \
                                 ENABLE_MPEG4_ENCODER   || \
                                 ENABLE_MSMPEG4_ENCODER || \
                                 ENABLE_WMV_ENCODER)*/
#define ENABLE_ANY_H263 1//(ENABLE_ANY_H263_DECODER || ENABLE_ANY_H263_ENCODER)

#endif /* AVCODEC_H263_H */
