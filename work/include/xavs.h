/*****************************************************************************
 * x264.h: h264 encoder library
 *****************************************************************************
 * Copyright (C) 2003-2008 x264 Project
 *
 * Authors: Laurent Aimar <fenrir@via.ecp.fr>
 *          Loren Merritt <lorenm@u.washington.edu>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *****************************************************************************/

/*****************************************************************************
 * xavs.h: xavs encoder library
 *****************************************************************************
 * Copyright (C) 2009~2010 xavs project
 * Authors: Jianwen Chen <jianwen.chen.video@gmail.com>
 * This code is modified on x264 project and will follow the license of x264
 *****************************************************************************/

#ifndef _XAVS_XAVS_H_
#define _XAVS_XAVS_H_

#if !defined(_STDINT_H) && !defined(_STDINT_H_) && \
    !defined(_INTTYPES_H) && !defined(_INTTYPES_H_)
# ifdef _MSC_VER
#  pragma message("You must include stdint.h or inttypes.h before xavs.h")
# else
#  warning You must include stdint.h or inttypes.h before xavs.h
# endif
#endif

#include <stdarg.h>

#define XAVS_BUILD 1

//#define AVIS_INPUT

/* xavs_t:
 *      opaque handler for decoder and encoder */
typedef struct xavs_t xavs_t;

/****************************************************************************
 * Initialisation structure and function.
 ****************************************************************************/
/* CPU flags
 */
#define XAVS_CPU_3DNOW      0x000010    /* 3dnow! */
#define XAVS_CPU_3DNOWEXT   0x000020    /* 3dnow! ext */
#define XAVS_CPU_CACHELINE_32   0x000001        /* avoid memory loads that span the border between two cachelines */
#define XAVS_CPU_CACHELINE_64   0x000002        /* 32/64 is the size of a cacheline in bytes */
#define XAVS_CPU_ALTIVEC        0x000004
#define XAVS_CPU_MMX            0x000008
#define XAVS_CPU_MMXEXT         0x000010        /* MMX2 aka MMXEXT aka ISSE */
#define XAVS_CPU_SSE            0x000020
#define XAVS_CPU_SSE2           0x000040
#define XAVS_CPU_SSE2_IS_SLOW   0x000080        /* avoid most SSE2 functions on Athlon64 */
#define XAVS_CPU_SSE2_IS_FAST   0x000100        /* a few functions are only faster on Core2 and Phenom */
#define XAVS_CPU_SSE3           0x000200
#define XAVS_CPU_SSSE3          0x000400
#define XAVS_CPU_SHUFFLE_IS_FAST 0x000800       /* Penryn, Nehalem, and Phenom have fast shuffle units */
#define XAVS_CPU_STACK_MOD4     0x001000        /* if stack is only mod4 and not mod16 */
#define XAVS_CPU_SSE4           0x002000        /* SSE4.1 */
#define XAVS_CPU_SSE42          0x004000        /* SSE4.2 */
#define XAVS_CPU_SSE_MISALIGN   0x008000        /* Phenom support for misaligned SSE instruction arguments */
#define XAVS_CPU_LZCNT          0x010000        /* Phenom support for "leading zero count" instruction. */
#define XAVS_CPU_ARMV6          0x020000
#define XAVS_CPU_NEON           0x040000        /* ARM NEON */
#define XAVS_CPU_FAST_NEON_MRC  0x080000        /* Transfer from NEON to ARM register is fast (Cortex-A9) */

/* Analyse flags
 */
#define XAVS_ANALYSE_I4x4       0x0001  /* Analyse i4x4 */
#define XAVS_ANALYSE_I8x8       0x0002  /* Analyse i8x8 (requires 8x8 transform) */
#define XAVS_ANALYSE_PSUB16x16  0x0010  /* Analyse p16x8, p8x16 and p8x8 */
#define XAVS_ANALYSE_PSUB8x8    0x0020  /* Analyse p8x4, p4x8, p4x4 */
#define XAVS_ANALYSE_BSUB16x16  0x0100  /* Analyse b16x8, b8x16 and b8x8 */
#define XAVS_DIRECT_PRED_NONE        0
#define XAVS_DIRECT_PRED_SPATIAL     1
#define XAVS_DIRECT_PRED_TEMPORAL    2
#define XAVS_DIRECT_PRED_AUTO        3
#define XAVS_ME_DIA                  0
#define XAVS_ME_HEX                  1
#define XAVS_ME_UMH                  2
#define XAVS_ME_ESA                  3
#define XAVS_ME_TESA                 4
#define XAVS_CQM_FLAT                0
#define XAVS_CQM_JVT                 1
#define XAVS_CQM_CUSTOM              2

#define XAVS_RC_CQP                  0
#define XAVS_RC_CRF                  1
#define XAVS_RC_ABR                  2
#define XAVS_AQ_NONE                 0
#define XAVS_AQ_VARIANCE             1
#define XAVS_AQ_AUTOVARIANCE         2
#define XAVS_B_ADAPT_NONE            0
#define XAVS_B_ADAPT_FAST            1
#define XAVS_B_ADAPT_TRELLIS         2
static const char *const xavs_direct_pred_names[] = { "none", "spatial", "temporal", "auto", 0 };
static const char *const xavs_motion_est_names[] = { "dia", "hex", "umh", "esa", 0 };
static const char *const xavs_overscan_names[] = { "undef", "show", "crop", 0 };
static const char *const xavs_vidformat_names[] = { "component", "pal", "ntsc", "secam", "mac", "undef", 0 };
static const char *const xavs_fullrange_names[] = { "off", "on", 0 };
static const char *const xavs_colorprim_names[] = { "", "bt709", "undef", "", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", 0 };
static const char *const xavs_transfer_names[] = { "", "bt709", "undef", "", "bt470m", "bt470bg", "smpte170m", "smpte240m", "linear", "log100", "log316", 0 };
static const char *const xavs_colmatrix_names[] = { "GBR", "bt709", "undef", "", "fcc", "bt470bg", "smpte170m", "smpte240m", "YCgCo", 0 };

/* Colorspace type
 */
#define XAVS_CSP_MASK           0x00ff  /* */
#define XAVS_CSP_NONE           0x0000  /* Invalid mode     */
#define XAVS_CSP_I420           0x0001  /* yuv 4:2:0 planar */
#define XAVS_CSP_I422           0x0002  /* yuv 4:2:2 planar */
#define XAVS_CSP_I444           0x0003  /* yuv 4:4:4 planar */
#define XAVS_CSP_YV12           0x0004  /* yuv 4:2:0 planar */
#define XAVS_CSP_YUYV           0x0005  /* yuv 4:2:2 packed */
#define XAVS_CSP_RGB            0x0006  /* rgb 24bits       */
#define XAVS_CSP_BGR            0x0007  /* bgr 24bits       */
#define XAVS_CSP_BGRA           0x0008  /* bgr 32bits       */
#define XAVS_CSP_MAX            0x0009  /* end of list */
#define XAVS_CSP_VFLIP          0x1000  /* */

/* Slice type
 */
#define XAVS_TYPE_AUTO          0x0000  /* Let xavs choose the right type */
#define XAVS_TYPE_IDR           0x0001
#define XAVS_TYPE_I             0x0002
#define XAVS_TYPE_P             0x0003
#define XAVS_TYPE_BREF          0x0004  /* Non-disposable B-frame */
#define XAVS_TYPE_B             0x0005
#define IS_XAVS_TYPE_I(x) ((x)==XAVS_TYPE_I || (x)==XAVS_TYPE_IDR)
#define IS_XAVS_TYPE_B(x) ((x)==XAVS_TYPE_B || (x)==XAVS_TYPE_BREF)

/* Log level
 */
#define XAVS_LOG_NONE          (-1)
#define XAVS_LOG_ERROR          0
#define XAVS_LOG_WARNING        1
#define XAVS_LOG_INFO           2
#define XAVS_LOG_DEBUG          3

/* Threading */
#define XAVS_THREADS_AUTO 0     /* Automatically select optimal number of threads */
#define XAVS_SYNC_LOOKAHEAD_AUTO (-1)   /* Automatically select optimal lookahead thread buffer size */
typedef struct
{
  int i_start, i_end;           /* range of frame numbers */
  int b_force_qp;               /* whether to use qp vs bitrate factor */
  int i_qp;
  float f_bitrate_factor;
  struct xavs_param_t *param;
} xavs_zone_t;

typedef struct xavs_param_t
{
  /* CPU flags */
  unsigned int cpu;
  int i_threads;                /* encode multiple frames in parallel */
  int b_sliced_threads;         /* Whether to use slice-based threading. */
  int b_deterministic;          /* whether to allow non-deterministic optimizations when threaded */
  int i_sync_lookahead;         /* threaded lookahead buffer */

  /* Video Properties */
  int i_width;
  int i_height;
  int i_csp;                    /* CSP of encoded bitstream, only i420 supported */
  int i_level_idc;
  int i_frame_total;            /* number of frames to encode if known, else 0 */

  struct
  {
    /* they will be reduced to be 0 < x <= 65535 and prime */
    int i_sar_height;
    int i_sar_width;

    int i_overscan;             /* 0=undef, 1=no overscan, 2=overscan */

    /* see avs annex E for the values of the following */
    int i_vidformat;
    int b_fullrange;
    int i_colorprim;
    int i_transfer;
    int i_colmatrix;
    int i_chroma_loc;           /* both top & bottom */
  } vui;

  int i_fps_num;
  int i_fps_den;

  /* Bitstream parameters */
  int i_frame_reference;        /* Maximum number of reference frames */
  int i_keyint_max;             /* Force an IDR keyframe at this interval */
  int i_keyint_min;             /* Scenecuts closer together than this are coded as I, not IDR. */
  int i_scenecut_threshold;     /* how aggressively to insert extra I frames */
  int i_bframe;                 /* how many b-frame between 2 references pictures */
  int i_bframe_adaptive;
  int i_bframe_bias;
  int b_bframe_pyramid;         /* Keep some B-frames as references */

  int b_deblocking_filter;
  int i_deblocking_filter_alphac0;      /* [-6, 6] -6 light filter, 6 strong */
  int i_deblocking_filter_beta; /* [-6, 6]  idem */

  int b_cabac;
  int i_cabac_init_idc;
  int b_interlaced;
  int b_constrained_intra;

  int i_cqm_preset;
  char *psz_cqm_file;           /* JM format */
  uint8_t cqm_4iy[16];          /* used only if i_cqm_preset == XAVS_CQM_CUSTOM */
  uint8_t cqm_4ic[16];
  uint8_t cqm_4py[16];
  uint8_t cqm_4pc[16];
  uint8_t cqm_8iy[64];
  uint8_t cqm_8py[64];

  /* Log */
  void (*pf_log) (void *, int i_level, const char *psz, va_list);
  void *p_log_private;
  int i_log_level;
  int b_visualize;
  char *psz_dump_yuv;           /* filename for reconstructed frames */

  /* Encoder analyser parameters */
  struct
  {
    unsigned int intra;         /* intra partitions */
    unsigned int inter;         /* inter partitions */

    int b_transform_8x8;
    int b_weighted_bipred;      /* implicit weighting for B-frames */
    int i_direct_mv_pred;       /* spatial vs temporal mv prediction */
    int i_chroma_qp_offset;

    int i_me_method;            /* motion estimation algorithm to use (xavs_ME_*) */
    int i_me_range;             /* integer pixel motion estimation search range (from predicted mv) */
    int i_mv_range;             /* maximum length of a mv (in pixels) */
    int i_mv_range_thread;      /* minimum space between threads. -1 = auto, based on number of threads. */
    int i_subpel_refine;        /* subpixel motion estimation quality */
    int b_bidir_me;             /* jointly optimize both MVs in B-frames */
    int b_chroma_me;            /* chroma ME for subpel and mode decision in P-frames */
    int b_bframe_rdo;           /* RD based mode decision for B-frames */
    int b_mixed_references;     /* allow each mb partition in P-frames to have it's own reference number */
    int i_trellis;              /* trellis RD quantization */
    int b_fast_pskip;           /* early SKIP detection on P-frames */
    int b_dct_decimate;         /* transform coefficient thresholding on P-frames */
    int i_noise_reduction;      /* adaptive pseudo-deadzone */
    float f_psy_rd;             /* Psy RD strength */
    float f_psy_trellis;        /* Psy trellis strength */
    int b_psy;                  /* Toggle all psy optimizations */

    /* the deadzone size that will be used in luma quantization */
    int i_luma_deadzone[2];     /* {intra, inter} */

    int b_psnr;                 /* compute and print PSNR stats */
    int b_skip_mode;
    int b_ssim;                 /* compute and print SSIM stats */
  } analyse;

  /* Rate control parameters */
  struct
  {
    int i_rc_method;            /* XAVS_RC_* */
    int i_qp_constant;          /* 0-63 */
    int i_qp_min;               /* min allowed QP value */
    int i_qp_max;               /* max allowed QP value */
    int i_qp_step;              /* max QP step between frames */

    int i_bitrate;
    float f_rf_constant;        /* 1pass VBR, nominal QP */
    float f_rate_tolerance;
    int i_vbv_max_bitrate;
    int i_vbv_buffer_size;
    float f_vbv_buffer_init;    /* <=1: fraction of buffer_size. >1: kbit */
    float f_ip_factor;
    float f_pb_factor;

    int i_aq_mode;              /* psy adaptive QP. (XAVS_AQ_*) */
    float f_aq_strength;
    int b_mb_tree;              /* Macroblock-tree ratecontrol. */
    int i_lookahead;
    /* 2pass */
    int b_stat_write;           /* Enable stat writing in psz_stat_out */
    char *psz_stat_out;
    int b_stat_read;            /* Read stat from psz_stat_in and use it */
    char *psz_stat_in;

    /* 2pass params (same as ffmpeg ones) */
    float f_qcompress;          /* 0.0 => cbr, 1.0 => constant qp */
    float f_qblur;              /* temporally blur quants */
    float f_complexity_blur;    /* temporally blur complexity */
    xavs_zone_t *zones;         /* ratecontrol overrides */
    int i_zones;                /* sumber of zone_t's */
    char *psz_zones;            /* alternate method of specifying zones */
  } rc;

  int b_aud;                    /* generate access unit delimiters */
  int b_repeat_headers;         /* put SPS/PPS before each keyframe */
  int b_annexb;                 /* if set, place start codes (4 bytes) before NAL units,
                                 * otherwise place size (4 bytes) before NAL units. */
  int i_sps_id;                 /* SPS and PPS id number */

  int i_chroma_format;          /* 1: 4:2:0, 2: 4:2:2 */
  int i_sample_precision;       /* 1: 8 bits per sample */
  int i_aspect_ratio;           /* '0001':1/1, '0010':4/3, '0011': 16/9, '0100':2.21/ 1 */

} xavs_param_t;

typedef struct
{
  //int mv_range;    // max vertical mv component range (pixels)
  int level_idc;
  int samples_per_row;
  int lines_per_frame;
  int frames_per_second;
  int luma_samples_per_second;
  int bitrate;                  // max bitrate (kbit/sec) 
  int cpb;                      // max vbv buffer (kbit) 
  int frame_size;               // max frame size (macroblocks) 
  int mbps;                     // max macroblock processing rate (macroblocks/sec) 
  float frame_ver_mv_range_low; // max vertical mv component range (pixels) for frame
  float mv_range;               //frame_ver_mv_range_high;   // max vertical mv component range (pixels) for frame
  float field_ver_mv_range_low; // max vertical mv component range (pixels) for field
  float field_ver_mv_range_high;        // max vertical mv component range (pixels) for field
  float hor_mv_range_low;       // max horizontal mv component range (pixels) 
  float hor_mv_range_high;      // max horizontal mv component range (pixels) 
  int pic_format;               // 0:4:2:0, 1:4:2:0 or 4:2:2 
  int frame_only;               // forbid interlacing 
  int bits_per_mb_420;          // max bits per MB after coded for 4:2:0 
  int bits_per_mb_422;          // max bits per MB after coded for 4:2:2 
} xavs_level_t;

/* all of the levels defined in the standard, terminated by .level_idc=0 */
extern const xavs_level_t xavs_levels[];

/* xavs_param_default:
 *      fill xavs_param_t with default values and do CPU detection */
void xavs_param_default (xavs_param_t * param);


#define XAVS_PARAM_BAD_NAME  (-1)
#define XAVS_PARAM_BAD_VALUE (-2)
int xavs_param_parse (xavs_param_t *, const char *name, const char *value);

/****************************************************************************
 * Picture structures and functions.
 ****************************************************************************/
typedef struct
{
  int i_csp;

  int i_plane;
  int i_stride[4];
  uint8_t *plane[4];
} xavs_image_t;

typedef struct
{
  /* In: force picture type (if not auto)
   * Out: type of the picture encoded */
  int i_type;
  /* In: force quantizer for > 0 */
  int i_qpplus1;
  /* In: user pts, Out: pts of encoded picture (user) */
  int64_t i_pts;
  xavs_param_t *param;

  /* In: raw data */
  xavs_image_t img;
} xavs_picture_t;

/* xavs_picture_alloc:
 *  alloc data for a picture. You must call xavs_picture_clean on it. */
int xavs_picture_alloc (xavs_picture_t * pic, int i_csp, int i_width, int i_height);

/* xavs_picture_clean:
 *  free associated resource for a xavs_picture_t allocated with
 *  xavs_picture_alloc ONLY */
void xavs_picture_clean (xavs_picture_t * pic);

/****************************************************************************
 * NAL structure and functions:
 ****************************************************************************/
/* nal */
enum nal_unit_type_e
{
  NAL_UNKNOWN = 0,
  NAL_SLICE = 1,
  NAL_SLICE_DPA = 2,
  NAL_SLICE_DPB = 3,
  NAL_SLICE_DPC = 4,
  NAL_SLICE_IDR = 5,            /* ref_idc != 0 */
  NAL_SEI = 6,                  /* ref_idc == 0 */
  NAL_SPS = 7,
  NAL_PPS = 8,
  NAL_AUD = 9,
  /* ref_idc == 0 for 6,9,10,11,12 */
};
enum nal_priority_e
{
  NAL_PRIORITY_DISPOSABLE = 0,
  NAL_PRIORITY_LOW = 1,
  NAL_PRIORITY_HIGH = 2,
  NAL_PRIORITY_HIGHEST = 3,
};

typedef struct
{
  int i_ref_idc;                /* nal_priority_e */
  int i_type;                   /* nal_unit_type_e */

  /* This data are raw payload */
  int i_payload;
  uint8_t *p_payload;
} xavs_nal_t;

/* xavs_nal_encode:
 *      encode a nal into a buffer, setting the size.
 *      if b_annexeb then a long synch work is added
 *      XXX: it currently doesn't check for overflow */
int xavs_nal_encode (void *, int *, int b_annexeb, xavs_nal_t * nal);

/* xavs_nal_decode:
 *      decode a buffer nal into a xavs_nal_t */
int xavs_nal_decode (xavs_nal_t * nal, void *, int);

/****************************************************************************
 * Encoder functions:
 ****************************************************************************/

/* xavs_encoder_open:
 *      create a new encoder handler, all parameters from xavs_param_t are copied */
xavs_t *xavs_encoder_open (xavs_param_t *);
/* xavs_encoder_reconfig:
 *      change encoder options while encoding,
 *      analysis-related parameters from xavs_param_t are copied */
int xavs_encoder_reconfig (xavs_t *, xavs_param_t *);
/* xavs_encoder_headers:
 *      return the SPS and PPS that will be used for the whole stream */
int xavs_encoder_headers (xavs_t *, xavs_nal_t **, int *);
/* xavs_encoder_encode:
 *      encode one picture */
int xavs_encoder_encode (xavs_t *, xavs_nal_t **, int *, xavs_picture_t *, xavs_picture_t *);
/* xavs_encoder_close:
 *      close an encoder handler */
void xavs_encoder_close (xavs_t *);
/* xavs_encoder_delayed_frames:
 *      return the number of currently delayed (buffered) frames
 *      this should be used at the end of the stream, to know when you have all the encoded frames. */
int xavs_encoder_delayed_frames (xavs_t *);

#endif
