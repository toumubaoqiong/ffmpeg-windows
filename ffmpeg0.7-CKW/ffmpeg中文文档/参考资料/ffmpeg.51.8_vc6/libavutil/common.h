/**
 * @file common.h
 * common internal api header.
 */

#ifndef COMMON_H
#define COMMON_H

#ifndef _SIZE_T_DEFINED
#define size_t unsigned int
#define _SIZE_T_DEFINED
#endif

#ifdef WIN32
#define inline __inline
#endif

#ifndef NULL
#define NULL (void*)0
#endif

#if defined(WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#    define CONFIG_WIN32
#endif

#if defined(WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__) && !defined(EMULATE_INTTYPES)
#    define EMULATE_INTTYPES
#endif

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#include "../config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#ifndef __BEOS__
#   include <errno.h>
#else
#   include "berrno.h"
#endif
#include <math.h>

#include <stddef.h>
#ifndef offsetof
# define offsetof(T,F) ((unsigned int)((char *)&((T *)0)->F))
#endif

#define AVOPTION_CODEC_BOOL(name, help, field) \
    { name, help, offsetof(AVCodecContext, field), FF_OPT_TYPE_BOOL }
#define AVOPTION_CODEC_DOUBLE(name, help, field, minv, maxv, defval) \
    { name, help, offsetof(AVCodecContext, field), FF_OPT_TYPE_DOUBLE, minv, maxv, defval }
#define AVOPTION_CODEC_FLAG(name, help, field, flag, defval) \
    { name, help, offsetof(AVCodecContext, field), FF_OPT_TYPE_FLAG, flag, 0, defval }
#define AVOPTION_CODEC_INT(name, help, field, minv, maxv, defval) \
    { name, help, offsetof(AVCodecContext, field), FF_OPT_TYPE_INT, minv, maxv, defval }
#define AVOPTION_CODEC_STRING(name, help, field, str, val) \
    { name, help, offsetof(AVCodecContext, field), FF_OPT_TYPE_STRING, .defval = val, .defstr = str }
#define AVOPTION_CODEC_RCOVERRIDE(name, help, field) \
    { name, help, offsetof(AVCodecContext, field), FF_OPT_TYPE_RCOVERRIDE, .defval = 0, .defstr = NULL }
#define AVOPTION_SUB(ptr) { .name = NULL, .help = (const char*)ptr }
#define AVOPTION_END() AVOPTION_SUB(NULL)

typedef signed char  int8_t;
typedef signed short int16_t;
typedef signed int   int32_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;

#ifdef CONFIG_WIN32
	typedef signed __int64   int64_t;
	typedef unsigned __int64 uint64_t;
#else /* other OS */
	typedef signed long long   int64_t;
	typedef unsigned long long uint64_t;
#endif /* other OS */

#ifndef PRId64
#define PRId64 "lld"
#endif

#ifndef PRIu64
#define PRIu64 "llu"
#endif

#ifndef PRIx64
#define PRIx64 "llx"
#endif

#ifndef PRId32
#define PRId32 "d"
#endif

#ifndef PRIdFAST16
#define PRIdFAST16 PRId32
#endif

#ifndef PRIdFAST32
#define PRIdFAST32 PRId32
#endif

#ifndef INT16_MIN
#define INT16_MIN       (-0x7fff-1)
#endif

#ifndef INT16_MAX
#define INT16_MAX       0x7fff
#endif

#ifndef INT32_MIN
#define INT32_MIN       (-0x7fffffff-1)
#endif

#ifndef INT32_MAX
#define INT32_MAX       0x7fffffff
#endif

#ifndef UINT32_MAX
#define UINT32_MAX      0xffffffff
#endif

#ifndef INT64_MIN
#define INT64_MIN       (-1-(int64_t)0x7fffffffffffffff)
#endif

#ifndef INT64_MAX
#define INT64_MAX int64_t_C(9223372036854775807)
#endif

#ifndef UINT64_MAX
#define UINT64_MAX uint64_t_C(0xFFFFFFFFFFFFFFFF)
#endif

typedef signed char int_fast8_t;
typedef signed int  int_fast16_t;
typedef signed int  int_fast32_t;
typedef unsigned char uint_fast8_t;
typedef unsigned int  uint_fast16_t;
typedef unsigned int  uint_fast32_t;
typedef uint64_t      uint_fast64_t;

#ifndef INT_BIT
#    if INT_MAX != 2147483647
#        define INT_BIT 64
#    else
#        define INT_BIT 32
#    endif
#endif

#if defined(CONFIG_OS2) || defined(CONFIG_SUNOS)
static inline float floorf(float f) 
{
    return floor(f);
}
#endif

#ifdef CONFIG_WIN32 /* windows */

#    if !defined(__MINGW32__) && !defined(__CYGWIN__)
#        define int64_t_C(c)     (c ## i64)
#        define uint64_t_C(c)    (c ## i64)

#    ifdef HAVE_AV_CONFIG_H
#            define inline __inline
#    endif

#    else
#        define int64_t_C(c)     (c ## LL)
#        define uint64_t_C(c)    (c ## ULL)
#    endif /* __MINGW32__ */

#    define snprintf _snprintf
#    define vsnprintf _vsnprintf

#else /* unix */

#ifndef int64_t_C
#define int64_t_C(c)     (c ## LL)
#define uint64_t_C(c)    (c ## ULL)
#endif

#ifdef USE_FASTMEMCPY
#  include "fastmemcpy.h"
# endif

#endif /* !CONFIG_WIN32 && !CONFIG_OS2 */


#    include "bswap.h"

// Use rip-relative addressing if compiling PIC code on x86-64.
#    if defined(__MINGW32__) || defined(__CYGWIN__) || \
        defined(__OS2__) || (defined (__OpenBSD__) && !defined(__ELF__))
#        if defined(ARCH_X86_64) && defined(PIC)
#            define MANGLE(a) "_" #a"(%%rip)"
#        else
#            define MANGLE(a) "_" #a
#        endif
#    else
#        if defined(ARCH_X86_64) && defined(PIC)
#            define MANGLE(a) #a"(%%rip)"
#        elif defined(CONFIG_DARWIN)
#            define MANGLE(a) "_" #a
#        else
#            define MANGLE(a) #a
#        endif
#    endif

/* debug stuff */

#    include <assert.h>

/* dprintf macros */
#    if defined(CONFIG_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)

inline void dprintf(const char* fmt,...) {}

#    else

#        ifdef DEBUG
#            define dprintf(fmt,...) av_log(NULL, AV_LOG_DEBUG, fmt, __VA_ARGS__)
#        else
#            define dprintf(fmt,...)
#        endif

#    endif /* !CONFIG_WIN32 */

#define av_abort()      do { abort(); } while (0)

#define RSHIFT(a,b) ((a) > 0 ? ((a) + ((1<<(b))>>1))>>(b) : ((a) + ((1<<(b))>>1)-1)>>(b))
#define ROUNDED_DIV(a,b) (((a)>0 ? (a) + ((b)>>1) : (a) - ((b)>>1))/(b))
#define ABS(a) ((a) >= 0 ? (a) : (-(a)))

#define FFMIN(a,b) ((a) > (b) ? (b) : (a))
#define FFMAX(a,b) ((a) > (b) ? (a) : (b))

#define FASTDIV(a,b)   ((a)/(b))

extern const uint8_t ff_log2_tab[256];

static inline int av_log2(unsigned int v)
{
    int n;

    n = 0;
    if (v & 0xffff0000) {
        v >>= 16;
        n += 16;
    }
    if (v & 0xff00) {
        v >>= 8;
        n += 8;
    }
    n += ff_log2_tab[v];

    return n;
}

static inline int av_log2_16bit(unsigned int v)
{
    int n;

    n = 0;
    if (v & 0xff00) {
        v >>= 8;
        n += 8;
    }
    n += ff_log2_tab[v];

    return n;
}

static inline int mid_pred(int a, int b, int c)
{
    if(a>b){
        if(c>b){
            if(c>a) b=a;
            else    b=c;
        }
    }else{
        if(b>c){
            if(c>a) b=c;
            else    b=a;
        }
    }
    return b;
}

static inline int clip(int a, int amin, int amax)
{
    if (a < amin)
        return amin;
    else if (a > amax)
        return amax;
    else
        return a;
}

static inline int clip_uint8(int a)
{
    if (a&(~255)) return (-a)>>31;
    else          return a;
}

extern const uint8_t ff_sqrt_tab[128];

int64_t ff_gcd(int64_t a, int64_t b);

static inline int ff_sqrt(int a)
{
    int ret=0;
    int s;
    int ret_sq=0;

    if(a<128) return ff_sqrt_tab[a];

    for(s=15; s>=0; s--){
        int b= ret_sq + (1<<(s*2)) + (ret<<s)*2;
        if(b<=a){
            ret_sq=b;
            ret+= 1<<s;
        }
    }
    return ret;
}

static inline int ff_get_fourcc(const char *s)
{
    assert( strlen(s)==4 );

    return (s[0]) + (s[1]<<8) + (s[2]<<16) + (s[3]<<24);
}

#define MKTAG(a,b,c,d) (a | (b << 8) | (c << 16) | (d << 24))
#define MKBETAG(a,b,c,d) (d | (c << 8) | (b << 16) | (a << 24))

#define MASK_ABS(mask, level)\
            mask= level>>31;\
            level= (level^mask)-mask;

#define COPY3_IF_LT(x,y,a,b,c,d)\
if((y)<(x)){\
     (x)=(y);\
     (a)=(b);\
     (c)=(d);\
}

#define CHECKED_ALLOCZ(p, size)\
{\
    p= av_mallocz(size);\
    if(p==NULL && (size)!=0){\
        goto fail;\
    }\
};

static inline long int lrintf(float x)
{
    return (int)(x + (x < 0 ? -0.5 : 0.5));
}

#endif /* COMMON_H */
