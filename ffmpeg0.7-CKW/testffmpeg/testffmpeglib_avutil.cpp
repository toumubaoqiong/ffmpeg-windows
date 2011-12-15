#include "testffmpeglib.h"


#ifdef __cplusplus
extern "C"
{
#endif

#include <libavutil/common.h>
#include <libavutil/rational.h>
#include <libavutil/mathematics.h>

#ifdef __cplusplus
}
#endif

#pragma comment(lib, "libavutil.lib")

static int ffmpegTest_libavutil_avrational(void);
static int ffmpegTest_libavutil_mathematics(void);

int ffmpegTest_libavutil(void)
{
	ffmpegTest_libavutil_avrational();
	ffmpegTest_libavutil_mathematics();
	return 0;
}

static int ffmpegTest_libavutil_avrational(void)
{
	int tmpDst_num = 0,
		tmpDst_den = 0,
		tmpNum = 0,
		tmpDen = 0,
		tmpMax = 0;
	tmpDst_num = 0;
	tmpDst_den = 0;
	tmpNum = 1;
	tmpDen = 30;
	tmpMax = 29;
	AVRational tmpRationalA = {1, 15};
	AVRational tmpRationalB = {1, 30};
	AVRational tmpRationalC = {0, 0};
	tmpRationalC = av_mul_q(tmpRationalA, tmpRationalB);
	tmpRationalC = av_div_q(tmpRationalA, tmpRationalB);
	tmpRationalC = av_add_q(tmpRationalA, tmpRationalB);
	tmpRationalC = av_sub_q(tmpRationalA, tmpRationalB);
	tmpRationalC = av_d2q(0.05121, 500);
	av_reduce(&tmpDst_num, &tmpDst_den, tmpNum, tmpDen, tmpMax);
	AVRational tmpRationalStard = {1, 25};
	int tmpBl = av_nearer_q(tmpRationalStard, tmpRationalA, tmpRationalB);
	AVRational tmpRationalArray[] = 
	{
		{1, 15},
		{1, 16},
		{1, 17},
		{1, 20},
		{1, 23},
		{1, 25},
		{1, 0},
	};
	av_find_nearest_q_idx(tmpRationalB, tmpRationalArray);
	av_rational_test_local();
	return 0;
}

static int ffmpegTest_libavutil_mathematics(void)
{
	int64_t tmpNumA = 198,
		tmpNumB = 320,
		tmpNumC = 113,
		tmpNumD = 120;
	tmpNumA = av_gcd(198, 320);
	int tmpNumInt1 = av_log2_c(198);
	int tmpNumInt2 = av_log2_16bit_c(320);
	tmpNumA = 2;
	tmpNumB = 3;
	tmpNumC = 4;
	tmpNumD = 0;
	tmpNumD = av_rescale_rnd(tmpNumA, 
		tmpNumB, tmpNumC, AV_ROUND_ZERO);
	tmpNumD = av_rescale_rnd(tmpNumA, 
		tmpNumB, tmpNumC, AV_ROUND_INF);
	tmpNumD = av_rescale_rnd(tmpNumA, 
		tmpNumB, tmpNumC, AV_ROUND_DOWN);
	tmpNumD = av_rescale_rnd(tmpNumA, 
		tmpNumB, tmpNumC, AV_ROUND_UP);
	tmpNumD = av_rescale_rnd(tmpNumA, 
		tmpNumB, tmpNumC, AV_ROUND_NEAR_INF);
	tmpNumA = 0xFFFFFFFFFF;
	tmpNumB = 0xFFFFFFFFFE;
	tmpNumC = 0xFFFFFFFFFD;
	tmpNumD = av_rescale_rnd(tmpNumA, 
		tmpNumB, tmpNumC, AV_ROUND_NEAR_INF);
	tmpNumA = 320;
	AVRational tmpRation1 = {1, 25};
	AVRational tmpRation2 = {1, 50};
	tmpNumA = av_rescale_q(tmpNumA, tmpRation1, tmpRation2);
	tmpNumA = 17;
	tmpNumB = 28;
	int tmpBL = 
		av_compare_ts(tmpNumA, tmpRation1, tmpNumB, tmpRation2);
	tmpNumD = av_compare_mod(tmpNumA, tmpNumB, 64);
	unsigned int tmpUInt = ff_sqrt(25);
	tmpUInt = ff_sqrt(320);
	int a, b, c;
	a = 2;
	b = 2;
	c = RSHIFT(a, b);
	a = -2;
	b = 2;
	c = RSHIFT(a, b);
	a = b = 2;
	c = ROUNDED_DIV(a,b);
	a = -2;
	b = 2;
	c = ROUNDED_DIV(a,b);
	a = 2;
	b = 2;
	c = FFALIGN(a, b);
	a = 10;
	b = 15;
	c = FFALIGN(a, b);
	return 0;
}