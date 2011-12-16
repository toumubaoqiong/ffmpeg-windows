#include "testffmpeglib.h"


#ifdef __cplusplus
extern "C"
{
#endif

#include <libavutil/common.h>
#include <libavutil/rational.h>
#include <libavutil/mathematics.h>
#include <libavutil/adler32.h>
#include <libavutil/aes.h>
#include <libavutil/lfg.h>
#include <libavutil/md5.h>
#include <libavutil/audioconvert.h>

#ifdef __cplusplus
}
#endif

#pragma comment(lib, "libavutil.lib")

static int ffmpegTest_libavutil_avrational(void);
static int ffmpegTest_libavutil_mathematics(void);
static int ffmpegTest_libavutil_adler32(void);
static int ffmpegTest_libavutil_aes(void);
static int ffmpegTest_libavutil_md5(void);
static int ffmpegTest_libavutil_audioconvert(void);

int ffmpegTest_libavutil(void)
{
	ffmpegTest_libavutil_audioconvert();
	ffmpegTest_libavutil_md5();
	ffmpegTest_libavutil_aes();
	ffmpegTest_libavutil_adler32();
	ffmpegTest_libavutil_avrational();
	ffmpegTest_libavutil_mathematics();
	return 0;
}

static int ffmpegTest_libavutil_audioconvert(void)
{
	int64_t tmpLayout = 
		av_get_channel_layout("5.1+downmix");
	printf("layout: %ld\n", tmpLayout);
	int audioChannel = 
		av_get_channel_layout_nb_channels(tmpLayout);
	char *tmpStr = "5.1+downmix";
	av_get_channel_layout_string(tmpStr, strlen(tmpStr), 5, 1);
	return 0;
}

static int ffmpegTest_libavutil_md5(void)
{
	//uint64_t md5val;
	//int i;
	//uint8_t in[1000];

	//for(i = 0; i < 1000; i++) 
	//{
	//	in[i] = i * i;
	//}
	//av_md5_sum( (uint8_t *)&md5val, in,  1000);
	//printf("%"PRId64"\n", md5val);
	//av_md5_sum( (uint8_t *)&md5val, in,  63);
	//printf("%"PRId64"\n", md5val);
	//av_md5_sum( (uint8_t *)&md5val, in,  64);
	//printf("%"PRId64"\n", md5val);
	//av_md5_sum( (uint8_t *)&md5val, in,  65);
	//printf("%"PRId64"\n", md5val);
	//for(i = 0; i < 1000; i++) 
	//{
	//	in[i] = i % 127;
	//}
	//av_md5_sum( (uint8_t *)&md5val, in,  999);
	//printf("%"PRId64"\n", md5val);

	return 0;
}

static int ffmpegTest_libavutil_aes(void)
{
	int i, j;
	AVAES ae, ad, b;
	uint8_t rkey[2][16] =
	{
		{0},
		{0x10, 0xa5, 0x88, 0x69, 0xd7, 0x4b, 0xe5, 0xa3, 0x74, 0xcf, 0x86, 0x7c, 0xfb, 0x47, 0x38, 0x59}
	};
	uint8_t pt[16], rpt[2][16] =
	{
		{0x6a, 0x84, 0x86, 0x7c, 0xd7, 0x7e, 0x12, 0xad, 0x07, 0xea, 0x1b, 0xe8, 0x95, 0xc5, 0x3f, 0xa3},
		{0}
	};
	uint8_t rct[2][16] =
	{
		{0x73, 0x22, 0x81, 0xc0, 0xa0, 0xaa, 0xb8, 0xf7, 0xa5, 0x4a, 0x0c, 0x67, 0xa0, 0xc4, 0x5e, 0xcf},
		{0x6d, 0x25, 0x1e, 0x69, 0x44, 0xb0, 0x51, 0xe0, 0x4e, 0xaa, 0x6f, 0xb4, 0xdb, 0xf7, 0x84, 0x65}
	};
	uint8_t temp[16];
	AVLFG prng;

	av_aes_init(&ae, "PI=3.141592654..", 128, 0);
	av_aes_init(&ad, "PI=3.141592654..", 128, 1);
	av_log_set_level(AV_LOG_DEBUG);
	av_lfg_init(&prng, 1);

	for(i = 0; i < 2; i++)
	{
		av_aes_init(&b, rkey[i], 128, 1);
		av_aes_crypt(&b, temp, rct[i], 1, NULL, 1);
		for(j = 0; j < 16; j++)
			if(rpt[i][j] != temp[j])
				av_log(NULL, AV_LOG_ERROR, "%d %02X %02X\n", j, rpt[i][j], temp[j]);
	}

	for(i = 0; i < 10000; i++)
	{
		for(j = 0; j < 16; j++)
		{
			pt[j] = av_lfg_get(&prng);
		}
		{
			START_TIMER
				av_aes_crypt(&ae, temp, pt, 1, NULL, 0);
			if(!(i&(i - 1)))
				av_log(NULL, AV_LOG_ERROR, "%02X %02X %02X %02X\n", temp[0], temp[5], temp[10], temp[15]);
			av_aes_crypt(&ad, temp, temp, 1, NULL, 1);
			STOP_TIMER("aes")
		}
		for(j = 0; j < 16; j++)
		{
			if(pt[j] != temp[j])
			{
				av_log(NULL, AV_LOG_ERROR, "%d %d %02X %02X\n", i, j, pt[j], temp[j]);
			}
		}
	}
	return 0;
}

static int ffmpegTest_libavutil_adler32(void)
{
	volatile int checksum;
#define TEST_av_adler32_update_LEN 7001
	int i;
	char data[TEST_av_adler32_update_LEN];
	av_log_set_level(AV_LOG_DEBUG);
	for(i = 0; i < TEST_av_adler32_update_LEN; i++)
	{
		data[i] = ((i * i) >> 3) + 123 * i;
	}
	for(i = 0; i < 1000; i++)
	{
		checksum = av_adler32_update(1, 
				(const uint8_t *)data, TEST_av_adler32_update_LEN);
	}
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