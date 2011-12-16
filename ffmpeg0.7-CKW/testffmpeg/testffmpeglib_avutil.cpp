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
#include <libavutil/avstring.h>
#include <libavutil/avutil.h>
#include <libavutil/base64.h>
#include <libavutil/crc.h>
#include <libavutil/des.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/error.h>
#include <libavutil/eval.h>
#include <libavutil/file.h>

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
static int ffmpegTest_libavutil_avstring(void);
static int ffmpegTest_libavutil_avutil(void);
static int ffmpegTest_libavutil_base64(void);
static int ffmpegTest_libavutil_crc(void);
static int ffmpegTest_libavutil_des(void);
static int ffmpegTest_libavutil_error(void);
static int ffmpegTest_libavutil_eval(void);
static int ffmpegTest_libavutil_file(void);

int ffmpegTest_libavutil(void)
{
	ffmpegTest_libavutil_file();
	ffmpegTest_libavutil_eval();
	ffmpegTest_libavutil_error();
	ffmpegTest_libavutil_des();
	ffmpegTest_libavutil_crc();
	ffmpegTest_libavutil_base64();
	ffmpegTest_libavutil_avutil();
	ffmpegTest_libavutil_avstring();
	ffmpegTest_libavutil_audioconvert();
	ffmpegTest_libavutil_md5();
	ffmpegTest_libavutil_aes();
	ffmpegTest_libavutil_adler32();
	ffmpegTest_libavutil_avrational();
	ffmpegTest_libavutil_mathematics();
	return 0;
}


static int ffmpegTest_libavutil_file(void)
{
	uint8_t *buf;
	size_t size;
	if (av_file_map("d:\\video\\test.avi", &buf, &size, 0, NULL) < 0)
	{
		return 1;
	}
	//buf[0] = 's';
	printf("%s", buf);
	av_file_unmap(buf, size);
	return 0;
}


static double const_values[] =
{
	M_PI,
	M_E,
	0
};

static const char *const_names[] =
{
	"PI",
	"E",
	0
};

static int ffmpegTest_libavutil_eval(void)
{
	int i;
	double d;
	const char **expr, *exprs[] =
	{
		"",
		"1;2",
		"-20",
		"-PI",
		"+PI",
		"1+(5-2)^(3-1)+1/2+sin(PI)-max(-2.2,-3.1)",
		"80G/80Gi"
		"1k",
		"1Gi",
		"1gi",
		"1GiFoo",
		"1k+1k",
		"1Gi*3foo",
		"foo",
		"foo(",
		"foo()",
		"foo)",
		"sin",
		"sin(",
		"sin()",
		"sin)",
		"sin 10",
		"sin(1,2,3)",
		"sin(1 )",
		"1",
		"1foo",
		"bar + PI + E + 100f*2 + foo",
		"13k + 12f - foo(1, 2)",
		"1gi",
		"1Gi",
		"st(0, 123)",
		"st(1, 123); ld(1)",
		/* compute 1+2+...+N */
		"st(0, 1); while(lte(ld(0), 100), st(1, ld(1)+ld(0));st(0, ld(0)+1)); ld(1)",
		/* compute Fib(N) */
		"st(1, 1); st(2, 2); st(0, 1); while(lte(ld(0),10), st(3, ld(1)+ld(2)); st(1, ld(2)); st(2, ld(3)); st(0, ld(0)+1)); ld(3)",
		"while(0, 10)",
		"st(0, 1); while(lte(ld(0),100), st(1, ld(1)+ld(0)); st(0, ld(0)+1))",
		"isnan(1)",
		"isnan(NAN)",
		"floor(NAN)",
		"floor(123.123)",
		"floor(-123.123)",
		"trunc(123.123)",
		"trunc(-123.123)",
		"ceil(123.123)",
		"ceil(-123.123)",
		NULL
	};

	for (expr = exprs; *expr; expr++)
	{
		printf("Evaluating '%s'\n", *expr);
		av_expr_parse_and_eval(&d, *expr,
			const_names, const_values,
			NULL, NULL, NULL, NULL, NULL, 0, NULL);
		printf("'%s' -> %f\n\n", *expr, d);
	}

	av_expr_parse_and_eval(&d, "1+(5-2)^(3-1)+1/2+sin(PI)-max(-2.2,-3.1)",
		const_names, const_values,
		NULL, NULL, NULL, NULL, NULL, 0, NULL);
	printf("%f == 12.7\n", d);
	av_expr_parse_and_eval(&d, "80G/80Gi",
		const_names, const_values,
		NULL, NULL, NULL, NULL, NULL, 0, NULL);
	printf("%f == 0.931322575\n", d);

	for (i = 0; i < 1050; i++)
	{
		START_TIMER
			av_expr_parse_and_eval(&d, "1+(5-2)^(3-1)+1/2+sin(PI)-max(-2.2,-3.1)",
			const_names, const_values,
			NULL, NULL, NULL, NULL, NULL, 0, NULL);
		STOP_TIMER("av_expr_parse_and_eval")
	}
	return 0;
}


#include <winsock2.h>
#include <stdlib.h>
#include <stdio.h>


static int ffmpegTest_libavutil_error(void)
{
	char printStr[256] = {0};
	av_strerror(AVERROR_EOF, printStr, 250);
	printf("errorStr %s\n", printStr);
	av_strerror(AVERROR_IO, printStr, 250);
	printf("errorStr %s\n", printStr);
}

static uint64_t ffmpegTest_libavutil_des_rand64(void)
{
	uint64_t r = rand();
	r = (r << 32) | rand();
	return r;
}

static const uint8_t test_key[] = {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0};
static const DECLARE_ALIGNED(8, uint8_t, plain)[] = {0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10};
static const DECLARE_ALIGNED(8, uint8_t, crypt)[] = {0x4a, 0xb6, 0x5b, 0x3d, 0x4b, 0x06, 0x15, 0x18};
static DECLARE_ALIGNED(8, uint8_t, tmp)[8];
static DECLARE_ALIGNED(8, uint8_t, large_buffer)[10002][8];
static const uint8_t cbc_key[] =
{
	0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
	0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01,
	0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23
};

static int ffmpegTest_libavutil_des_run_test(int cbc, int decrypt)
{
	AVDES d;
	int delay = cbc && !decrypt ? 2 : 1;
	uint64_t res;
	AV_WB64(large_buffer[0], 0x4e6f772069732074ULL);
	AV_WB64(large_buffer[1], 0x1234567890abcdefULL);
	AV_WB64(tmp,             0x1234567890abcdefULL);
	av_des_init(&d, cbc_key, 192, decrypt);
	av_des_crypt(&d, large_buffer[delay], large_buffer[0], 10000, cbc ? tmp : NULL, decrypt);
	res = AV_RB64(large_buffer[9999 + delay]);
	if (cbc)
	{
		if (decrypt)
			return res == 0xc5cecf63ecec514cULL;
		else
			return res == 0xcb191f85d1ed8439ULL;
	}
	else
	{
		if (decrypt)
			return res == 0x8325397644091a0aULL;
		else
			return res == 0xdd17e8b8b437d232ULL;
	}
}

static int ffmpegTest_libavutil_des(void)
{
	AVDES d;
	int i;
#ifdef GENTABLES
	int j;
#endif
	struct timeval tv;
	uint64_t key[3];
	uint64_t data;
	uint64_t ct;
	uint64_t roundkeys[16];
	//gettimeofday(&tv, NULL);
	srand(tv.tv_sec * 1000 * 1000 + tv.tv_usec);
	key[0] = AV_RB64(test_key);
	data = AV_RB64(plain);
	//gen_roundkeys(roundkeys, key[0]);
	//if (des_encdec(data, roundkeys, 0) != AV_RB64(crypt))
	//{
	//	printf("Test 1 failed\n");
	//	return 1;
	//}
	av_des_init(&d, test_key, 64, 0);
	av_des_crypt(&d, tmp, plain, 1, NULL, 0);
	if (memcmp(tmp, crypt, sizeof(crypt)))
	{
		printf("Public API decryption failed\n");
		return 1;
	}
	if (!ffmpegTest_libavutil_des_run_test(0, 0) 
		|| !ffmpegTest_libavutil_des_run_test(0, 1) 
		|| !ffmpegTest_libavutil_des_run_test(1, 0) 
		|| !ffmpegTest_libavutil_des_run_test(1, 1))
	{
		printf("Partial Monte-Carlo test failed\n");
		return 1;
	}
	for (i = 0; i < 100; i++)
	{
		key[0] = ffmpegTest_libavutil_des_rand64();
		key[1] = ffmpegTest_libavutil_des_rand64();
		key[2] = ffmpegTest_libavutil_des_rand64();
		data = ffmpegTest_libavutil_des_rand64();
		av_des_init(&d, (const uint8_t *)key, 192, 0);
		av_des_crypt(&d, (uint8_t *)&ct, (const uint8_t *)&data, 1, NULL, 0);
		av_des_init(&d, (const uint8_t *)key, 192, 1);
		av_des_crypt(&d, (uint8_t *)&ct, (const uint8_t *)&ct, 1, NULL, 1);
		if (ct != data)
		{
			printf("Test 2 failed\n");
			return 1;
		}
	}
#ifdef GENTABLES
	printf("static const uint32_t S_boxes_P_shuffle[8][64] = {\n");
	for (i = 0; i < 8; i++)
	{
		printf("    {");
		for (j = 0; j < 64; j++)
		{
			uint32_t v = S_boxes[i][j >> 1];
			v = j & 1 ? v >> 4 : v & 0xf;
			v <<= 28 - 4 * i;
			v = shuffle(v, P_shuffle, sizeof(P_shuffle));
			printf((j & 7) == 0 ? "\n    " : " ");
			printf("0x%08X,", v);
		}
		printf("\n    },\n");
	}
	printf("};\n");
#endif
	return 0;
}

static int ffmpegTest_libavutil_crc(void)
{
	uint8_t buf[1999];
	int i;
	int p[4][3] = {{AV_CRC_32_IEEE_LE, 0xEDB88320, 0x3D5CDD04},
	{AV_CRC_32_IEEE   , 0x04C11DB7, 0xC0F5BAE0},
	{AV_CRC_16_ANSI   , 0x8005,     0x1FBB    },
	{AV_CRC_8_ATM     , 0x07,       0xE3      },
	};
	const AVCRC *ctx;

	for(i = 0; i < sizeof(buf); i++)
		buf[i] = i + i * i;

	for(i = 0; i < 4; i++)
	{
		ctx = av_crc_get_table((AVCRCId)p[i][0]);
		printf("crc %08X =%X\n", p[i][1], 
			av_crc(ctx, 0, buf, sizeof(buf)));
	}
	return 0;
}


static int ffmpegTest_libavutil_base64_test_encode_decode(
			const uint8_t *data, 
			unsigned int data_size,
			const char *encoded_ref)
{
#define MAX_DATA_SIZE    1024
#define MAX_ENCODED_SIZE 2048
	char  encoded[MAX_ENCODED_SIZE];
	uint8_t data2[MAX_DATA_SIZE];
	int data2_size, max_data2_size = MAX_DATA_SIZE;

	if (!av_base64_encode(encoded, MAX_ENCODED_SIZE, data, data_size))
	{
		printf("Failed: cannot encode the input data\n");
		return 1;
	}
	if (encoded_ref && strcmp(encoded, encoded_ref))
	{
		printf("Failed: encoded string differs from reference\n"
			"Encoded:\n%s\nReference:\n%s\n", encoded, encoded_ref);
		return 1;
	}

	if ((data2_size = av_base64_decode(data2, encoded, max_data2_size)) < 0)
	{
		printf("Failed: cannot decode the encoded string\n"
			"Encoded:\n%s\n", encoded);
		return 1;
	}
	if (memcmp(data2, data, data_size))
	{
		printf("Failed: encoded/decoded data differs from original data\n");
		return 1;
	}

	printf("Passed!\n");
	return 0;
}

static int ffmpegTest_libavutil_base64(void)
{
	int i, error_count = 0;
	struct test
	{
		const uint8_t *data;
		const char *encoded_ref;
	} tests[] =
	{
		{ "",        ""},
		{ "1",       "MQ=="},
		{ "22",      "MjI="},
		{ "333",     "MzMz"},
		{ "4444",    "NDQ0NA=="},
		{ "55555",   "NTU1NTU="},
		{ "666666",  "NjY2NjY2"},
		{ "abc:def", "YWJjOmRlZg=="},
	};

	printf("Encoding/decoding tests\n");
	for (i = 0; i < FF_ARRAY_ELEMS(tests); i++)
		error_count += 
		ffmpegTest_libavutil_base64_test_encode_decode(
			tests[i].data, 
			strlen((const char *)tests[i].data), 
			tests[i].encoded_ref);

	return error_count;
}


static int ffmpegTest_libavutil_avutil(void)
{
#define AV_STRINGIFY111_1111(s)		AV_TOSTRING111_1111(s)
#define AV_TOSTRING111_1111(s)		#s
	printf("wangjing printf %s\n", 
		AV_STRINGIFY111_1111(11231432143124323214));
	printf("version: %f, config: %s, licese: %d",
		avutil_version(), 
		avutil_configuration(),
		avutil_license());
	return 0;
}

static int ffmpegTest_libavutil_avstring(void)
{
	char *tmpStr = NULL;
	int tmpbl = av_strstart(
		"wangjingisgood", "wangjing", (const char**)&tmpStr);
	tmpbl = av_stristart(
		"wangjingisgood", "WANGJING", (const char**)&tmpStr);
	tmpStr = av_stristr("wangjingisgood", "IS");
	tmpStr = av_stristr("wangjingisgood", "IS");
	char tmpStrsA[256] = {0};
	char tmpStrsB[256] = {0};
	char tmpStrsC[256] = {0};
	tmpbl = av_strlcpy(tmpStrsA, "wangjingisgood", 200);
	tmpbl = av_strlcat(tmpStrsA, "wangjingisgood", 200);
	tmpbl = av_strlcatf(tmpStrsB, 250, "wangjing %s is %d\n", 
		"wangjingisgood", 1111111);
	char *tmpStrDouble = av_d2str(2.33339913314564566445654648841156);
	char *tmpStrStrA = "wangjing is good is ok is very niceis ok is good";
	av_get_token(&tmpStrStrA, "is");
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