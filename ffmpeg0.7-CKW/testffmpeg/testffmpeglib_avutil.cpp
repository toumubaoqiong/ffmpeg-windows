#include "testffmpeglib.h"


#ifdef __cplusplus
extern "C"
{
#endif

#include <libavutil/rational.h>

#ifdef __cplusplus
}
#endif

#pragma comment(lib, "libavutil.lib")


int ffmpegTest_libavutil(void)
{
	AVRational tmp1, tmp2;
	av_mul_q(tmp1, tmp2);
	return 0;
}