// testffmpeg.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "testffmpeglib.h"

static int runffmpegTestFunction(void);

int _tmain(int argc, _TCHAR* argv[])
{
	runffmpegTestFunction();
	return 0;
}


static int runffmpegTestFunction(void)
{
	ffmpegTest_libavutil();
	return 0;
}