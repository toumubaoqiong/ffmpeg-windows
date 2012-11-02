#include <stdio.h>
#include "mediaNetServer.h"

//#pragma comment(lib, "avcodec.lib")
//#pragma comment(lib, "avformat.lib")
//#pragma comment(lib, "avutil.lib")
//#pragma comment(lib, "swscale.lib")

int main(void)
{
	if(mediaNetServer_Init(555) < 0)
	{
		printf("mediaNetServer_Init(555) fail \n");
		return -1;
	}
	else
	{
		printf("mediaNetServer_Init(555) sucess \n");
	}
	mediaNetServer_Start();
	return 0;
}
