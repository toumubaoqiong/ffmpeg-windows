#ifndef MEDIANETSERVER_INCLUDED
#define MEDIANETSERVER_INCLUDED
#include <BasicUsageEnvironment.hh>
#include "TestdynamicRTSPServer1.h"

class mediaNetServer
{
public:
	mediaNetServer(const int rtspServerPort);
	~mediaNetServer();
public:
	int start(void);
	int end(void);
	int addMediaSession(void *const sessionParams);
private:
	TaskScheduler *scheduler;
    UsageEnvironment *env;	
	UserAuthenticationDatabase *authDB;
    RTSPServer *rtspServer;
	portNumBits rtspServerPortNum;
};

extern int mediaNetServer_Init(const int rtspServerPort);
extern int mediaNetServer_Start(void);
extern int mediaNetServer_addMediaSession(void *const sessionParams);

#endif