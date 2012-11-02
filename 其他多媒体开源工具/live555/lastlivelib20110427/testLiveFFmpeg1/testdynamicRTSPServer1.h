#ifndef TESTTestDynamicRTSPServer1_INCLUDED
#define TESTTestDynamicRTSPServer1_INCLUDED

#include "RTSPServer.hh"


class TestDynamicRTSPServer: public RTSPServer 
{
public:
	static TestDynamicRTSPServer* createNew(
		UsageEnvironment& env, 
		Port ourPort,
		UserAuthenticationDatabase* authDatabase,
		unsigned reclamationTestSeconds = 65);
	
private:
	TestDynamicRTSPServer(
		UsageEnvironment& env, 
		int ourSocket, 
		Port ourPort,
		UserAuthenticationDatabase* authDatabase, 
		unsigned reclamationTestSeconds);
	// called only by createNew();
	virtual ~TestDynamicRTSPServer();
	
private: // redefined virtual functions
	virtual ServerMediaSession* lookupServerMediaSession(char const* streamName);
};



#endif