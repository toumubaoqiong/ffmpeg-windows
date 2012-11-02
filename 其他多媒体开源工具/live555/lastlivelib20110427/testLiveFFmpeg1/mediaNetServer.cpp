#include "mediaNetServer.h"
#define MEDIA_SERVER_VERSION_STRING "0.67"


mediaNetServer::mediaNetServer(const int rtspServerPort)
{
	// Begin by setting up our usage environment:
    scheduler = BasicTaskScheduler::createNew();
    env = BasicUsageEnvironment::createNew(*scheduler);
	
	authDB = NULL;
#ifdef ACCESS_CONTROL
    // To implement client access control to the RTSP server, do the following:
    authDB = new UserAuthenticationDatabase;
    authDB->addUserRecord("username1", "password1"); // replace these with real strings
    // Repeat the above with each <username>, <password> that you wish to allow
    // access to the server.
#endif
	// Create the RTSP server.  Try first with the default port number (554),
    // and then with the alternative port number (8554):
    rtspServerPortNum = rtspServerPort;
	//portNumBits rtspServerPortNum = 554;
    rtspServer = TestDynamicRTSPServer::createNew(*env, rtspServerPortNum, authDB);
    if (rtspServer == NULL)
    {
        rtspServerPortNum = 8554;
        rtspServer = TestDynamicRTSPServer::createNew(*env, rtspServerPortNum, authDB);
    }
    if (rtspServer == NULL)
    {
        *env << "Failed to create RTSP server: " << env->getResultMsg() << "\n";
		exit(1);
    }
	*env << "LIVE555 Media Server\n";
    *env << "\tversion " << MEDIA_SERVER_VERSION_STRING
		<< " (LIVE555 Streaming Media library version "
		<< LIVEMEDIA_LIBRARY_VERSION_STRING << ").\n";
	
    char *urlPrefix = rtspServer->rtspURLPrefix();
    *env << "Play streams from this server using the URL\n\t"
		<< urlPrefix << "<filename>\nwhere <filename> is a file present in the current directory.\n";
    *env << "Each file's type is inferred from its name suffix:\n";
    *env << "\t\".264\" => a H.264 Video Elementary Stream file\n";
    *env << "\t\".aac\" => an AAC Audio (ADTS format) file\n";
    *env << "\t\".ac3\" => an AC-3 Audio file\n";
    *env << "\t\".amr\" => an AMR Audio file\n";
    *env << "\t\".dv\" => a DV Video file\n";
    *env << "\t\".m4e\" => a MPEG-4 Video Elementary Stream file\n";
    *env << "\t\".mp3\" => a MPEG-1 or 2 Audio file\n";
    *env << "\t\".mpg\" => a MPEG-1 or 2 Program Stream (audio+video) file\n";
    *env << "\t\".ts\" => a MPEG Transport Stream file\n";
    *env << "\t\t(a \".tsx\" index file - if present - provides server 'trick play' support)\n";
    *env << "\t\".wav\" => a WAV Audio file\n";
    *env << "See http://www.live555.com/mediaServer/ for additional documentation.\n";
	
    // Also, attempt to create a HTTP server for RTSP-over-HTTP tunneling.
    // Try first with the default HTTP port (80), and then with the alternative HTTP
    // port numbers (8000 and 8080).
	
    if (rtspServer->setUpTunnelingOverHTTP(80) 
		|| rtspServer->setUpTunnelingOverHTTP(8000) 
		|| rtspServer->setUpTunnelingOverHTTP(8080))
    {
        *env << "(We use port " << rtspServer->httpServerPortNum() << " for optional RTSP-over-HTTP tunneling.)\n";
    }
    else
    {
        *env << "(RTSP-over-HTTP tunneling is not available.)\n";
    }
}

mediaNetServer::~mediaNetServer()
{
	
}

int mediaNetServer::start(void)
{
	if(!env
		|| !scheduler
		|| !rtspServer)
	{
		return -1;
	}
	env->taskScheduler().doEventLoop(); // does not return
	return 0;
}

int mediaNetServer::end(void)
{
	return 0;
}

int mediaNetServer::addMediaSession(void *const sessionParams)
{
	return 0;
}





static mediaNetServer *g_mediaNetServer = NULL;

int mediaNetServer_Init(const int rtspServerPort)
{	
	g_mediaNetServer = new mediaNetServer(rtspServerPort);
	if(!g_mediaNetServer)
	{
		return -1;
	}
    return 0; // only to prevent compiler warning
}

int mediaNetServer_Start(void)
{
	if(!g_mediaNetServer)
	{
		return -1;	
	}
	return g_mediaNetServer->start();
}


int mediaNetServer_addMediaSession(void *const sessionParams)
{
	if(!g_mediaNetServer)
	{
		return -1;
	}
	return g_mediaNetServer->addMediaSession(sessionParams);
}