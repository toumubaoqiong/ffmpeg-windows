#include "TestdynamicRTSPServer1.h"
#include "RTSPServer.hh"
#include "liveMedia.hh"


static ServerMediaSession *TestCreateNewSMS(
					UsageEnvironment &env, 
					char const *fileName, 
					FILE*fid);


TestDynamicRTSPServer* TestDynamicRTSPServer::createNew(
		UsageEnvironment& env, 
		Port ourPort,
		UserAuthenticationDatabase* authDatabase,
		unsigned reclamationTestSeconds)
{
    int ourSocket = -1;	
    do
    {
        int ourSocket = setUpOurSocket(env, ourPort);
        if (ourSocket == -1) break;
		
        return new TestDynamicRTSPServer(env, ourSocket, ourPort, authDatabase, reclamationTestSeconds);
    }
    while (0);
	
    if (ourSocket != -1) ::closeSocket(ourSocket);
    return NULL;
}

TestDynamicRTSPServer::TestDynamicRTSPServer(
		UsageEnvironment& env, 
		int ourSocket, 
		Port ourPort,
		UserAuthenticationDatabase* authDatabase, 
		unsigned reclamationTestSeconds):
	RTSPServer(env, ourSocket, ourPort, authDatabase, reclamationTestSeconds)
{
	
}

TestDynamicRTSPServer::~TestDynamicRTSPServer()
{
}

ServerMediaSession* TestDynamicRTSPServer::lookupServerMediaSession(char const* streamName)
{
	// First, check whether the specified "streamName" exists as a local file:
    FILE *fid = fopen(streamName, "rb");
    Boolean fileExists = fid != NULL;
	
    // Next, check whether we already have a "ServerMediaSession" for this file:
    ServerMediaSession *sms = RTSPServer::lookupServerMediaSession(streamName);
    Boolean smsExists = sms != NULL;
	
    // Handle the four possibilities for "fileExists" and "smsExists":
    if (!fileExists)
    {
        if (smsExists)
        {
            // "sms" was created for a file that no longer exists. Remove it:
            removeServerMediaSession(sms);
        }
        return NULL;
    }
    else
    {
        if (!smsExists)
        {
            // Create a new "ServerMediaSession" object for streaming from the named file.
            sms = TestCreateNewSMS(envir(), streamName, fid);
            addServerMediaSession(sms);
        }
        fclose(fid);
        return sms;
    }
}


static ServerMediaSession *TestNewSMSByStr(UsageEnvironment &env,
                                        char const *fileName,
										const char *const description)
{
	ServerMediaSession *sms = NULL;
	char tmpStr[1024] = {0};
	sprintf(tmpStr, 
			"%s, streamed by the LIVE555 Media Server", 
			description);
	sms = ServerMediaSession::createNew(env, fileName, fileName, description);
	return sms;
}

static ServerMediaSession *TestCreateNewSMS(UsageEnvironment &env,
                                        char const *fileName, FILE*fid)
{
    // Use the file name extension to determine the type of "ServerMediaSession":
    char const *extension = strrchr(fileName, '.');
    if (extension == NULL) return NULL;

    ServerMediaSession *sms = NULL;
    Boolean const reuseSource = False;
    if (strcmp(extension, ".aac") == 0)
    {
        // Assumed to be an AAC Audio (ADTS format) file:
        sms = TestNewSMSByStr(env, fileName,"AAC Audio");
        sms->addSubsession(ADTSAudioFileServerMediaSubsession::createNew(env, fileName, reuseSource));
    }
    else if (strcmp(extension, ".amr") == 0)
    {
        // Assumed to be an AMR Audio file:
        sms = TestNewSMSByStr(env, fileName,"AMR Audio");
        sms->addSubsession(AMRAudioFileServerMediaSubsession::createNew(env, fileName, reuseSource));
    }
    else if (strcmp(extension, ".ac3") == 0)
    {
        // Assumed to be an AC-3 Audio file:
       sms = TestNewSMSByStr(env, fileName,"AC-3 Audio");
        sms->addSubsession(AC3AudioFileServerMediaSubsession::createNew(env, fileName, reuseSource));
    }
    else if (strcmp(extension, ".m4e") == 0)
    {
        // Assumed to be a MPEG-4 Video Elementary Stream file:
        sms = TestNewSMSByStr(env, fileName,"MPEG-4 Video");
        sms->addSubsession(MPEG4VideoFileServerMediaSubsession::createNew(env, fileName, reuseSource));
    }
    else if (strcmp(extension, ".264") == 0)
    {
        // Assumed to be a H.264 Video Elementary Stream file:
        sms = TestNewSMSByStr(env, fileName,"H.264 Video");
        OutPacketBuffer::maxSize = 100000; // allow for some possibly large H.264 frames
        sms->addSubsession(H264VideoFileServerMediaSubsession::createNew(env, fileName, reuseSource));
    }
    else if (strcmp(extension, ".mp3") == 0)
    {
        // Assumed to be a MPEG-1 or 2 Audio file:
        sms = TestNewSMSByStr(env, fileName,"MPEG-1 or 2 Audio");
        // To stream using 'ADUs' rather than raw MP3 frames, uncomment the following:
        //#define STREAM_USING_ADUS 1
        // To also reorder ADUs before streaming, uncomment the following:
        //#define INTERLEAVE_ADUS 1
        // (For more information about ADUs and interleaving,
        //  see <http://www.live555.com/rtp-mp3/>)
        Boolean useADUs = False;
        Interleaving *interleaving = NULL;
#ifdef STREAM_USING_ADUS
        useADUs = True;
#ifdef INTERLEAVE_ADUS
        unsigned char interleaveCycle[] = {0, 2, 1, 3}; // or choose your own...
        unsigned const interleaveCycleSize
        = (sizeof interleaveCycle) / (sizeof (unsigned char));
        interleaving = new Interleaving(interleaveCycleSize, interleaveCycle);
#endif
#endif
        sms->addSubsession(MP3AudioFileServerMediaSubsession::createNew(env, fileName, reuseSource, useADUs, interleaving));
    }
    else if (strcmp(extension, ".mpg") == 0)
    {
        // Assumed to be a MPEG-1 or 2 Program Stream (audio+video) file:
        sms = TestNewSMSByStr(env, fileName,"MPEG-1 or 2 Program Stream");
        MPEG1or2FileServerDemux *demux
        = MPEG1or2FileServerDemux::createNew(env, fileName, reuseSource);
        sms->addSubsession(demux->newVideoServerMediaSubsession());
        sms->addSubsession(demux->newAudioServerMediaSubsession());
    }
    else if (strcmp(extension, ".ts") == 0)
    {
        // Assumed to be a MPEG Transport Stream file:
        // Use an index file name that's the same as the TS file name, except with ".tsx":
        unsigned indexFileNameLen = strlen(fileName) + 2; // allow for trailing "x\0"
        char *indexFileName = new char[indexFileNameLen];
        sprintf(indexFileName, "%sx", fileName);
        sms = TestNewSMSByStr(env, fileName,"MPEG Transport Stream");
        sms->addSubsession(MPEG2TransportFileServerMediaSubsession::createNew(env, fileName, indexFileName, reuseSource));
        delete[] indexFileName;
    }
    else if (strcmp(extension, ".wav") == 0)
    {
        // Assumed to be a WAV Audio file:
        sms = TestNewSMSByStr(env, fileName,"WAV Audio Stream");
        // To convert 16-bit PCM data to 8-bit u-law, prior to streaming,
        // change the following to True:
        Boolean convertToULaw = False;
        sms->addSubsession(WAVAudioFileServerMediaSubsession::createNew(env, fileName, reuseSource, convertToULaw));
    }
    else if (strcmp(extension, ".dv") == 0)
    {
        // Assumed to be a DV Video file
        // First, make sure that the RTPSinks' buffers will be large enough to handle the huge size of DV frames (as big as 288000).
        OutPacketBuffer::maxSize = 300000;

        sms = TestNewSMSByStr(env, fileName,"DV Video");
        sms->addSubsession(DVVideoFileServerMediaSubsession::createNew(env, fileName, reuseSource));
    }

    return sms;
}
