# Microsoft Developer Studio Project File - Name="liveMedia" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=liveMedia - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "liveMedia.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "liveMedia.mak" CFG="liveMedia - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "liveMedia - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "liveMedia - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "liveMedia - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD BASE RSC /l 0x804 /d "NDEBUG"
# ADD RSC /l 0x804 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "liveMedia - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD BASE RSC /l 0x804 /d "_DEBUG"
# ADD RSC /l 0x804 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=copy .\Debug\liveMedia.lib ..\debug\liveMedia.lib
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "liveMedia - Win32 Release"
# Name "liveMedia - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\AC3AudioFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\AC3AudioRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\AC3AudioRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\AC3AudioStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\ADTSAudioFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\ADTSAudioFileSource.cpp
# End Source File
# Begin Source File

SOURCE=.\AMRAudioFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\AMRAudioFileSink.cpp
# End Source File
# Begin Source File

SOURCE=.\AMRAudioFileSource.cpp
# End Source File
# Begin Source File

SOURCE=.\AMRAudioRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\AMRAudioRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\AMRAudioSource.cpp
# End Source File
# Begin Source File

SOURCE=.\AudioInputDevice.cpp
# End Source File
# Begin Source File

SOURCE=.\AudioRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\AVIFileSink.cpp
# End Source File
# Begin Source File

SOURCE=.\Base64.cpp
# End Source File
# Begin Source File

SOURCE=.\BasicUDPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\BasicUDPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\BitVector.cpp
# End Source File
# Begin Source File

SOURCE=.\ByteStreamFileSource.cpp
# End Source File
# Begin Source File

SOURCE=.\ByteStreamMultiFileSource.cpp
# End Source File
# Begin Source File

SOURCE=.\DarwinInjector.cpp
# End Source File
# Begin Source File

SOURCE=.\DeviceSource.cpp
# End Source File
# Begin Source File

SOURCE=.\DigestAuthentication.cpp
# End Source File
# Begin Source File

SOURCE=.\DVVideoFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\DVVideoRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\DVVideoRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\DVVideoStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\FileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\FileSink.cpp
# End Source File
# Begin Source File

SOURCE=.\FramedFileSource.cpp
# End Source File
# Begin Source File

SOURCE=.\FramedFilter.cpp
# End Source File
# Begin Source File

SOURCE=.\FramedSource.cpp
# End Source File
# Begin Source File

SOURCE=.\GSMAudioRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\H261VideoRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\H263plusVideoFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\H263plusVideoRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\H263plusVideoRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\H263plusVideoStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\H263plusVideoStreamParser.cpp
# End Source File
# Begin Source File

SOURCE=.\H264VideoFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\H264VideoFileSink.cpp
# End Source File
# Begin Source File

SOURCE=.\H264VideoRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\H264VideoRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\H264VideoStreamDiscreteFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\H264VideoStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\HTTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\InputFile.cpp
# End Source File
# Begin Source File

SOURCE=.\JPEGVideoRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\JPEGVideoRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\JPEGVideoSource.cpp
# End Source File
# Begin Source File

SOURCE=.\Locale.cpp
# End Source File
# Begin Source File

SOURCE=.\Media.cpp
# End Source File
# Begin Source File

SOURCE=.\MediaSession.cpp
# End Source File
# Begin Source File

SOURCE=.\MediaSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MediaSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3ADU.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3ADUdescriptor.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3ADUinterleaving.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3ADURTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3ADURTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3ADUTranscoder.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3AudioFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3FileSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3HTTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3Internals.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3InternalsHuffman.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3InternalsHuffmanTable.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3StreamState.cpp
# End Source File
# Begin Source File

SOURCE=.\MP3Transcoder.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2AudioRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2AudioRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2AudioStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2Demux.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2DemuxedElementaryStream.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2DemuxedServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2FileServerDemux.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2VideoFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2VideoHTTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2VideoRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2VideoRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2VideoStreamDiscreteFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG1or2VideoStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2IndexFromTransportStream.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2TransportFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2TransportStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2TransportStreamFromESSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2TransportStreamFromPESSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2TransportStreamIndexFile.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2TransportStreamMultiplexor.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG2TransportStreamTrickModeFilter.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4ESVideoRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4ESVideoRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4GenericRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4GenericRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4LATMAudioRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4LATMAudioRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4VideoFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4VideoStreamDiscreteFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEG4VideoStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEGVideoStreamFramer.cpp
# End Source File
# Begin Source File

SOURCE=.\MPEGVideoStreamParser.cpp
# End Source File
# Begin Source File

SOURCE=.\MultiFramedRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\MultiFramedRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\OnDemandServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\our_md5.c
# End Source File
# Begin Source File

SOURCE=.\our_md5hl.c
# End Source File
# Begin Source File

SOURCE=.\OutputFile.cpp
# End Source File
# Begin Source File

SOURCE=.\PassiveServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\QCELPAudioRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\QuickTimeFileSink.cpp
# End Source File
# Begin Source File

SOURCE=.\QuickTimeGenericRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\RTCP.cpp
# End Source File
# Begin Source File

SOURCE=.\rtcp_from_spec.c
# End Source File
# Begin Source File

SOURCE=.\RTPInterface.cpp
# End Source File
# Begin Source File

SOURCE=.\RTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\RTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\RTSPClient.cpp
# End Source File
# Begin Source File

SOURCE=.\RTSPCommon.cpp
# End Source File
# Begin Source File

SOURCE=.\RTSPServer.cpp
# End Source File
# Begin Source File

SOURCE=.\ServerMediaSession.cpp
# End Source File
# Begin Source File

SOURCE=.\SimpleRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\SimpleRTPSource.cpp
# End Source File
# Begin Source File

SOURCE=.\SIPClient.cpp
# End Source File
# Begin Source File

SOURCE=.\StreamParser.cpp
# End Source File
# Begin Source File

SOURCE=.\uLawAudioFilter.cpp
# End Source File
# Begin Source File

SOURCE=.\VideoRTPSink.cpp
# End Source File
# Begin Source File

SOURCE=.\WAVAudioFileServerMediaSubsession.cpp
# End Source File
# Begin Source File

SOURCE=.\WAVAudioFileSource.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\our_md5.h
# End Source File
# Begin Source File

SOURCE=.\rtcp_from_spec.h
# End Source File
# End Group
# End Target
# End Project
