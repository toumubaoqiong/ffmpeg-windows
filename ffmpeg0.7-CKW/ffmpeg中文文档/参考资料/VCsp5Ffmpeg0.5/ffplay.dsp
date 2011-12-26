# Microsoft Developer Studio Project File - Name="ffplay" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=ffplay - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ffplay.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ffplay.mak" CFG="ffplay - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ffplay - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "ffplay - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ffplay - Win32 Release"

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
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x804 /d "NDEBUG"
# ADD RSC /l 0x804 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "ffplay - Win32 Debug"

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
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /I "d:\work\include" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD BASE RSC /l 0x804 /d "_DEBUG"
# ADD RSC /l 0x804 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "ffplay - Win32 Release"
# Name "ffplay - Win32 Debug"
# Begin Group "libavcodec"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavcodec\4xm_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\8bps.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aasc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3tab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\adpcm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\adx.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\alac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\allcodecs.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\asv1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\avcodec.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\avs_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bitstream.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bitstream.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bmp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cabac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cabac.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cinepak.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cljr.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cook.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cookdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cscd.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cyuv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dpcm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dsputil.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dsputil.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dv_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvbsub.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvbsubdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvdsub.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvdsubenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\error_resilience.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eval.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faandct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faandct.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\fft.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ffv1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\flac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\flicvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\fraps.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\g726.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\golomb.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\golomb.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h261.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h261data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h263.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h263data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h263dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264idct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\huffyuv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\idcinvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgconvert.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgconvert_template.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgresample.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\indeo2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\indeo2data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\indeo3.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\indeo3data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\internal.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\interplayvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jfdctfst.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jfdctint.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jrevdct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lcl.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\loco.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lzo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lzo.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mace.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mdct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mem.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mmvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\motion_est.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudio.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudio.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodectab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiotab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msrle.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msvideo1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\nuv_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\opt.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\opt.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pcm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\png_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pnm_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qdm2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qdm2data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qdrw.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qtrle.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ra144.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ra144.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ra288.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ra288.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rangecoder.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rangecoder.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ratecontrol.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\raw_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\resample.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\resample2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\roqvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rpza.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rtjpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rtjpeg.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv10.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\shorten.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\simple_idct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\simple_idct.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\smacker_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\smc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\snow.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\snow.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\sonic.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\sp5x.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1_cb.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1_vlc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\truemotion1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\truemotion1data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\truemotion2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\truespeech.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\truespeech_data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\tscc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\tta_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ulti.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ulti_cb.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\utils_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc9.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc9data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vcr1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vmdav.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vqavideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmadata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmadec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wnv1.c
# End Source File
# Begin Source File

SOURCE=".\libavcodec\ws-snd1.c"
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xan.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xl.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\zmbv.c
# End Source File
# End Group
# Begin Group "libavformat"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavformat\4xm_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\adtsenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\aiff.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\allformats.c
# End Source File
# Begin Source File

SOURCE=".\libavformat\asf-enc.c"
# End Source File
# Begin Source File

SOURCE=.\libavformat\asf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\asf.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\au.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\avformat.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\avi.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\avidec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\avienc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\avio.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\avio.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\aviobuf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\avs_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\barpainet.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\crc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\cutils.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\daud.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\dv.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\dv1394.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\dv_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\electronicarts.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ffm.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\file.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flic.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flvdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flvenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\framehook.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\framehook.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\gif.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\gifdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\http.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\idcin.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\idroq.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\img.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\img2.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ipmovie.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\jpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\matroska.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mm.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mmf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mov.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mov.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\movenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mp3.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpegts.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpegts.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpegtsenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpjpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\nsvdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\nut.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\nuv_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ogg2.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ogg2.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparseflac.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparsetheora.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparsevorbis.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\os_support.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\os_support.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\png_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\pnm_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\psxstr.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\qtpalette.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\raw_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rm.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtpproto.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtsp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtsp.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtspcodes.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\segafilm.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\sgi.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\sierravmd.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\smacker_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\sol.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\swf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\tcp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\tta_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\udp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\utils_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\voc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\voc.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\wav.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\wc3movie.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\westwood.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\yuv.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\yuv4mpeg.c
# End Source File
# End Group
# Begin Group "libavutil"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavutil\avutil.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\bswap.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\common.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\crc.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\crc_util.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\integer.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\integer.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\intfloat_readwrite.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\intfloat_readwrite.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\mathematics.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\mathematics.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\rational.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\rational.h
# End Source File
# End Group
# Begin Source File

SOURCE=.\berrno.h
# End Source File
# Begin Source File

SOURCE=.\cmdutils.c
# End Source File
# Begin Source File

SOURCE=.\cmdutils.h
# End Source File
# Begin Source File

SOURCE=.\config.h
# End Source File
# Begin Source File

SOURCE=.\ffplay.c
# End Source File
# Begin Source File

SOURCE=.\update.txt
# End Source File
# End Target
# End Project
