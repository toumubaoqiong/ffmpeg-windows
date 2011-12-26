# Microsoft Developer Studio Project File - Name="ffmpeg_vc" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=ffmpeg_vc - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ffmpeg_vc.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ffmpeg_vc.mak" CFG="ffmpeg_vc - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ffmpeg_vc - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "ffmpeg_vc - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ffmpeg_vc - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x804 /d "NDEBUG"
# ADD RSC /l 0x804 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib SDL.lib SDLMain.lib /nologo /stack:0x1fffff,0xf000 /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "ffmpeg_vc - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /Zp16 /MD /W3 /Gm /GX /ZI /Od /I "..\work\include" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /GZ /c
# SUBTRACT CPP /WX
# ADD BASE RSC /l 0x804 /d "_DEBUG"
# ADD RSC /l 0x804 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib SDL.lib SDLmain.lib /nologo /stack:0x2fffff,0x10000 /subsystem:console /debug /machine:I386 /pdbtype:sept /libpath:"..\work\lib\debug"

!ENDIF 

# Begin Target

# Name "ffmpeg_vc - Win32 Release"
# Name "ffmpeg_vc - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\cmdutils.c
# End Source File
# Begin Source File

SOURCE=.\ffplay.c
# End Source File
# Begin Source File

SOURCE=.\unistd.h
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\cmdutils.h
# End Source File
# Begin Source File

SOURCE=.\inttypes.h
# End Source File
# Begin Source File

SOURCE=.\stdint.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# Begin Group "libavutil"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavutil\avstring.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\avstring.h
# End Source File
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

SOURCE=.\libavutil\crc.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\des.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\internal.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\intfloat_readwrite.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\intfloat_readwrite.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\lfg.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\log.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\log.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\mathematics.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\mathematics.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\md5.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\mem.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\mem.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\random.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\rational.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\rational.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\rc4.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\utils_util.c
# End Source File
# End Group
# Begin Group "libavformat"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavformat\allformats.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\amr.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\asf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\asfcrypt.c
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

SOURCE=.\libavformat\aviobuf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\cutils.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\dv.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\file.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\isom.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mov.c
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

SOURCE=.\libavformat\mpegtsenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparseflac.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparseogm.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparsespeex.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparsetheora.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggparsevorbis.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\raw.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\riff.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\riff.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rmdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtsp.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\sierravmd.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\txd.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\utils_format.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\vc1test.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\vc1testenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\voc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\vocdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\vocenc.c
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

SOURCE=.\libavformat\wv.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\xa.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\yuv4mpeg.c
# End Source File
# End Group
# Begin Group "libavdevice"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavdevice\alldevices.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\avdevice.h
# End Source File
# End Group
# Begin Group "libswscale"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libswscale\rgb2rgb.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale.h
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale_avoption.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale_internal.h
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb_bfin.c
# End Source File
# End Group
# Begin Group "libavcodec"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavcodec\aac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac_ac3_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aactab.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aandcttab.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3_parser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3dec_data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3tab.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\allcodecs.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\audioconvert.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\audioconvert.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\avcodec.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bitstream.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bitstream_filter.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cabac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cook.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dca.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dca_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dsputil.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eac3dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eaidct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\error_resilience.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eval.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eval.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faandct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faanidct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\fft.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\golomb.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h261.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h261enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h263.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h263.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h263_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h263dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264idct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264pred.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgconvert.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgconvert.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\intrax8.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\intrax8dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jfdctfst.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jfdctint.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jpegls.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jpeglsdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jrevdct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mdct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpegbdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpegdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4audio.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4video_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudio.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudio_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodata.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodecheader.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo_enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4data.c
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

SOURCE=.\libavcodec\rangecoder.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ratecontrol.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\raw_codec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv10.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv34.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv40.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv40dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\simple_idct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\snow.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\sp5xdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\utils.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis_data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis_dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wma.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmadec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmv2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmv2dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xiph.c
# End Source File
# End Group
# End Target
# End Project
