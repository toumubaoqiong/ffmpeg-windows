# Microsoft Developer Studio Project File - Name="ffmpeg" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=ffmpeg - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ffmpeg.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ffmpeg.mak" CFG="ffmpeg - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ffmpeg - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "ffmpeg - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ffmpeg - Win32 Release"

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
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib  kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib  kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "ffmpeg - Win32 Debug"

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
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ  /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ  /c
# ADD BASE RSC /l 0x804 /d "_DEBUG"
# ADD RSC /l 0x804 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib  kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib  kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "ffmpeg - Win32 Release"
# Name "ffmpeg - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\ffmpeg.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# Begin Group "libavdevice"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavdevice\alldevices.c
# End Source File
# Begin Source File

SOURCE=".\libavdevice\alsa-audio-common.c"
# End Source File
# Begin Source File

SOURCE=".\libavdevice\alsa-audio-dec.c"
# End Source File
# Begin Source File

SOURCE=".\libavdevice\alsa-audio-enc.c"
# End Source File
# Begin Source File

SOURCE=".\libavdevice\alsa-audio.h"
# End Source File
# Begin Source File

SOURCE=.\libavdevice\avdevice.h
# End Source File
# Begin Source File

SOURCE=.\libavdevice\beosaudio.cpp
# End Source File
# Begin Source File

SOURCE=.\libavdevice\bktr.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\dv1394.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\dv1394.h
# End Source File
# Begin Source File

SOURCE=.\libavdevice\libdc1394.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\oss_audio.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\v4l.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\v4l2.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\vfwcap.c
# End Source File
# Begin Source File

SOURCE=.\libavdevice\x11grab.c
# End Source File
# End Group
# Begin Group "libavfilter"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavfilter\allfilters.c
# End Source File
# Begin Source File

SOURCE=.\libavfilter\avfilter.c
# End Source File
# Begin Source File

SOURCE=.\libavfilter\avfilter.h
# End Source File
# Begin Source File

SOURCE=.\libavfilter\avfiltergraph.c
# End Source File
# Begin Source File

SOURCE=.\libavfilter\avfiltergraph.h
# End Source File
# Begin Source File

SOURCE=.\libavfilter\defaults.c
# End Source File
# Begin Source File

SOURCE=.\libavfilter\formats.c
# End Source File
# Begin Source File

SOURCE=.\libavfilter\graphparser.c
# End Source File
# Begin Source File

SOURCE=.\libavfilter\graphparser.h
# End Source File
# End Group
# Begin Group "libavutil"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavutil\adler32.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\adler32.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\aes.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\aes.h
# End Source File
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

SOURCE=.\libavutil\base64.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\base64.h
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

SOURCE=.\libavutil\crc.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\crc_data.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\des.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\des.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\fifo.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\fifo.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\integer.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\integer.h
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

SOURCE=.\libavutil\intreadwrite.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\lfg.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\lfg.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\lls.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\lls.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\log.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\log.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\lzo.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\lzo.h
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

SOURCE=.\libavutil\md5.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\mem.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\mem.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\pca.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\pca.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\pixfmt.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\random.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\random.h
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

SOURCE=.\libavutil\rc4.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\sha1.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\sha1.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\softfloat.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\softfloat.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\timer.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\tree.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\tree.h
# End Source File
# Begin Source File

SOURCE=.\libavutil\utils_util.c
# End Source File
# Begin Source File

SOURCE=.\libavutil\x86_cpu.h
# End Source File
# End Group
# Begin Group "libpostproc"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libpostproc\postprocess.c
# End Source File
# Begin Source File

SOURCE=.\libpostproc\postprocess.h
# End Source File
# Begin Source File

SOURCE=.\libpostproc\postprocess_altivec_template.c
# End Source File
# Begin Source File

SOURCE=.\libpostproc\postprocess_internal.h
# End Source File
# Begin Source File

SOURCE=.\libpostproc\postprocess_template.c
# End Source File
# End Group
# Begin Group "libswscale"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libswscale\cs_test.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\rgb2rgb.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\rgb2rgb.h
# End Source File
# Begin Source File

SOURCE=.\libswscale\rgb2rgb_template.c
# End Source File
# Begin Source File

SOURCE=".\libswscale\swscale-example.c"
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale.h
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale_altivec_template.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale_avoption.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale_bfin.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale_internal.h
# End Source File
# Begin Source File

SOURCE=.\libswscale\swscale_template.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb_altivec.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb_bfin.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb_mlib.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb_template.c
# End Source File
# Begin Source File

SOURCE=.\libswscale\yuv2rgb_vis.c
# End Source File
# End Group
# Begin Group "libavcodec"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavcodec\4xm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\8bps.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\8svx.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac_ac3_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac_ac3_parser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aac_parser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aacdectab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aacenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aacpsy.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aacpsy.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aactab.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aactab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aandcttab.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aandcttab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\aasc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3.h
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

SOURCE=.\libavcodec\ac3dec.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3dec_data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3dec_data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3tab.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ac3tab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\acelp_filters.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\acelp_filters.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\acelp_pitch_delay.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\acelp_pitch_delay.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\acelp_vectors.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\acelp_vectors.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\adpcm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\adx.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\adxdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\adxenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\alac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\alacenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\allcodecs.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\apedec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\apiexample.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\asv1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\atrac3.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\atrac3data.h
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

SOURCE=.\libavcodec\avs.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\beosthread.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bethsoftvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bethsoftvideo.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bfi.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bitstream.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bitstream.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bitstream_filter.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bmp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bmp.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bmpenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\bytestream.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\c93.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cabac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cabac.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cavs.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cavs.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cavs_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cavsdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cavsdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cavsdsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\celp_filters.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\celp_filters.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\celp_math.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\celp_math.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cinepak.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\cljr.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\colorspace.h
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

SOURCE=.\libavcodec\dca.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dca.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dca_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dcadata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dcahuff.h
# End Source File
# Begin Source File

SOURCE=".\libavcodec\dct-test.c"
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dirac_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dnxhd_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dnxhddata.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dnxhddata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dnxhddec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dnxhdenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dnxhdenc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dpcm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dsicinav.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dsputil.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dsputil.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dump_extradata_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvbsub.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvbsub_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvbsubdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvdsub_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvdsubdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dvdsubenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\dxa.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eac3dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eacmv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eaidct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eatgq.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eatgv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\eatqi.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\elbg.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\elbg.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\error_resilience.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\escape124.c
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

SOURCE=.\libavcodec\faandct.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faanidct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faanidct.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faxcompr.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\faxcompr.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\fdctref.c
# End Source File
# Begin Source File

SOURCE=".\libavcodec\fft-test.c"
# End Source File
# Begin Source File

SOURCE=.\libavcodec\fft.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ffv1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\flac.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\flacdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\flacenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\flashsv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\flashsvenc.c
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

SOURCE=.\libavcodec\g729.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\g729data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\g729dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\gif.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\gifdec.c
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

SOURCE=.\libavcodec\h261.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h261_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h261data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h261dec.c
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

SOURCE=.\libavcodec\h263_parser.h
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

SOURCE=.\libavcodec\h264.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264_mp4toannexb_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264_parser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264dspenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264idct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264pred.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\h264pred.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\huffman.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\huffman.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\huffyuv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\idcinvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\iirfilter.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\iirfilter.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imcdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgconvert.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgconvert.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgconvert_template.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imgresample.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\imx_dump_header_bsf.c
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

SOURCE=.\libavcodec\intrax8.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\intrax8.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\intrax8dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\intrax8huf.h
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

SOURCE=.\libavcodec\jpegls.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jpeglsdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jpeglsdec.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jpeglsenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\jrevdct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\kmvc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lcl.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lcldec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lclenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libamr.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libdirac.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libdirac_libschro.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libdirac_libschro.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libdiracdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libdiracenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libfaac.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libfaad.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libgsm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libmp3lame.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libopenjpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libschroedinger.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libschroedinger.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libschroedingerdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libschroedingerenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libspeexdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libtheoraenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libvorbis.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libx264.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libxvid_internal.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libxvid_rc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\libxvidff.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ljpegenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\loco.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lpc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lpc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lsp.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lzw.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lzw.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\lzwenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mace.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mathops.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mdct.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mimic.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpeg.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpeg_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpega_dump_header_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpegbdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpegdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpegdec.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpegenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mjpegenc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mlp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mlp.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mlp_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mlp_parser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mlpdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mmvideo.c
# End Source File
# Begin Source File

SOURCE=".\libavcodec\motion-test.c"
# End Source File
# Begin Source File

SOURCE=.\libavcodec\motion_est.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\motion_est_template.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\motionpixels.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\movsub_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mp3_header_compress_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mp3_header_decompress_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpc7.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpc7data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpc8.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpc8data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpc8huff.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpcdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12decdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg12enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4audio.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4audio.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4video_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpeg4video_parser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudio.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudio.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudio_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodata.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodecheader.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodecheader.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudiodectab.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegaudioenc.c
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

SOURCE=.\libavcodec\mpegvideo_common.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo_enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\mpegvideo_xvmc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msmpeg4data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msrle.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msrledec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msrledec.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\msvideo1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\nellymoser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\nellymoser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\nellymoserdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\nellymoserenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\noise_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\nuv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\opt.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\opt.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\options.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\os2thread.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\parser.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pcm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pcx.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pixdesc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pixdesc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\png.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\png.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pngdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pngenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pnm.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pnm.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pnm_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pnmenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\pthread.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ptx.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qcelp_lsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qcelpdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\qcelpdec.c
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

SOURCE=.\libavcodec\qtrleenc.c
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

SOURCE=.\libavcodec\ratecontrol.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\raw.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\raw.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rawdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rawenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rdft.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rectangle.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\remove_extradata_bsf.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\resample.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\resample2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rl.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rl2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rle.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rle.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\roqaudioenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\roqvideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\roqvideo.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\roqvideodec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\roqvideoenc.c
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

SOURCE=.\libavcodec\rv30.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv30data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv30dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv34.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv34.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv34data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv34vlc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv40.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv40data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv40dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\rv40vlc2.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\s3tc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\s3tc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\sgi.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\sgidec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\sgienc.c
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

SOURCE=.\libavcodec\smacker.c
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

SOURCE=.\libavcodec\sp5xdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\sunrast.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1_cb.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1_vlc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq1enc_cb.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\svq3.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\targa.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\targaenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\tiertexseqv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\tiff.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\tiff.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\tiffenc.c
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

SOURCE=.\libavcodec\tta.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\txd.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ulti.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\ulti_cb.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\unary.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\utils.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vb.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1acdata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vc1dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vcr1.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vdpau.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vdpau.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vdpau_internal.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vmdav.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vmnc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis_data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis_dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis_enc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vorbis_enc_data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3_parser.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp3dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp5.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp56.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp56.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp56data.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp56data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp5data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp6.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp6data.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vp6dsp.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\vqavideo.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\w32thread.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wavpack.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wma.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wma.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmadata.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmadec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmaenc.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmv2.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmv2.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmv2dec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\wmv2enc.c
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

SOURCE=.\libavcodec\xiph.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xiph.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xl.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xsubdec.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xvmc.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\xvmc_internal.h
# End Source File
# Begin Source File

SOURCE=.\libavcodec\zmbv.c
# End Source File
# Begin Source File

SOURCE=.\libavcodec\zmbvenc.c
# End Source File
# End Group
# Begin Group "libavformat"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libavformat\4xm1.c
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

SOURCE=.\libavformat\amr.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\apc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ape.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\asf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\asf.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\asfcrypt.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\asfcrypt.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\asfdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\asfenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\assdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\assenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\au.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\audiointerleave.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\audiointerleave.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\avc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\avc.h
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

SOURCE=.\libavformat\avisynth.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\avs1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\bethsoftvid.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\bfi1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\c931.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\crcenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\cutils.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\daud.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\dsicin.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\dv.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\dv1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\dvenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\dxa1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\eacdata.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\electronicarts.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ffm.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\ffmdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ffmenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\file.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flacdec1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flacenc.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\flacenc1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flic.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flv.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\flvdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\flvenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\framecrcenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\framehook.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\framehook.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\gif1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\gopher.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\gxf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\gxf.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\gxfenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\http.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\id3v2.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\id3v2.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\idcin.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\idroq.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\iff.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\img2.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\internal1.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\ipmovie.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\isom.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\isom.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\iss.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\libnut.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\lmlm4.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\matroska.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\matroska.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\matroskadec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\matroskaenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\metadata.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\metadata.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\metadata_compat.c
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

SOURCE=.\libavformat\movenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mp3.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpc1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpc81.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpeg.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpeg.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\mpegenc.c
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

SOURCE=.\libavformat\msnwc_tcp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mtv.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mvi.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mxf.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mxf.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\mxfdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\mxfenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\ncdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\network.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\nsvdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\nut.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\nut.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\nutdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\nutenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\nuv1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggdec.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\oggenc.c
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

SOURCE=.\libavformat\oma.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\options1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\os_support.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\os_support.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\psxstr.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\pva.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\qtpalette.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\r3d.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\raw1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\raw1.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rdt.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rdt.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\riff.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\riff.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rl21.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rm.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rm.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rmdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rmenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rpl.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp_aac.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp_h263.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp_h263.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp_h264.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp_h264.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtp_mpv.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtpdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtpdec.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtpenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtpenc.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\rtpenc_h264.c
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

SOURCE=.\libavformat\sdp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\segafilm.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\sierravmd.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\siff.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\smacker1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\sol.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\swf.h
# End Source File
# Begin Source File

SOURCE=.\libavformat\swfdec.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\swfenc.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\tcp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\thp.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\tiertexseq.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\tta1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\txd1.c
# End Source File
# Begin Source File

SOURCE=.\libavformat\udp.c
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

SOURCE=.\libavformat\voc.h
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
# End Target
# End Project
