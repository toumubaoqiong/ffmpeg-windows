#ifndef _OS_SUPPORT_H
#define _OS_SUPPORT_H

/**
 * @file os_support.h
 * miscellaneous OS support macros and functions.
 *
 * - usleep() (Win32, BeOS, OS/2)
 * - floatf() (OS/2)
 * - strcasecmp() (OS/2)
 */

#ifdef WIN32
__declspec(dllimport) void __stdcall Sleep(unsigned long dwMilliseconds);
// #  include <windows.h>
#  define usleep(t)    Sleep((t) / 1000)
#endif

#endif /* _OS_SUPPORT_H */
