/*
    File: strings.h
    Copyright: Public Domain

    This file is provided because non ANSI fuctions are described in string.h 
    that belong in strings.h.  These functions are provided for in the OLDNAME
    libraries.
*/
#if !defined(_STRINGS_H_)
# define _STRINGS_H_ 1
# include <string.h>

#if defined(__WIN32__) || defined(_WIN32) || defined(_QNX4)
#define strcasecmp   _stricmp
#define strncasecmp  _strnicmp
#define snprintf     _snprintf
#define vsnprintf    _vsnprintf
#endif

#endif
