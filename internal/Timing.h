#ifndef _Timing_h
#define _Timing_h

#include <string>
#include <stdio.h>

#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h> // QueryPerformanceFrequency, QueryPerformanceCounter

inline double GetTime()
{
	unsigned long long counter, frequency;
	QueryPerformanceCounter((LARGE_INTEGER*)(&counter));
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);

	return (double)counter / (double)frequency;
}

#else
# include <sys/time.h>
# include <unistd.h>

inline double GetTime()
{
	timeval tv;
	gettimeofday( &tv, NULL );
	return (double)(tv.tv_sec*1000000+tv.tv_usec)/1000000.0;
}


#endif

#endif