#ifndef _Timing_h
#define _Timing_h

#include <string>
#include <stdio.h>
#include <list>

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

struct EventItem
{
	const char* mark;
	double time;
};

inline std::list<EventItem>& s_event_list()
{
	static std::list<EventItem> _event_list;
	return _event_list;
}

inline void s_put_event(const char* mark)
{
	std::list<EventItem>& event_list = s_event_list();
	event_list.push_back({ mark, GetTime() });
}

inline void s_print_events()
{
	std::list<EventItem>& event_list = s_event_list();
	std::list<EventItem>::iterator it = event_list.begin();
	while (it != event_list.end())
	{
		printf("%s: %f\n", it->mark, it->time);
		it++;
	}
	event_list.clear();
}

#endif