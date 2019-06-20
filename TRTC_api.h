#ifndef _TRTC_API_h
#define _TRTC_API_h

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#pragma warning( disable: 4275 )
#pragma warning( disable: 4251 )
#pragma warning( disable: 4530 )
#if defined THRUST_RTC_DLL_EXPORT
#define THRUST_RTC_API __declspec(dllexport)
#elif defined THRUST_RTC_DLL_IMPORT
#define THRUST_RTC_API __declspec(dllimport)
#endif
#endif

#ifndef THRUST_RTC_API
#define THRUST_RTC_API
#endif

#endif

