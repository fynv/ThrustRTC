#ifndef _TRTC_Functor_h
#define _TRTC_Functor_h

#include "TRTCContext.h"

struct Functor
{
	std::vector<TRTCContext::AssignedParam> arg_map;
	std::vector<const char*> functor_params;
	const char* functor_ret;
	const char* code_body;

	std::string THRUST_RTC_API generate_code(const char* type_ret, const std::vector<const char*>& args) const;
};

#endif