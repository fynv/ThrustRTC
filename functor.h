#ifndef _TRTC_Functor_h
#define _TRTC_Functor_h

#include "TRTCContext.h"

class THRUST_RTC_API Functor : public DeviceViewable
{
public:
	Functor(const std::vector<CapturedDeviceViewable>& arg_map, const std::vector<const char*>& functor_params, const char* code_body);
	Functor(const char* name_built_in_view_cls);

	virtual ViewBuf view() const;

private:
	std::vector<ViewBuf> m_view_args;
	std::vector<size_t> m_offsets;
};

#endif