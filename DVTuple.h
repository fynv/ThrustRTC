#ifndef _DVTuple_h
#define _DVTuple_h

#include "TRTC_api.h"
#include "DeviceViewable.h"
#include "TRTCContext.h"

class THRUST_RTC_API DVTuple : public DeviceViewable
{
public:
	DVTuple(const std::vector<CapturedDeviceViewable>& elem_map);
	virtual ViewBuf view() const;

private:
	std::vector<ViewBuf> m_view_elems;
	std::vector<size_t> m_offsets;

};

#endif
