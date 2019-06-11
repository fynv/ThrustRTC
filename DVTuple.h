#ifndef _DVTuple_h
#define _DVTuple_h

#include "TRTC_api.h"
#include "DeviceViewable.h"
#include "TRTCContext.h"

class THRUST_RTC_API DVTuple : public DeviceViewable
{
public:
	DVTuple(const std::vector<AssignedParam>& elem_map);

	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	std::string m_name_view_cls;
	std::vector<ViewBuf> m_view_elems;
	std::vector<size_t> m_offsets;

};

#endif
