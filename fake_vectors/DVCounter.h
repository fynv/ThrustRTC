#ifndef _DVCounter_h
#define _DVCounter_h

#include "DVVector.h"

class THRUST_RTC_API DVCounter : public DVVectorLike
{
public:
	ViewBuf value_init() const { return m_value_init; }
	DVCounter(const DeviceViewable& dvobj_init, size_t size = (size_t)(-1));
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	ViewBuf m_value_init;
	size_t m_offsets[3];
};

#endif
