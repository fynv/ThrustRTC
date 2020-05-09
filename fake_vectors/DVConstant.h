#ifndef _DVConstant_h
#define _DVConstant_h

#include "DVVector.h"

class THRUST_RTC_API DVConstant : public DVVectorLike
{
public:
	ViewBuf value() const { return m_value; }
	DVConstant(const DeviceViewable& dvobj, size_t size = (size_t)(-1));
	virtual ViewBuf view() const;

private:
	ViewBuf m_value;
	size_t m_offsets[3];
};

#endif
