#ifndef _DVDiscard_h
#define _DVDiscard_h

#include "DVVector.h"

class THRUST_RTC_API DVDiscard : public DVVectorLike
{
public:
	DVDiscard(const char* elem_cls, size_t size = (size_t)(-1));
	virtual ViewBuf view() const;
	virtual bool is_readable() const { return false; }
	virtual bool is_writable() const { return true; }

};

#endif
