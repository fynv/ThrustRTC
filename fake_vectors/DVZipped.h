#ifndef _DVZipped_h
#define _DVZipped_h

#include "DVVector.h"

class THRUST_RTC_API DVZipped : public DVVectorLike
{
public:
	DVZipped(TRTCContext& ctx, const std::vector<DVVectorLike*>& vecs, const std::vector<const char*>& elem_names);
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;
	virtual bool is_readable() const { return m_readable; }
	virtual bool is_writable() const { return m_writable; }

private:
	bool m_readable;
	bool m_writable;
	std::string m_name_view_cls;
	std::vector<ViewBuf> m_view_elems;
	std::vector<size_t> m_offsets;
};

#endif


