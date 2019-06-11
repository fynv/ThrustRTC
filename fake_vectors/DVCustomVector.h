#ifndef _DVCustomVector_h
#define _DVCustomVector_h

#include "TRTC_api.h"
#include "DVVector.h"
#include "TRTCContext.h"

class THRUST_RTC_API DVCustomVector : public DVVectorLike
{
public:
	DVCustomVector(const std::vector<AssignedParam>& arg_map, const char* name_idx, const char* code_body, 
		const char* elem_cls, size_t size = (size_t)(-1), bool read_only = true);

	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

	virtual bool is_writable() const { return !m_read_only; }
	   
private:
	std::string m_name_view_cls;
	size_t m_size;
	bool m_read_only;
	std::vector<ViewBuf> m_view_args;
	std::vector<size_t> m_arg_offsets;
	size_t m_offsets[3];
};


#endif
