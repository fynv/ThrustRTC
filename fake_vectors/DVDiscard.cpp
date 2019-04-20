#include "DVDiscard.h"

DVDiscard::DVDiscard(TRTCContext& ctx, const char* elem_cls, size_t size)
	:DVVectorLike(ctx, elem_cls, size){}

std::string DVDiscard::name_view_cls() const
{
	return std::string("DiscardView<") + m_elem_cls + ">";
}

ViewBuf DVDiscard::view() const
{
	ViewBuf buf(sizeof(size_t));
	size_t& size = *(size_t*)(buf.data());
	size = m_size;
	return buf;
}


