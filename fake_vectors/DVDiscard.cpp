#include "DVDiscard.h"

DVDiscard::DVDiscard(const char* elem_cls, size_t size)
	:DVVectorLike(elem_cls, (std::string("_Sink<")+ elem_cls +">&").c_str(), size)
{
	m_name_view_cls = std::string("DiscardView<") + m_elem_cls + ">";
}

ViewBuf DVDiscard::view() const
{
	ViewBuf buf(sizeof(size_t) + 1);
	size_t& size = *(size_t*)(buf.data());
	size = m_size;
	return buf;
}


