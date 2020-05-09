#include <memory.h>
#include "fake_vectors/DVCounter.h"

DVCounter::DVCounter(const DeviceViewable& dvobj_init, size_t size) :
	DVVectorLike(dvobj_init.name_view_cls().c_str(), dvobj_init.name_view_cls().c_str(), size)
{
	m_value_init = dvobj_init.view();
	m_name_view_cls = std::string("CounterView<") + m_elem_cls + ">";
	TRTC_Query_Struct(m_name_view_cls.c_str(), { "_size", "_value_init" }, m_offsets);
}

ViewBuf DVCounter::view() const
{
	ViewBuf buf(m_offsets[2]);
	*(size_t*)(buf.data() + m_offsets[0]) = m_size;
	memcpy(buf.data() + m_offsets[1], m_value_init.data(), m_value_init.size());;
	return buf;
}