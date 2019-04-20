#include <memory.h>
#include "fake_vectors/DVCounter.h"

DVCounter::DVCounter(TRTCContext& ctx, const DeviceViewable& dvobj_init, size_t size) :
	DVVectorLike(ctx, dvobj_init.name_view_cls().c_str(), size)
{
	m_value_init = dvobj_init.view();
}

std::string DVCounter::name_view_cls() const
{
	return std::string("CounterView<") + m_elem_cls + ">";
}

ViewBuf DVCounter::view() const
{
	ViewBuf buf(m_elem_size + sizeof(size_t));
	memcpy(buf.data(), m_value_init.data(), m_value_init.size());
	size_t& size = *(size_t*)(buf.data() + m_elem_size);
	size = m_size;
	return buf;
}