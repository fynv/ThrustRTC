#include <memory.h>
#include "fake_vectors/DVCounter.h"

DVCounter::DVCounter(TRTCContext& ctx, const DeviceViewable& dvobj_init, size_t size) :
	DVVectorLike(ctx, dvobj_init.name_view_cls().c_str(), size)
{
	m_value_init = dvobj_init.view();

	std::string name_struct = name_view_cls();
	ctx.query_struct(name_struct.c_str(), { "_size", "_value_init" }, m_offsets);
}

std::string DVCounter::name_view_cls() const
{
	return std::string("CounterView<") + m_elem_cls + ">";
}

ViewBuf DVCounter::view() const
{
	ViewBuf buf(m_offsets[2]);
	*(size_t*)(buf.data() + m_offsets[0]) = m_size;
	memcpy(buf.data() + m_offsets[1], m_value_init.data(), m_value_init.size());;
	return buf;
}