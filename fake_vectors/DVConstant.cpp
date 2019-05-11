#include <memory.h>
#include "fake_vectors/DVConstant.h"

DVConstant::DVConstant(TRTCContext& ctx, const DeviceViewable& dvobj, size_t size) :
	DVVectorLike(ctx, dvobj.name_view_cls().c_str(), (std::string("const ") + dvobj.name_view_cls() + "&").c_str(), size)
{
	m_value = dvobj.view();
	
	std::string name_struct = name_view_cls();
	ctx.query_struct(name_struct.c_str(), { "_size", "_value" }, m_offsets);
}

std::string DVConstant::name_view_cls() const
{
	return std::string("ConstantView<") + m_elem_cls + ">";
}

ViewBuf DVConstant::view() const
{
	ViewBuf buf(m_offsets[2]);
	*(size_t*)(buf.data() + m_offsets[0]) = m_size;
	memcpy(buf.data() + m_offsets[1], m_value.data(), m_value.size());;
	return buf;
}
