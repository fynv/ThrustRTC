#include <memory.h>
#include "fake_vectors/DVConstant.h"

DVConstant::DVConstant(TRTCContext& ctx, const DeviceViewable& dvobj, size_t size) :
	DVVectorLike(ctx, dvobj.name_view_cls().c_str(), size)
{
	m_value = dvobj.view();
}

std::string DVConstant::name_view_cls() const
{
	return std::string("ConstantView<") + m_elem_cls + ">";
}

ViewBuf DVConstant::view() const
{
	ViewBuf buf(m_elem_size + sizeof(size_t));
	memcpy(buf.data(), m_value.data(), m_value.size());
	size_t& size = *(size_t*)(buf.data() + m_elem_size);
	size = m_size;
	return buf;
}