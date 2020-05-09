#include <memory.h>
#include "fake_vectors/DVConstant.h"

DVConstant::DVConstant(const DeviceViewable& dvobj, size_t size) :
	DVVectorLike(dvobj.name_view_cls().c_str(), (std::string("const ") + dvobj.name_view_cls() + "&").c_str(), size)
{
	m_value = dvobj.view();	
	m_name_view_cls = std::string("ConstantView<") + m_elem_cls + ">";
	TRTC_Query_Struct(m_name_view_cls.c_str(), { "_size", "_value" }, m_offsets);
}

ViewBuf DVConstant::view() const
{
	ViewBuf buf(m_offsets[2]);
	*(size_t*)(buf.data() + m_offsets[0]) = m_size;
	memcpy(buf.data() + m_offsets[1], m_value.data(), m_value.size());;
	return buf;
}
