#include "memory.h"
#include "DVTuple.h"

DVTuple::DVTuple(const std::vector<CapturedDeviceViewable>& elem_map)
{
	std::string struct_body;
	m_view_elems.resize(elem_map.size());
	std::vector<const char*> name_elems(elem_map.size());
	for (size_t i = 0; i < elem_map.size(); i++)
	{
		struct_body += std::string("    ") + elem_map[i].obj->name_view_cls() + " " + elem_map[i].obj_name + ";\n";
		m_view_elems[i] = elem_map[i].obj->view();
		name_elems[i] = elem_map[i].obj_name;
	}

	m_name_view_cls = TRTC_Add_Struct(struct_body.c_str());
	m_offsets.resize(elem_map.size() + 1);
	TRTC_Query_Struct(m_name_view_cls.c_str(), name_elems, m_offsets.data());
}

std::string DVTuple::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf DVTuple::view() const
{
	ViewBuf ret(m_offsets[m_offsets.size() - 1]);
	for (size_t i = 0; i < m_view_elems.size(); i++)
		memcpy(ret.data() + m_offsets[i], m_view_elems[i].data(), m_view_elems[i].size());
	return ret;
}

