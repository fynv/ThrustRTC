#include <memory.h>
#include "DVZipped.h"

static std::string s_add_elem_struct(TRTCContext& ctx, const std::vector<DVVectorLike*>& vecs, const std::vector<const char*>& elem_names)
{
	std::string struct_body;
	for (size_t i = 0; i < vecs.size(); i++)
		struct_body += std::string("    ") + vecs[i]->name_elem_cls() + " " + elem_names[i] + ";\n";
	return ctx.add_struct(struct_body.c_str());
}

static std::string s_add_ref_struct(TRTCContext& ctx, const std::vector<DVVectorLike*>& vecs, const std::vector<const char*>& elem_names, bool& readable, bool& writable)
{
	std::string elem_cls = s_add_elem_struct(ctx, vecs, elem_names);
	readable = true;
	writable = true;
	for (size_t i = 0; i < vecs.size(); i++)
	{
		if (!vecs[i]->is_readable()) readable = false;
		if (!vecs[i]->is_writable()) writable = false;
	}

	std::string struct_body;
	for (size_t i = 0; i < vecs.size(); i++)
		struct_body += std::string("    ") + vecs[i]->name_ref_type() + " " + elem_names[i] + ";\n";

	if (readable)
	{
		struct_body += std::string("    __device__ operator ") + elem_cls + "()\n    {\n";
		struct_body += "        return {";
		for (size_t i = 0; i < vecs.size(); i++)
		{
			struct_body += elem_names[i];
			if (i < vecs.size() - 1)
				struct_body += ", ";
		}
		struct_body += "};\n    }\n";
	}

	if (writable)
	{
		struct_body += std::string("    __device__ CurType& operator = (const ") + elem_cls + "& in)\n    {\n";
		for (size_t i = 0; i < vecs.size(); i++)
			struct_body += std::string("         this->") + elem_names[i] + " = in." + elem_names[i] + ";\n";
		struct_body += "        return *this;\n    }\n";
	}

	return ctx.add_struct(struct_body.c_str());
}

DVZipped::DVZipped(TRTCContext& ctx, const std::vector<DVVectorLike*>& vecs, const std::vector<const char*>& elem_names)
	:DVVectorLike(ctx,
		s_add_elem_struct(ctx, vecs, elem_names).c_str(),
		s_add_ref_struct(ctx, vecs, elem_names, m_readable, m_writable).c_str(), vecs[0]->size())
{
	m_view_elems.resize(vecs.size());
	for (size_t i = 0; i < vecs.size(); i++)
		m_view_elems[i] = vecs[i]->view();

	std::string struct_body;
	struct_body += "    typedef " + m_elem_cls + " value_t;\n";
	struct_body += "    typedef " + m_ref_type + " ref_t;\n";
	for (size_t i = 0; i < vecs.size(); i++)
		struct_body += std::string("    ") + vecs[i]->name_view_cls() + " " + elem_names[i] + ";\n";
	struct_body += std::string("    __device__ size_t size() const\n    {\n") +
		"    return " + elem_names[0] + ".size();\n    }\n";
	struct_body += "    __device__ ref_t operator [](size_t idx)\n    {\n";
	struct_body += "        return {";
	for (size_t i = 0; i < vecs.size(); i++)
	{
		struct_body += std::string(elem_names[i]) + "[idx]";
		if (i < vecs.size() - 1)
			struct_body += ", ";
	}
	struct_body += "};\n    }\n";

	m_name_view_cls = ctx.add_struct(struct_body.c_str());

	m_offsets.resize(vecs.size() + 1);
	std::string name_struct = name_view_cls();
	ctx.query_struct(name_struct.c_str(), elem_names, m_offsets.data());
}


std::string DVZipped::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf DVZipped::view() const
{
	ViewBuf ret(m_offsets[m_offsets.size() - 1]);
	for (size_t i = 0; i < m_view_elems.size(); i++)
		memcpy(ret.data() + m_offsets[i], m_view_elems[i].data(), m_view_elems[i].size());
	return ret;
}
