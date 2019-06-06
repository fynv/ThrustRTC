#include <stdio.h>
#include <memory.h>
#include "DVCustomVector.h"

DVCustomVector::DVCustomVector(TRTCContext& ctx, const std::vector<TRTCContext::AssignedParam>& arg_map, 
	const char* name_idx, const char* code_body, const char* elem_cls, size_t size, bool read_only)
	: DVVectorLike(ctx, elem_cls, read_only ? elem_cls : (std::string(elem_cls) + "&").c_str(), size), m_size(size), m_read_only(read_only)
{
	std::string functor_body;
	m_view_args.resize(arg_map.size());
	std::vector<const char*> args(arg_map.size());
	for (size_t i = 0; i < arg_map.size(); i++)
	{
		functor_body += std::string("    ") + arg_map[i].arg->name_view_cls() + " " + arg_map[i].param_name + ";\n";
		m_view_args[i] = arg_map[i].arg->view();
		args[i] = arg_map[i].param_name;
	}

	functor_body += "    __device__ inline "+ m_ref_type + " operator()(size_t " + name_idx + ")\n    {\n";
	functor_body += code_body;
	functor_body += "    }\n";

	std::string name_functor_cls = ctx.add_struct(functor_body.c_str());
	m_arg_offsets.resize(arg_map.size() + 1);
	ctx.query_struct(name_functor_cls.c_str(), args, m_arg_offsets.data());

	std::string struct_body;
	struct_body += "    typedef " + m_elem_cls + " value_t;\n";
	struct_body += "    typedef " + m_ref_type + " ref_t;\n";
	struct_body += "    size_t _size;\n";
	struct_body += "    " + name_functor_cls + " _op;\n";
	struct_body += std::string("    __device__ inline size_t size() const\n    {\n") +
		"        return _size;\n    }\n";
	struct_body += "    __device__ ref_t operator [](size_t idx)\n    {\n";
	struct_body += "        return _op(idx);\n    }\n";
	m_name_view_cls = ctx.add_struct(struct_body.c_str());
	
	ctx.query_struct(m_name_view_cls.c_str(), { "_size", "_op" }, m_offsets);
}


std::string DVCustomVector::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf DVCustomVector::view() const
{
	ViewBuf view_op(m_arg_offsets[m_arg_offsets.size() - 1]);
	for (size_t i = 0; i < m_view_args.size(); i++)
		memcpy(view_op.data() + m_arg_offsets[i], m_view_args[i].data(), m_view_args[i].size());

	ViewBuf ret(m_offsets[2]);
	memcpy(ret.data() + m_offsets[0], &m_size, sizeof(size_t));
	memcpy(ret.data() + m_offsets[1], view_op.data(), view_op.size());
	return ret;
}
