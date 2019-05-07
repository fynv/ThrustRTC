#include "memory.h"
#include "functor.h"

Functor::Functor(TRTCContext& ctx, const std::vector<TRTCContext::AssignedParam>& arg_map, const std::vector<const char*>& functor_params, const char* code_body)
{
	std::string struct_body;
	m_view_args.resize(arg_map.size());
	m_size_view = 0;
	for (size_t i = 0; i < arg_map.size(); i++)
	{
		struct_body += std::string("    ") + arg_map[i].arg->name_view_cls() + " " + arg_map[i].param_name + ";\n";
		m_view_args[i] = arg_map[i].arg->view();
		m_size_view += m_view_args[i].size();
	}
	if (m_size_view < 1) m_size_view = 1;

	if (functor_params.size() > 0)
	{
		struct_body += "    template<";
		for (size_t i = 0; i < functor_params.size(); i++)
		{
			char tn[32];
			sprintf(tn, "class _T%d", (int)i);
			struct_body += tn;
			if (i < functor_params.size() - 1)
				struct_body += ",";
		}
		struct_body += ">\n";
	}
	struct_body += "    __device__ inline auto operator()(";
	for (size_t i = 0; i < functor_params.size(); i++)
	{
		char param[64];
		sprintf(param, "_T%d %s", (int)i, functor_params[i]);
		struct_body += param;
		if (i < functor_params.size() - 1)
			struct_body += ",";
	}
	struct_body += ")\n    {\n";
	struct_body += code_body;
	struct_body += "    }\n";
	m_name_view_cls = ctx.add_struct(struct_body.c_str());
}

std::string Functor::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf Functor::view() const
{
	ViewBuf ret(m_size_view);
	char* p = ret.data();
	for (size_t i = 0; i < m_view_args.size(); i++)
	{
		memcpy(p, m_view_args[i].data(), m_view_args[i].size());
		p += m_view_args[i].size();
	}
	return ret;
}
