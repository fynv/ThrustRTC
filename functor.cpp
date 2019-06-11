#include "memory.h"
#include "functor.h"

Functor::Functor(const std::vector<CapturedDeviceViewable>& arg_map, const std::vector<const char*>& functor_params, const char* code_body)
{
	std::string struct_body;
	m_view_args.resize(arg_map.size());
	std::vector<const char*> members(arg_map.size());
	for (size_t i = 0; i < arg_map.size(); i++)
	{
		struct_body += std::string("    ") + arg_map[i].obj->name_view_cls() + " " + arg_map[i].obj_name + ";\n";
		m_view_args[i] = arg_map[i].obj->view();
		members[i] = arg_map[i].obj_name;
	}

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
		sprintf(param, "const _T%d& %s", (int)i, functor_params[i]);
		struct_body += param;
		if (i < functor_params.size() - 1)
			struct_body += ",";
	}
	struct_body += ")\n    {\n";
	struct_body += code_body;
	struct_body += "    }\n";
	m_name_view_cls = TRTC_Add_Struct(struct_body.c_str());

	m_offsets.resize(arg_map.size()+1);
	TRTC_Query_Struct(m_name_view_cls.c_str(), members, m_offsets.data());
}

Functor::Functor(const char* name_built_in_view_cls)
{
	m_name_view_cls = name_built_in_view_cls;
	m_offsets.resize(1);
	m_offsets[0] = 1;
}

std::string Functor::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf Functor::view() const
{
	ViewBuf ret(m_offsets[m_offsets.size()-1]);
	for (size_t i = 0; i < m_view_args.size(); i++)
		memcpy(ret.data()+ m_offsets[i], m_view_args[i].data(), m_view_args[i].size());
	return ret;
}

