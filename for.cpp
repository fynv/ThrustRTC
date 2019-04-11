#include "for.h"
#include <string>

TRTC_For_Template::TRTC_For_Template(const std::vector<const char*>& template_params, const std::vector<TRTCContext::ParamDesc>& _params, const char* name_iter, const char* _body)
{
	std::vector<TRTCContext::ParamDesc> params = _params;
	params.push_back({ "size_t", "_begin" });
	params.push_back({ "size_t", "_end" });

	std::string body = std::string("    size_t ") + name_iter + " = threadIdx.x + blockIdx.x*blockDim.x + _begin;\n"
		"    if (" + name_iter + ">=_end) return; \n" + _body;

	m_ker_templ = new TRTCContext::KernelTemplate(template_params, params, body.c_str());
}

TRTC_For_Template::~TRTC_For_Template()
{
	delete m_ker_templ;
}

size_t TRTC_For_Template::num_template_params() const
{
	return m_ker_templ->num_template_params();
}

size_t TRTC_For_Template::num_params() const
{
	return m_ker_templ->num_params() - 2;
}

const std::string* TRTC_For_Template::type_params()
{
	return m_ker_templ->type_params();
}

bool TRTC_For_Template::deduce_template_args(const DeviceViewable** _args, std::vector<std::string>& template_args) const
{
	size_t total_params = m_ker_templ->num_params();
	std::vector<const DeviceViewable*> args(total_params);
	for (int i = 0; i <total_params - 2; i++)
		args[i] = _args[i];

	DVSizeT begin(0), end(0);
	args[total_params - 2] = &begin;
	args[total_params - 1] = &end;

	m_ker_templ->deduce_template_args(args.data(), template_args);
}

void TRTC_For_Template::launch_explict(TRTCContext& ctx, const std::vector<std::string>& template_args, size_t begin, size_t end, const DeviceViewable** args, unsigned sharedMemBytes)
{
	TRTC_For concrete(ctx, *this, template_args);
	concrete.launch(begin, end, args, sharedMemBytes);
}

bool TRTC_For_Template::launch(TRTCContext& ctx, size_t begin, size_t end, const DeviceViewable** args, unsigned sharedMemBytes)
{
	std::vector<std::string> template_args;
	if (deduce_template_args(args, template_args))
	{
		size_t total = num_template_params();
		if (template_args.size() >= total)
		{
			size_t i = 0;
			for (; i <total; i++)
				if (template_args[i].size() < 1) break;
			if (i >= total)
			{
				TRTC_For concrete(ctx, *this, template_args);
				concrete.launch(begin, end, args, sharedMemBytes);
				return true;
			}
		}
	}
	const std::string* t_params = type_params();

	puts("Failed to deduce some of the template arguments.");
	puts("Parameter types:");
	for (size_t i = 0; i < num_params(); i++)
		printf("%s, ", t_params[i].c_str());
	puts("\nArgument types:");
	for (size_t i = 0; i < num_params(); i++)
		printf("%s, ", args[i]->name_view_cls().c_str());
	puts("");

	return false;
}


TRTC_For::TRTC_For(TRTCContext& ctx, TRTC_For_Template& templ, const std::vector<std::string>& template_args) : m_ctx(ctx)
{
	m_ker_id = templ.m_ker_templ->instantiate(ctx, template_args);
}


TRTC_For::TRTC_For(TRTCContext& ctx, const std::vector<TRTCContext::ParamDesc>& params, const char* name_iter, const char* body) : m_ctx(ctx)
{
	TRTC_For_Template templ({}, params, name_iter, body);
	m_ker_id = templ.m_ker_templ->instantiate(ctx, {});
}

size_t TRTC_For::num_params() const
{
	return m_ctx.get_num_of_params(m_ker_id) - 2;
}

void TRTC_For::launch(size_t begin, size_t end, const DeviceViewable** _args, unsigned sharedMemBytes) const
{
	unsigned num_blocks = (unsigned)((end - begin + 127) / 128);

	size_t total_params = m_ctx.get_num_of_params(m_ker_id);
	std::vector<const DeviceViewable*> args(total_params);
	for (int i = 0; i <total_params - 2; i++)
		args[i] = _args[i];

	DVSizeT dvbegin(begin), dvend(end);
	args[total_params - 2] = &dvbegin;
	args[total_params - 1] = &dvend;

	m_ctx.launch_kernel(m_ker_id, { num_blocks, 1, 1 }, { 128,1,1 }, args.data(), sharedMemBytes);
}

void TRTC_For_Once(TRTCContext& ctx, size_t begin, size_t end, const std::vector<TRTCContext::AssignedParam>& arg_map, const char* name_iter, const char* code_body, unsigned sharedMemBytes)
{
	size_t num_params = arg_map.size();
	std::vector<TRTCContext::ParamDesc> params(num_params);
	std::vector<std::string> param_types(num_params);
	std::vector<const DeviceViewable*> args(num_params);
	for (size_t i = 0; i < num_params; i++)
	{
		param_types[i] = arg_map[i].arg->name_view_cls();
		params[i] = { param_types[i].c_str(), arg_map[i].param_name };
		args[i] = arg_map[i].arg;
	}
	TRTC_For concrete(ctx, params, name_iter, code_body);
	concrete.launch(begin, end, args.data(), sharedMemBytes);
}


