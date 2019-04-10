#include "fill.h"
#include "TRTCContext.h"

bool TRTC_Fill(TRTCContext& ctx, DVVector& vec, const DeviceViewable& value, size_t begin, size_t end)
{
	static TRTCContext::KernelTemplate s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec" }, { "T", "value" }, { "size_t", "begin" }, { "size_t", "end" } },
	"    size_t id = threadIdx.x + blockIdx.x*blockDim.x + begin;\n"
	"    if (id>=end) return;\n"
	"    view_vec[id]=value;\n");

	if (vec.name_elem_cls() != value.name_view_cls())
	{
		printf("TRTC_Fill: vector type mismatch with value type.\n");
		return false;
	}

	if (end == (size_t)(-1)) end = vec.size();
	unsigned num_blocks = (unsigned)((end - begin + 127) / 128);
	DVSizeT dvbegin(begin), dvend(end);
	const DeviceViewable* args[] = { &vec, &value, &dvbegin, &dvend };
	ctx.launch_kernel_template(s_templ, { num_blocks, 1, 1 }, { 128, 1, 1 }, args);

	return true;
}
