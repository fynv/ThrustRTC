#include "transform_scan.h"
#include "general_scan.h"

bool TRTC_Transform_Inclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& unary_op, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src({ {"vec_in", &vec_in}, {"unary_op", &unary_op}, {"vec_out", &vec_out}, {"begin_in", &dvbegin_in } }, { "idx" },
		"        return (decltype(vec_out)::value_t)unary_op(vec_in[idx + begin_in]);\n");
	return general_scan(n, src, vec_out, binary_op, begin_out);
}

bool TRTC_Transform_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& unary_op, const DeviceViewable& init, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src({ {"vec_in", &vec_in}, {"unary_op", &unary_op}, {"vec_out", &vec_out}, {"begin_in", &dvbegin_in }, {"init", &init} }, { "idx" },
		"        return idx>0? (decltype(vec_out)::value_t)unary_op(vec_in[idx - 1 + begin_in]): (decltype(vec_out)::value_t)init;\n");
	return general_scan(n, src, vec_out, binary_op, begin_out);
}

