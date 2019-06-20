#include "transform_scan.h"
#include "general_scan.h"

bool TRTC_Transform_Inclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& unary_op, const Functor& binary_op)
{
	Functor src({ {"vec_in", &vec_in}, {"unary_op", &unary_op}, {"vec_out", &vec_out} }, { "idx" },
		"        return (decltype(vec_out)::value_t)unary_op(vec_in[idx]);\n");
	return general_scan(vec_in.size(), src, vec_out, binary_op);
}

bool TRTC_Transform_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& unary_op, const DeviceViewable& init, const Functor& binary_op)
{
	Functor src({ {"vec_in", &vec_in}, {"unary_op", &unary_op}, {"vec_out", &vec_out}, {"init", &init} }, { "idx" },
		"        return idx>0? (decltype(vec_out)::value_t)unary_op(vec_in[idx - 1]): (decltype(vec_out)::value_t)init;\n");
	return general_scan(vec_in.size(), src, vec_out, binary_op);
}

