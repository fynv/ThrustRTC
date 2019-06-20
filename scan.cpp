#include "scan.h"
#include "general_scan.h"

bool TRTC_Inclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op)
{
	Functor src({ {"vec_in", &vec_in} }, { "idx" },
		"        return vec_in[idx];\n");
	return general_scan(vec_in.size(), src, vec_out, binary_op);
}


bool TRTC_Inclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	Functor plus("Plus");
	return TRTC_Inclusive_Scan(vec_in, vec_out, plus);
}

bool TRTC_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	Functor src({ {"vec_in", &vec_in} }, { "idx" },
		"        return idx>0? vec_in[idx - 1] : (decltype(vec_in)::value_t) 0;\n");
	Functor plus("Plus");
	return general_scan(vec_in.size(), src, vec_out, plus);
}


bool TRTC_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_op)
{
	Functor src({ {"vec_in", &vec_in}, {"init", &init} }, { "idx" },
		"        return idx>0? vec_in[idx - 1] : (decltype(vec_in)::value_t)init;\n");
	return general_scan(vec_in.size(), src, vec_out, binary_op);
}


bool TRTC_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init)
{
	Functor plus("Plus");
	return TRTC_Exclusive_Scan(vec_in, vec_out, init, plus);
}

