#include <memory.h>
#include "reduce.h"
#include "general_reduce.h"

bool TRTC_Reduce(const DVVectorLike& vec, ViewBuf& ret)
{
	Functor src({ {"vec_in", &vec} }, { "idx" },
		"        return vec_in[idx];\n");
	Functor op("Plus");
	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (!general_reduce(vec.size(), vec.name_elem_cls().c_str(), src, op, ret)) return false;
	return true;
}

bool TRTC_Reduce(const DVVectorLike& vec, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret)
{
	Functor src({ {"vec_in", &vec}, {"init", &init} }, { "idx" },
		"        return idx>0 ? vec_in[idx - 1] : (decltype(vec_in)::value_t)init;\n");
	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (!general_reduce(vec.size()+1, vec.name_elem_cls().c_str(), src, binary_op, ret)) return false;
	return true;
}


bool TRTC_Reduce(const DVVectorLike& vec, const DeviceViewable& init, ViewBuf& ret)
{
	Functor op("Plus");
	return TRTC_Reduce(vec, init, op, ret);
}

#include "scan.h"
#include "general_copy_if.h"

uint32_t TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out)
{
	DVVector scan_dst(value_out.name_elem_cls().c_str(), key_in.size());
	TRTC_Inclusive_Scan_By_Key(key_in, value_in, scan_dst);	
	Functor src_scan({ {"key_in", &key_in} }, { "idx" },
		"        return idx == (key_in.size()-1) || key_in[idx]!=key_in[idx+1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(key_in.size(), src_scan, key_in, scan_dst, key_out, value_out);

}

uint32_t TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred)
{
	DVVector scan_dst(value_out.name_elem_cls().c_str(), key_in.size());
	TRTC_Inclusive_Scan_By_Key(key_in, value_in, scan_dst, binary_pred);
	Functor src_scan({ {"key_in", &key_in},  {"binary_pred", &binary_pred} }, { "idx" },
		"        return idx == (key_in.size()-1) || !binary_pred(key_in[idx],key_in[idx+1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(key_in.size(), src_scan, key_in, scan_dst, key_out, value_out);
}

uint32_t TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred, const Functor& binary_op)
{
	DVVector scan_dst(value_out.name_elem_cls().c_str(), key_in.size());
	TRTC_Inclusive_Scan_By_Key(key_in, value_in, scan_dst, binary_pred, binary_op);
	Functor src_scan({ {"key_in", &key_in},  {"binary_pred", &binary_pred} }, { "idx" },
		"        return idx == (key_in.size()-1) || !binary_pred(key_in[idx],key_in[idx+1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(key_in.size(), src_scan, key_in, scan_dst, key_out, value_out);
}
