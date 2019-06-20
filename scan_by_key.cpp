#include "scan.h"
#include "general_scan_by_key.h"

bool TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const Functor& binary_pred, const Functor& binary_op)
{
	Functor value_in({ {"value_in", &vec_value} }, { "idx" },
		"        return value_in[idx];\n");
	return general_scan_by_key(vec_key.size(), value_in, vec_key, vec_out, binary_pred, binary_op);
}


bool TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const Functor& binary_pred)
{
	Functor binary_op("Plus");
	return TRTC_Inclusive_Scan_By_Key(vec_key, vec_value, vec_out, binary_pred, binary_op);
}


bool TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out)
{
	Functor binary_pred("EqualTo");
	return TRTC_Inclusive_Scan_By_Key(vec_key, vec_value, vec_out, binary_pred);
}

bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out)
{
	Functor binary_pred("EqualTo");
	Functor binary_op("Plus");
	Functor value_in({ {"value_in", &vec_value}, {"key", &vec_key } }, { "idx" },
		"        return (idx>0 && key[idx-1] == key[idx])? value_in[idx-1]: (decltype(value_in)::value_t)0; \n");
	return general_scan_by_key(vec_key.size(), value_in, vec_key, vec_out, binary_pred, binary_op);
}


bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_pred, const Functor& binary_op)
{
	Functor value_in({ {"value_in", &vec_value}, {"key", &vec_key }, {"init", &init}, {"binary_pred", &binary_pred} }, { "idx" },
		"        return (idx>0 && binary_pred(key[idx-1], key[idx]))? value_in[idx-1]: (decltype(value_in)::value_t)init; \n");
	return general_scan_by_key(vec_key.size(), value_in, vec_key, vec_out, binary_pred, binary_op);
}


bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_pred)
{
	Functor binary_op("Plus");
	return TRTC_Exclusive_Scan_By_Key(vec_key, vec_value, vec_out, init, binary_pred, binary_op);
}

bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init)
{
	Functor binary_pred("EqualTo");
	return  TRTC_Exclusive_Scan_By_Key(vec_key, vec_value, vec_out, init, binary_pred);
}
