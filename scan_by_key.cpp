#include "scan.h"
#include "general_scan_by_key.h"

bool TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out)
{
	if (end_key == (size_t)(-1)) end_key = vec_key.size();
	size_t n = end_key - begin_key;
	DVSizeT dvbegin_value(begin_value);
	Functor value_in({ {"value_in", &vec_value}, {"begin_value", &dvbegin_value } }, { "idx" },
		"        return value_in[idx + begin_value];\n");
	Functor binary_pred("EqualTo");
	Functor binary_op("Plus");
	return general_scan_by_key(n, value_in, vec_key, vec_out, binary_pred, binary_op, begin_key, begin_out);
}

bool TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const Functor& binary_pred, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out)
{
	if (end_key == (size_t)(-1)) end_key = vec_key.size();
	size_t n = end_key - begin_key;
	DVSizeT dvbegin_value(begin_value);
	Functor value_in({ {"value_in", &vec_value}, {"begin_value", &dvbegin_value } }, { "idx" },
		"        return value_in[idx + begin_value];\n");
	Functor binary_op("Plus");
	return general_scan_by_key(n, value_in, vec_key, vec_out, binary_pred, binary_op, begin_key, begin_out);
}

bool TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const Functor& binary_pred, const Functor& binary_op, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out)
{
	if (end_key == (size_t)(-1)) end_key = vec_key.size();
	size_t n = end_key - begin_key;
	DVSizeT dvbegin_value(begin_value);
	Functor value_in({ {"value_in", &vec_value}, {"begin_value", &dvbegin_value } }, { "idx" },
		"        return value_in[idx + begin_value];\n");
	return general_scan_by_key(n, value_in, vec_key, vec_out, binary_pred, binary_op, begin_key, begin_out);
}

bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out)
{
	if (end_key == (size_t)(-1)) end_key = vec_key.size();
	size_t n = end_key - begin_key;
	DVSizeT dvbegin_value(begin_value);
	DVSizeT dvbegin_key(begin_key);
	Functor binary_pred("EqualTo");
	Functor binary_op("Plus");
	Functor value_in({ {"value_in", &vec_value}, {"begin_value", &dvbegin_value }, {"key", &vec_key }, {"begin_key", &dvbegin_key } }, { "idx" },
		"        return (idx>0 && key[idx-1] == key[idx])? value_in[idx-1]: (decltype(value_in)::value_t)0; \n");
	return general_scan_by_key(n, value_in, vec_key, vec_out, binary_pred, binary_op, begin_key, begin_out);
}

bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out)
{
	if (end_key == (size_t)(-1)) end_key = vec_key.size();
	size_t n = end_key - begin_key;
	DVSizeT dvbegin_value(begin_value);
	DVSizeT dvbegin_key(begin_key);
	Functor binary_pred("EqualTo");
	Functor binary_op("Plus");
	Functor value_in({ {"value_in", &vec_value}, {"begin_value", &dvbegin_value }, {"key", &vec_key }, {"begin_key", &dvbegin_key }, {"init", &init} }, { "idx" },
		"        return (idx>0 && key[idx-1] == key[idx])? value_in[idx-1]: (decltype(value_in)::value_t)init; \n");
	return general_scan_by_key(n, value_in, vec_key, vec_out, binary_pred, binary_op, begin_key, begin_out);
}

bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_pred, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out)
{
	if (end_key == (size_t)(-1)) end_key = vec_key.size();
	size_t n = end_key - begin_key;
	DVSizeT dvbegin_value(begin_value);
	DVSizeT dvbegin_key(begin_key);
	Functor binary_op("Plus");
	Functor value_in({ {"value_in", &vec_value}, {"begin_value", &dvbegin_value }, {"key", &vec_key }, {"begin_key", &dvbegin_key }, {"init", &init}, {"binary_pred", &binary_pred} }, { "idx" },
		"        return (idx>0 && binary_pred(key[idx-1], key[idx]))? value_in[idx-1]: (decltype(value_in)::value_t)init; \n");
	return general_scan_by_key(n, value_in, vec_key, vec_out, binary_pred, binary_op, begin_key, begin_out);
}

bool TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_pred, const Functor& binary_op, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out)
{
	if (end_key == (size_t)(-1)) end_key = vec_key.size();
	size_t n = end_key - begin_key;
	DVSizeT dvbegin_value(begin_value);
	DVSizeT dvbegin_key(begin_key);
	Functor value_in({ {"value_in", &vec_value}, {"begin_value", &dvbegin_value }, {"key", &vec_key }, {"begin_key", &dvbegin_key }, {"init", &init}, {"binary_pred", &binary_pred} }, { "idx" },
		"        return (idx>0 && binary_pred(key[idx-1], key[idx]))? value_in[idx-1]: (decltype(value_in)::value_t)init; \n");
	return general_scan_by_key(n, value_in, vec_key, vec_out, binary_pred, binary_op, begin_key, begin_out);

}

