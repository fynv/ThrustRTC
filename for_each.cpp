#include "for_each.h"
#include "DeviceViewable.h"

bool TRTC_For_Each(DVVectorLike& vec, const Functor& f, size_t begin, size_t end)
{
	static TRTC_For s_for({ "view_vec", "f"}, "idx",
		"    f(view_vec[idx]);\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &f };
	return s_for.launch(begin, end, args);
}
