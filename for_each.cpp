#include "for_each.h"
#include "DeviceViewable.h"

bool TRTC_For_Each(DVVectorLike& vec, const Functor& f)
{
	static TRTC_For s_for({ "view_vec", "f"}, "idx",
		"    f(view_vec[idx]);\n"
	);

	const DeviceViewable* args[] = { &vec, &f };
	return s_for.launch_n(vec.size(), args);
}
