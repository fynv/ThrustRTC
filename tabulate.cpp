#include "tabulate.h"

bool TRTC_Tabulate(DVVectorLike& vec, const Functor& op)
{
	static TRTC_For s_for({ "view_vec", "op" }, "idx",
		"    view_vec[idx] = op(view_vec[idx]);\n"
	);

	const DeviceViewable* args[] = { &vec, &op };
	return s_for.launch_n(vec.size(), args);
}
