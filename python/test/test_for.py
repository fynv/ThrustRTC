import ThrustRTC as trtc
import numpy as np

ctx = trtc.Context()

# interface with numpy
harr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
darr = trtc.device_vector_from_numpy(ctx, harr)

harr2 = np.array([6,7,8,9,10], dtype='int32')
darr2 = trtc.device_vector_from_numpy(ctx, harr2)

# test launching non-templated for
forLoop = trtc.For(['arr_in','arr_out','k'], "idx",
	'''
	arr_out[idx] = arr_in[idx]*k;
	''')

darr_out = trtc.device_vector(ctx, 'float', 5)
forLoop.launch_n(ctx, 5, [darr, darr_out, trtc.DVFloat(10.0)])
print (darr_out.to_host())

darr_out = trtc.device_vector(ctx, 'int32_t', 5)
forLoop.launch_n(ctx, 5, [darr2, darr_out, trtc.DVInt32(5)])
print (darr_out.to_host())

# test Context.launch_once interface, the kernel is used only once
ctx.launch_for_n(5, {'arr': darr_out}, "idx",
	'''
	arr[idx]*=100.0f;
	''')

print(darr_out.to_host())
