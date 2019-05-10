import ThrustRTC as trtc
import numpy as np

ctx = trtc.Context()

# interface with numpy
harr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
darr = trtc.device_vector_from_numpy(ctx, harr)
print(darr.to_host())

# C data type
print(darr.name_view_cls())

harr2 = np.array([6,7,8,9,10], dtype='int32')
darr2 = trtc.device_vector_from_numpy(ctx, harr2)

# kernel with auto parameters, launched twice with different types
kernel = trtc.Kernel(['arr_in', 'arr_out', 'k'],
	'''
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arr_in.size()) return;
	arr_out[idx] = arr_in[idx]*k;
	''')

darr_out = trtc.device_vector(ctx, 'float', 5)
kernel.launch(ctx, 1,128, [darr, darr_out, trtc.DVFloat(10.0)])
print (darr_out.to_host())

darr_out = trtc.device_vector(ctx, 'int32_t', 5)
kernel.launch(ctx, 1,128, [darr2, darr_out, trtc.DVInt32(5)])
print (darr_out.to_host())

# create a vector from python list with C type specified
darr3 = trtc.device_vector_from_list(ctx, [3.0, 5.0, 7.0, 9.0 , 11.0], 'float')
print(darr3.to_host())

# test Context.launch_once interface, the kernel is used only once
ctx.launch_kernel(1,128, {'arr': darr3}, 
	'''
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arr.size()) return;
	arr[idx]*=100.0f;
	''')

print(darr3.to_host())
