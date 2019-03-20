import ThrustRTC as trtc
import numpy as np

trtc.set_ptx_cache('__ptx_cache__')

ctx = trtc.Context()

harr = np.array([1,2,3,4,5], dtype='int32')
darr = trtc.device_vector_from_numpy(ctx, harr)

print(darr.to_host())
print(darr.name_view_cls())

harr2 = np.array([6,7,8,9,10], dtype='int32')
darr2 = trtc.device_vector_from_numpy(ctx, harr2)

darr_super = trtc.device_vector_from_dvs(ctx, [darr, darr2])
print(darr_super.name_view_cls())

kernel = trtc.Kernel(ctx, 
	[('VectorView<int32_t>','in1'),('VectorView<int32_t>','in2'), ('VectorView<int32_t>','out')],
	'''
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= in1.size) return;
	out[idx]=in1[idx]+in2[idx];
	''')

darr_out = trtc.device_vector(ctx, 'int32_t', 5)
kernel.launch(1, 128, [darr, darr2, darr_out])
print (darr_out.to_host())

kernelTempl = trtc.KernelTemplate(ctx,
	['T'],
	[("VectorView<T>", "arr_in"), ("VectorView<T>", "arr_out"), ("double", "k")],
	'''
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arr_in.size) return;
	arr_out[idx] = arr_in[idx]*k;
	''')
kernelTempl.launch(1,128, [darr2, darr_out, trtc.DVDouble(4.0)])
print (darr_out.to_host())

darr3 = trtc.device_vector_from_list(ctx, [3.0, 5.0, 7.0, 9.0 , 11.0], 'float')
print(darr3.to_host())