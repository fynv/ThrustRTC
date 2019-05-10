import ThrustRTC as trtc

ctx = trtc.Context()

darr1 = trtc.device_vector_from_list(ctx, [  10, 20, 30, 40, 50, 60, 70, 80], 'int32_t')
darr2 = trtc.device_vector_from_list(ctx, [  1000, 900, 800, 700, 600, 500, 400, 300 ], 'int32_t')

trtc.Swap(ctx, darr1, darr2)
print (darr1.to_host())
print (darr2.to_host())
