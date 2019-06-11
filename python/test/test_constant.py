import ThrustRTC as trtc


darr = trtc.device_vector_from_list([3, 7, 2, 5 ], 'int32_t')
trtc.Transform_Binary(darr, trtc.DVConstant(trtc.DVInt32(10)), darr, trtc.Plus())
print (darr.to_host())
