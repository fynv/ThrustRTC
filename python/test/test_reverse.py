import ThrustRTC as trtc



dinput = trtc.device_vector_from_list([3, 7, 2, 5], 'int32_t')
doutput = trtc.device_vector('int32_t', 4)

dreverse = trtc.DVReverse(dinput)

trtc.Transform(dreverse, doutput, trtc.Negate())
print (doutput.to_host())

