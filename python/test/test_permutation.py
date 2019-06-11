import ThrustRTC as trtc



dvalues = trtc.device_vector_from_list([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 'float')
dindices =  trtc.device_vector_from_list([2,6,1,3], 'int32_t')
doutput = trtc.device_vector('float', 4)

perm = trtc.DVPermutation(dvalues, dindices)

trtc.Transform(perm, doutput, trtc.Negate())
print (doutput.to_host())

