import ThrustRTC as trtc



darr = trtc.device_vector_from_list([ 1, 0, 2, 2, 1, 3], 'int32_t')
print (trtc.Min_Element(darr))
print (trtc.Max_Element(darr))
print (trtc.MinMax_Element(darr))
