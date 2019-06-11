import ThrustRTC as trtc



dvalues = trtc.device_vector_from_list([ 1, 4, 2, 8, 5, 7 ], 'int32_t')
print (trtc.Is_Sorted(dvalues))
trtc.Sort(dvalues)
print (dvalues.to_host())
print (trtc.Is_Sorted(dvalues))

dvalues = trtc.device_vector_from_list([ 1, 4, 2, 8, 5, 7 ], 'int32_t')
print (trtc.Is_Sorted(dvalues, trtc.Greater()))
trtc.Sort(dvalues, trtc.Greater())
print (dvalues.to_host())
print (trtc.Is_Sorted(dvalues, trtc.Greater()))

dkeys = trtc.device_vector_from_list([ 1, 4, 2, 8, 5, 7 ], 'int32_t')
dvalues = trtc.device_vector_from_list([ 1, 2, 3, 4, 5, 6], 'int32_t')
trtc.Sort_By_Key(dkeys, dvalues)
print (dkeys.to_host())
print (dvalues.to_host())

dkeys = trtc.device_vector_from_list([ 1, 4, 2, 8, 5, 7 ], 'int32_t')
dvalues = trtc.device_vector_from_list([ 1, 2, 3, 4, 5, 6], 'int32_t')
trtc.Sort_By_Key(dkeys, dvalues, trtc.Greater())
print (dkeys.to_host())
print (dvalues.to_host())

dvalues = trtc.device_vector_from_list([ 0, 1, 2, 3, 0, 1, 2, 3 ], 'int32_t')
print (trtc.Is_Sorted_Until(dvalues))

dvalues = trtc.device_vector_from_list([ 3, 2, 1, 0, 3, 2, 1, 0 ], 'int32_t')
print (trtc.Is_Sorted_Until(dvalues, trtc.Greater()))
