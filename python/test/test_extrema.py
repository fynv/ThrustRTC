import ThrustRTC as trtc

ctx = trtc.Context()

darr = trtc.device_vector_from_list(ctx, [ 1, 0, 2, 2, 1, 3], 'int32_t')
print (trtc.Min_Element(ctx, darr))
print (trtc.Max_Element(ctx, darr))
print (trtc.MinMax_Element(ctx, darr))
