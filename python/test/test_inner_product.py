import ThrustRTC as trtc



darr1 = trtc.device_vector_from_list([ 1.0, 2.0, 5.0], 'float')
darr2 = trtc.device_vector_from_list([ 4.0, 1.0, 5.0], 'float')
print (trtc.Inner_Product(darr1, darr2, trtc.DVFloat(0.0)))
print (trtc.Inner_Product(darr1, darr2, trtc.DVFloat(0.0), trtc.Plus(), trtc.Multiplies()))

