import ThrustRTC as trtc



square_root = trtc.Functor( {}, ['x'], 
'''
         return sqrtf(x);
''')

dvalues = trtc.device_vector_from_list([1.0, 4.0, 9.0, 16.0], 'float')
doutput = trtc.device_vector('float', 4)

dtrans = trtc.DVTransform(dvalues, 'float', square_root)

trtc.Transform(dtrans, doutput, trtc.Negate())
print (doutput.to_host())

