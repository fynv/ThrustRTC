import ThrustRTC as trtc
import numpy as np
import threading

harr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
darr = trtc.device_vector_from_numpy(harr)

forLoop = trtc.For(['arr_in','arr_out','k'], "idx",
	'''
	arr_out[idx] = arr_in[idx]*k;
	''')

def thread_func():
	darr_out = trtc.device_vector('float', 5)
	forLoop.launch_n(5, [darr, darr_out, trtc.DVFloat(10.0)])
	print (darr_out.to_host())

a = threading.Thread(target = thread_func)
b = threading.Thread(target = thread_func)
c = threading.Thread(target = thread_func)

a.start()
b.start()
c.start()
c.join()
b.join()
a.join()


