import numpy as np
import ThrustRTC as trtc
import matplotlib.pyplot as plt


def demo_histogram(d_data):

    # sort data to bring equal elements together
    trtc.Sort(d_data)

    # caculate 20 bins from 0~200
    # 1 extra to exclude possible negative values
    d_cumulative_histogram =  trtc.device_vector("int32_t", 21)

    d_counter = trtc.DVCounter(trtc.DVFloat(0.0), 21)
    d_range_ends = trtc.DVTransform(d_counter, "float", trtc.Functor({}, ['x'], '        return x*10.0;\n' ))

    trtc.Upper_Bound_V(d_data, d_range_ends, d_cumulative_histogram)

    d_histogram = trtc.device_vector("int32_t", 21)
    trtc.Adjacent_Difference(d_cumulative_histogram, d_histogram)

    h_histogram = d_histogram.to_host(1, 21)

    # plot the histogram
    x_axis = [str(x) for x in np.arange(5, 200, 10)]
    positions = np.arange(len(x_axis))
    plt.bar(positions, h_histogram, align='center', alpha=0.5)
    plt.xticks(positions, x_axis)
    plt.ylabel('Count')
    plt.title('Histogram')

    plt.show()



if __name__ == '__main__':
    h_data = np.random.randn(2000).astype(np.float32) * 30.0 +100.0
    d_data = trtc.device_vector_from_numpy(h_data)

    demo_histogram(d_data)


