import numpy as np
import ThrustRTC as trtc
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def demo_k_means(d_x, d_y, k):

    n = d_x.size()

    # create a zipped vector for convenience
    d_points = trtc.DVZipped([d_x, d_y], ['x','y'])

    # operations
    point_plus = trtc.Functor({ }, ['pos1', "pos2"],
'''
        return decltype(pos1)({pos1.x + pos2.x, pos1.y + pos2.y});
''')

    point_div = trtc.Functor({ }, ['pos', "count"],
'''
        return decltype(pos)({pos.x/(float)count, pos.y/(float)count});
''')
    
    # initialize centers
    center_ids = [0] * k
    d_min_dis = trtc.device_vector("float", n)

    for i in range(1, k):
        d_count = trtc.DVInt32(i)
        d_center_ids =  trtc.device_vector_from_list(center_ids[0:i], 'int32_t')
        calc_min_dis = trtc.Functor({"points": d_points, "center_ids": d_center_ids, "count": d_count }, ['pos'], 
'''
        float minDis = FLT_MAX;
        for (int i=0; i<count; i++)
        {
            int j = center_ids[i];
            float dis = (pos.x - points[j].x)*(pos.x - points[j].x);
            dis+= (pos.y - points[j].y)*(pos.y - points[j].y);
            if (dis<minDis) minDis = dis;
        }
        return minDis;
''')
        trtc.Transform(d_points, d_min_dis, calc_min_dis)
        center_ids[i] = trtc.Max_Element(d_min_dis)

    d_count = trtc.DVInt32(k)
    d_center_ids =  trtc.device_vector_from_list(center_ids, 'int32_t')

    # initialize group-average values
    d_group_aves_x =  trtc.device_vector("float", k)
    d_group_aves_y =  trtc.device_vector("float", k)
    d_group_aves = trtc.DVZipped([d_group_aves_x, d_group_aves_y], ['x','y'])

    trtc.Gather(d_center_ids, d_points, d_group_aves)

    # initialize labels
    d_labels =  trtc.device_vector("int32_t", n)
    trtc.Fill(d_labels, trtc.DVInt32(-1))

    # buffer for new-calculated lables
    d_labels_new =  trtc.device_vector("int32_t", n)

    d_labels_sink = trtc.DVDiscard("int32_t", k)
    d_group_sums = trtc.device_vector(d_points.name_elem_cls(), k)
    d_group_cumulate_counts = trtc.device_vector("int32_t", k)
    d_group_counts = trtc.device_vector("int32_t", k)

    d_counter = trtc.DVCounter(trtc.DVInt32(0), k)

    # iterations
    while True:
        # calculate new labels
        calc_new_labels = trtc.Functor({"aves": d_group_aves, "count": d_count }, ['pos'], 
'''
        float minDis = FLT_MAX;
        int label = -1;
        for (int i=0; i<count; i++)
        {
            float dis = (pos.x - aves[i].x)*(pos.x - aves[i].x);
            dis+= (pos.y - aves[i].y)*(pos.y - aves[i].y);
            if (dis<minDis) 
            {
                minDis = dis;
                label = i;
            }
        }
        return label;
''')
        trtc.Transform(d_points, d_labels_new, calc_new_labels)
        if trtc.Equal(d_labels, d_labels_new):
            break
        trtc.Copy(d_labels_new, d_labels)

        # recalculate group-average values
        trtc.Sort_By_Key(d_labels, d_points)
        trtc.Reduce_By_Key(d_labels, d_points, d_labels_sink, d_group_sums, trtc.EqualTo(), point_plus)
        trtc.Upper_Bound_V(d_labels, d_counter, d_group_cumulate_counts)
        trtc.Adjacent_Difference(d_group_cumulate_counts, d_group_counts)
        trtc.Transform_Binary(d_group_sums, d_group_counts, d_group_aves, point_div)

    h_x = d_x.to_host()
    h_y = d_y.to_host()
    h_labels = d_labels.to_host()
    h_group_aves_x = d_group_aves_x.to_host()
    h_group_aves_y = d_group_aves_y.to_host()
    h_group_counts = d_group_counts.to_host()

    lines = []

    for i in range(n):
        label = h_labels[i]
        lines.append([(h_x[i], h_y[i]), (h_group_aves_x[label], h_group_aves_y[label]) ] )

    lc = mc.LineCollection(lines)

    fig, ax = plt.subplots()
    ax.set_xlim((0, 1000))
    ax.set_ylim((0, 1000))

    ax.add_collection(lc)

    plt.show()



if __name__ == '__main__':

    h_x = np.random.rand(1000).astype(np.float32)*1000.0
    h_y = np.random.rand(1000).astype(np.float32)*1000.0

    d_x = trtc.device_vector_from_numpy(h_x)
    d_y = trtc.device_vector_from_numpy(h_y)

    demo_k_means(d_x, d_y, 50)
