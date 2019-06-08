# ThrustRTC - Demo

## Thrust now available to Python!

This is not dreaming. With ThrustRTC, now you have almost all functions of Thrust directly available from Python!

Users are reluctant to use their GPUs sometimes, not because of that they don't like the high-performance, 
but because of that there's no easy way to write GPU programmes in a launguage other than C++. 
With Thrust, many tasks (like sorting, reduction) have become very easy to do. However, a C++ developing environment
plus CUDA SDK is still mandatory. Imagine that we have some data, and we want to do some simple statistical analysis,
we may think, well, this is just some simple work, let's just do it with Python, why bothering opening an IDE and
mess with the compilers etc..

Now with ThrustRTC, the story could change. The following 2 examples will show you how ThrustRTC can be used to do some
simple data anlaysis works using GPU. We could have done the same routines using Thrust in C++, but ThrustRTC makes it
happening in Python. You don't even need a full CUDA SDK to do these.

## Histogram

First, we pretend we have some data to analysis. We use *p.random.randn()* to generate some random numbers:

```python
h_data = np.random.randn(2000).astype(np.float32) * 30.0 +100.0
```

Second, we copy these data to GPU memory by creating a *DVector()* object:

```python
ctx = trtc.Context()
d_data = trtc.device_vector_from_numpy(ctx, h_data)
```

To build a histogram, we first sort the data, so elements of similar values are brought together:

```python
trtc.Sort(ctx, d_data)
```

The cumulative number of elements can be calculated by doing a binary-search of the upper-bounds of each bin of the histogram.
First, we can construct a Fake-Vector of the upper-bounds using a combination of *DVCounter* and *DVTransform*:

```python
d_counter = trtc.DVCounter(ctx, trtc.DVFloat(0.0), 21)
d_range_ends = trtc.DVTransform(ctx, d_counter, "float", trtc.Functor(ctx, {}, ['x'], '        return x*10.0;\n' ))
```

Now *d_range_ends* has 21 elements 0, 10, 20.. 200.

Calculate the cumulative histogram using binary-search:

```python
d_cumulative_histogram =  trtc.device_vector(ctx, "int32_t", 21)
trtc.Upper_Bound_V(ctx, d_data, d_range_ends, d_cumulative_histogram)
```

The final histogram we need can be calculated by doing a adjacent-difference to the cumulative histogram:

```python
d_histogram = trtc.device_vector(ctx, "int32_t", 21)
trtc.Adjacent_Difference(ctx, d_cumulative_histogram, d_histogram)
```

Finally, we copy the result to host-memory and plot it:

```python
h_histogram = d_histogram.to_host(1, 21)

x_axis = [str(x) for x in np.arange(5, 200, 10)]
positions = np.arange(len(x_axis))
plt.bar(positions, h_histogram, align='center', alpha=0.5)
plt.xticks(positions, x_axis)
plt.ylabel('Count')
plt.title('Histogram')

plt.show()
```

![Plot](histogram.png)


## K-Means



