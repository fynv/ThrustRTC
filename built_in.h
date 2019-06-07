#ifndef _built_in_h
#define _built_in_h

////////// Elements ///////////
template<class T1, class T2>
struct Pair
{
	T1 first;
	T2 second;
};

////////// Functors ///////////
#ifdef DEVICE_ONLY
struct Identity
{
	template<typename T>
	__device__ inline T operator()(const T& x)
	{
		return x;
	}
};

struct Maximum
{
	template<typename T>
	__device__ inline T operator()(const T& x, const T& y)
	{
		return x < y ? y : x;
	}
};

struct Minimum
{
	template<typename T>
	__device__ inline T operator()(const T& x, const T& y)
	{
		return x > y ? y : x;
	}
};


struct EqualTo
{
	template<typename T>
	__device__ inline bool operator()(const T& x, const T& y)
	{
		return x == y;
	}
};

struct NotEqualTo
{
	template<typename T>
	__device__ inline bool operator()(const T& x, const T& y)
	{
		return x != y;
	}
};

struct Greater
{
	template<typename T>
	__device__ inline bool operator()(const T& x, const T& y)
	{
		return x > y;
	}
};

struct Less
{
	template<typename T>
	__device__ inline bool operator()(const T& x, const T& y)
	{
		return x < y;
	}
};

struct GreaterEqual
{
	template<typename T>
	__device__ inline bool operator()(const T& x, const T& y)
	{
		return x >= y;
	}
};

struct LessEqual
{
	template<typename T>
	__device__ inline bool operator()(const T& x, const T& y)
	{
		return x <= y;
	}
};

struct Plus
{
	template<typename T>
	__device__ inline T operator()(const T& x, const T& y)
	{
		return x + y;
	}
};

struct Minus
{
	template<typename T>
	__device__ inline T operator()(const T& x, const T& y)
	{
		return x - y;
	}
};


struct Multiplies
{
	template<typename T>
	__device__ inline T operator()(const T& x, const T& y)
	{
		return x * y;
	}
};


struct Divides
{
	template<typename T>
	__device__ inline T operator()(const T& x, const T& y)
	{
		return x / y;
	}
};

struct Modulus
{
	template<typename T>
	__device__ inline T operator()(const T& x, const T& y)
	{
		return x % y;
	}
};

struct Negate
{
	template<typename T>
	__device__ inline T operator()(const T& x)
	{
		return -x;
	}
};
#endif

////////// Vectors ///////////
template<class _T>
struct VectorView
{
	typedef _T value_t;
	typedef _T& ref_t;

	value_t* _data;
	size_t _size;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _size;
	}

	__device__ ref_t operator [](size_t idx)
	{
		return _data[idx];
	}
#endif
};

template<class _T>
struct ConstantView
{
	typedef _T value_t;
	typedef const _T& ref_t;

	size_t _size;
	value_t _value;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _size;
	}

	__device__ ref_t operator [](size_t)
	{
		return _value;
	}
#endif
};

template<class _T>
struct CounterView
{
	typedef _T value_t;
	typedef _T ref_t;
	size_t _size;
	value_t _value_init;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _size;
	}

	__device__ ref_t operator [](size_t idx)
	{
		return _value_init + (value_t)idx;
	}
#endif
};

template<class _T>
struct _Sink
{
#ifdef DEVICE_ONLY
	__device__ const _T& operator = (const _T& in)
	{
		return in;
	}
#endif
};

template<class _T>
struct DiscardView
{
	typedef _T value_t;
	typedef _Sink<_T>& ref_t;
	size_t _size;
	_Sink<_T> _sink;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _size;
	}

	__device__ ref_t operator [](size_t)
	{
		return _sink;
	}
#endif
};

template<class _TVVALUE, class _TVINDEX>
struct PermutationView
{
	typedef typename _TVVALUE::value_t value_t;
	typedef typename _TVVALUE::ref_t ref_t;
	_TVVALUE _view_vec_value;
	_TVINDEX _view_vec_index;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _view_vec_index.size();
	}

	__device__ ref_t operator [](size_t idx)
	{
		return _view_vec_value[_view_vec_index[idx]];
	}
#endif
};

template<class _TVVALUE>
struct ReverseView
{
	typedef typename _TVVALUE::value_t value_t;
	typedef typename _TVVALUE::ref_t ref_t;
	_TVVALUE _view_vec_value;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _view_vec_value.size();
	}

	__device__ value_t& operator [](size_t idx)
	{
		return _view_vec_value[size() - 1 - idx];
	}
#endif
};

template<class _T, class _T_VIN, class _T_OP>
struct TransformView
{
	typedef _T value_t;
	typedef _T ref_t;
	_T_VIN _view_vec_in;
	_T_OP _view_op;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _view_vec_in.size();
	}

	__device__ ref_t operator [](size_t idx)
	{
		return _view_op(_view_vec_in[idx]);
	}
#endif
};

////////// Binary-Search routines ///////////

#ifdef DEVICE_ONLY

template<class TVec, class TComp>
__device__ inline size_t d_lower_bound(TVec& vec, const typename TVec::value_t& value, TComp& comp, size_t begin = 0, size_t end = (size_t)(-1))
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end <= begin) return begin;
	if (comp(vec[end - 1], value)) return end;
	while (end > begin + 1)
	{
		size_t mid = begin + ((end - begin) >> 1);
		if (comp(vec[mid - 1], value))
			begin = mid;
		else
			end = mid;
	}
	return begin;
}

template<class TVec, class TComp>
__device__ inline size_t d_upper_bound(TVec& vec, const typename TVec::value_t& value, TComp& comp, size_t begin = 0, size_t end = (size_t)(-1))
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end <= begin) return begin;
	if (comp(value, vec[begin])) return begin;
	while (end > begin + 1)
	{
		size_t mid = begin + ((end - begin) >> 1);
		if (comp(value, vec[mid]))
			end = mid;
		else
			begin = mid;
	}
	return end;
}

template<class TVec, class TComp>
__device__ inline bool d_binary_search(TVec& vec, const typename TVec::value_t& value, TComp& comp, size_t begin = 0, size_t end = (size_t)(-1))
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end <= begin) return false;
	if (comp(value, vec[begin]) || comp(vec[end - 1], value)) return false;
	do
	{
		if (!comp(vec[begin], value) || !comp(value, vec[end - 1])) return true;
		size_t mid = begin + ((end - begin) >> 1);
		if (!comp(vec[mid - 1], value)) end = mid;
		else if (!comp(value, vec[mid])) begin = mid;
		else return false;		
	} while (end > begin + 1);
	return false;
}

template<class T, class TComp>
__device__ inline unsigned d_lower_bound_s(const T* arr, unsigned n, const T& value, TComp& comp)
{
	if (n <= 0) return 0;
	if (comp(arr[n - 1], value)) return n;
	unsigned begin = 0;
	while (n > begin + 1)
	{
		size_t mid = begin + ((n - begin) >> 1);
		if (comp(arr[mid - 1], value))
			begin = mid;
		else
			n = mid;
	}
	return begin;
}

template<class T, class TComp>
__device__ inline unsigned d_upper_bound_s(const T* arr, unsigned n, const T& value, TComp& comp)
{
	if (n <= 0) return 0;
	if (comp(value, arr[0])) return 0;
	unsigned begin = 0;
	while (n > begin + 1)
	{
		size_t mid = begin + ((n - begin) >> 1);
		if (comp(value, arr[mid]))
			n = mid;
		else
			begin = mid;
	}
	return n;
}

#endif

////////// Any->UINT converters, for radix sort ///////////
#ifdef DEVICE_ONLY

template<typename T>
__device__ inline uint32_t d_u32(T v)
{
	return (uint32_t)v;
}

template <>
__device__ inline uint32_t d_u32<float>(float v)
{
	uint32_t u = *(uint32_t*)(&v);
	if (u & 0x80000000)
		u = -(u & 0x7FFFFFFF);
	return u;
}

template<typename T>
__device__ inline uint64_t d_u64(T v)
{
	return (uint64_t)v;
}

template <>
__device__ inline uint64_t d_u64<double>(double v)
{
	uint64_t u = *(uint64_t*)(&v);
	if (u & 0x8000000000000000)
		u = -(u & 0x7FFFFFFFFFFFFFFF);
	return u;
}

#endif


#endif

