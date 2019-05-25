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

#endif

