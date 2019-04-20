#ifndef _DVVector_h
#define _DVVector_h

#ifndef DEVICE_ONLY
#include "TRTC_api.h"
#include "DeviceViewable.h"
#include "TRTCContext.h"
#endif

template<class _T>
struct VectorView
{
	typedef _T value_t;

	value_t* data;
	size_t size;

#ifdef DEVICE_ONLY
	__device__ value_t& operator [](size_t idx)
	{
		return data[idx];
	}
#endif
};


#ifndef DEVICE_ONLY

class THRUST_RTC_API DVVectorLike : public DeviceViewable
{
public:
	std::string name_elem_cls() const { return m_elem_cls; }
	size_t elem_size() const { return m_elem_size; }
	size_t size() const { return m_size; }

	DVVectorLike(TRTCContext& ctx, const char* elem_cls, size_t size);
	virtual ~DVVectorLike() {}

protected:
	std::string m_elem_cls;
	size_t m_elem_size;
	size_t m_size;
};

class THRUST_RTC_API DVVector : public DVVectorLike
{
public:
	void* data() const { return m_data; }

	DVVector(TRTCContext& ctx, const char* elem_cls, size_t size, void* hdata=nullptr);
	~DVVector();

	void to_host(void* hdata);		
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	void* m_data;
};
#endif

#endif

