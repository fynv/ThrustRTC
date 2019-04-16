#ifndef _DVVector_h
#define _DVVector_h

#ifndef DEVICE_ONLY
#include <host_defines.h>
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
	__device__ value_t& operator [](size_t idx)
	{
		return data[idx];
	}
};


#ifndef DEVICE_ONLY

class THRUST_RTC_API DVVector : public DeviceViewable
{
public:
	std::string name_elem_cls() const { return m_elem_cls; }
	size_t elem_size() const { return m_elem_size; }
	size_t size() const { return m_size; }	
	void* data() const { return m_data; }

	DVVector(TRTCContext& ctx, const char* elem_cls, size_t size, void* hdata=nullptr);
	~DVVector();

	void to_host(void* hdata);	
	
	virtual std::string name_view_cls() const
	{
		return std::string("VectorView<") + m_elem_cls + ">";
	}

	virtual ViewBuf view() const
	{
		ViewBuf buf(sizeof(VectorView<char>));
		VectorView<char> *pview = (VectorView<char>*)buf.data();
		pview->data = (char*) m_data;
		pview->size = m_size;
		return buf;
	}

private:
	std::string m_elem_cls;
	size_t m_elem_size;
	size_t m_size;
	void* m_data;
};
#endif

#endif

