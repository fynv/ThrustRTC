#ifndef _DVVector_h
#define _DVVector_h

template<class T>
struct DVVectorView
{
	T* data;
	size_t size;
};


#ifndef DEVICE_ONLY
#include "TRTC_api.h"
#include "DeviceViewable.h"
#include "TRTCContext.h"

class THRUST_RTC_API DVVector : public DeviceViewable
{
public:
	std::string name_elem_cls() const { return m_elem_cls; }
	size_t elem_size() const { return m_elem_size; }
	size_t size() const { return m_size; }	
	void* data() const { return m_data; }

	DVVector(TRTCContext& ctx, const char* elem_cls, int size);
	DVVector(TRTCContext& ctx, const char* elem_cls, void* hdata, int size);
	~DVVector();

	void ToHost(void* hdata);	
	
	virtual std::string name_view_cls() const
	{
		return std::string("DVVectorView<") + m_elem_cls + ">";
	}

	virtual ViewBuf view() const
	{
		ViewBuf buf(sizeof(DVVectorView<void>));
		DVVectorView<void> *pview = (DVVectorView<void>*)buf.data();
		pview->data = m_data;
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

