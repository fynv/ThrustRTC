#ifndef _DVVector_h
#define _DVVector_h

#include "TRTC_api.h"
#include "DeviceViewable.h"
#include "TRTCContext.h"

class THRUST_RTC_API DVVectorLike : public DeviceViewable
{
public:
	std::string name_elem_cls() const { return m_elem_cls; }
	std::string name_ref_type() const { return m_ref_type; }
	size_t elem_size() const { return m_elem_size; }
	size_t size() const { return m_size; }

	DVVectorLike(TRTCContext& ctx, const char* elem_cls, const char* ref_type, size_t size);
	virtual ~DVVectorLike() {}
	virtual bool is_readable() const { return true; }
	virtual bool is_writable() const { return false; }

protected:
	std::string m_elem_cls;
	std::string m_ref_type;
	size_t m_elem_size;
	size_t m_size;
};

class THRUST_RTC_API DVVector : public DVVectorLike
{
public:
	void* data() const { return m_data; }

	DVVector(TRTCContext& ctx, const char* elem_cls, size_t size, void* hdata=nullptr);
	~DVVector();
	virtual bool is_writable() const { return true; }

	void to_host(void* hdata, size_t begin=0, size_t end = (size_t)(-1));
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	void* m_data;
};

class THRUST_RTC_API DVVectorAdaptor : public DVVectorLike
{
public:
	DVVectorAdaptor(TRTCContext& ctx, const char* elem_cls, size_t size, void* ddata);
	virtual bool is_writable() const { return true; }

	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	void* m_data;

};

#endif

