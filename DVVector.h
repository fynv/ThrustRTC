#ifndef _DVVector_h
#define _DVVector_h

#include "TRTC_api.h"
#include "DeviceViewable.h"
#include "TRTCContext.h"

class THRUST_RTC_API DVVectorLike : public DeviceViewable
{
public:
	const std::string& name_elem_cls() const { return m_elem_cls; }
	const std::string& name_ref_type() const { return m_ref_type; }
	size_t elem_size() const { return m_elem_size; }
	size_t size() const { return m_size; }

	DVVectorLike(const char* elem_cls, const char* ref_type, size_t size);
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

	DVVector(const char* elem_cls, size_t size, void* hdata=nullptr);
	~DVVector();
	virtual bool is_writable() const { return true; }

	void to_host(void* hdata, size_t begin=0, size_t end = (size_t)(-1)) const;
	virtual ViewBuf view() const;

private:
	void* m_data;
};

class THRUST_RTC_API DVVectorAdaptor : public DVVectorLike
{
public:
	void* data() const { return m_data; }

	DVVectorAdaptor(const char* elem_cls, size_t size, void* ddata);

	DVVectorAdaptor(const DVVector& vec, size_t begin = 0, size_t end = (size_t)(-1));
	DVVectorAdaptor(const DVVectorAdaptor& vec, size_t begin = 0, size_t end = (size_t)(-1));

	virtual bool is_writable() const { return true; }
	virtual ViewBuf view() const;

private:
	void* m_data;

};

#endif

