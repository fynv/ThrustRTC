#include "cuda_wrapper.h"
#include "DVVector.h"
#include "built_in.h"

inline bool CheckCUresult(CUresult res, const char* name_call)
{
	if (res != CUDA_SUCCESS)
	{
		printf("%s failed with Error code: %u\n", name_call, res);
		const char *name = nullptr;
		const char *desc = nullptr;
		cuGetErrorName(res, &name);
		cuGetErrorString(res, &desc);
		if (name != nullptr)
		{
			printf("Error Name: %s\n", name);
		}
		if (desc != nullptr)
		{
			printf("Error Description: %s\n", desc);
		}
		return false;
	}
	return true;
}

DVVectorLike::DVVectorLike(const char* elem_cls, const char* ref_type, size_t size)
{
	m_elem_cls = elem_cls;
	m_ref_type = ref_type;
	m_elem_size = TRTC_Size_Of(elem_cls);
	m_size = size;
}

DVVector::DVVector(const char* elem_cls, size_t size, void* hdata)
	: DVVectorLike(elem_cls, (std::string(elem_cls)+"&").c_str(), size)
{
	TRTC_Try_Init();
	
	CUdeviceptr dptr;
	if (!CheckCUresult(cuMemAlloc(&dptr, m_elem_size*m_size), "cuMemAlloc()")) return;
	m_data = (void*)dptr;
	if (hdata)
	{
		if (!CheckCUresult(cuMemcpyHtoD(dptr, hdata, m_elem_size*m_size), "cuMemcpyHtoD()")) return;
	}
	else
	{
		if (!CheckCUresult(cuMemsetD8(dptr, 0, m_elem_size*m_size), "cuMemsetD8()")) return;
	}

	m_name_view_cls = std::string("VectorView<") + m_elem_cls + ">";
}

DVVector::~DVVector()
{
	cuMemFree((CUdeviceptr)m_data);
}

void DVVector::to_host(void* hdata, size_t begin, size_t end) const
{
	if (end == (size_t)(-1) || end > m_size) end = m_size;
	size_t n = end - begin;
	if (!CheckCUresult(cuMemcpyDtoH(hdata, (CUdeviceptr)((char*)m_data + begin* m_elem_size), m_elem_size*n), "cuMemcpyDtoH()")) return;
}

ViewBuf DVVector::view() const
{
	ViewBuf buf(sizeof(VectorView<char>));
	VectorView<char> *pview = (VectorView<char>*)buf.data();
	pview->_data = (char*)m_data;
	pview->_size = m_size;
	return buf;
}

DVVectorAdaptor::DVVectorAdaptor(const char* elem_cls, size_t size, void* ddata)
	: DVVectorLike(elem_cls, (std::string(elem_cls) + "&").c_str(), size), m_data(ddata)
{
	m_name_view_cls = std::string("VectorView<") + m_elem_cls + ">";
}

DVVectorAdaptor::DVVectorAdaptor(const DVVector& vec, size_t begin, size_t end)
	: DVVectorLike(vec.name_elem_cls().c_str(), vec.name_ref_type().c_str(), (end == (size_t)(-1)? vec.size() : end) - begin)
{
	m_data = (char*)vec.data() + begin * vec.elem_size();
	m_name_view_cls = std::string("VectorView<") + m_elem_cls + ">";
}

DVVectorAdaptor::DVVectorAdaptor(const DVVectorAdaptor& vec, size_t begin, size_t end)
	: DVVectorLike(vec.name_elem_cls().c_str(), vec.name_ref_type().c_str(), (end == (size_t)(-1) ? vec.size() : end) - begin)
{
	m_data = (char*)vec.data() + begin * vec.elem_size();
	m_name_view_cls = std::string("VectorView<") + m_elem_cls + ">";
}


ViewBuf DVVectorAdaptor::view() const
{
	ViewBuf buf(sizeof(VectorView<char>));
	VectorView<char> *pview = (VectorView<char>*)buf.data();
	pview->_data = (char*)m_data;
	pview->_size = m_size;
	return buf;
}
