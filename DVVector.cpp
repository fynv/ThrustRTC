#include <cuda.h>
#include "DVVector.h"


DVVectorLike::DVVectorLike(TRTCContext& ctx, const char* elem_cls, size_t size)
{
	m_elem_cls = elem_cls;
	m_elem_size = ctx.size_of(elem_cls);
	m_size = size;
}


DVVector::DVVector(TRTCContext& ctx, const char* elem_cls, size_t size, void* hdata)
	: DVVectorLike(ctx, elem_cls, size)
{
	CUdeviceptr dptr;
	cuMemAlloc(&dptr, m_elem_size*m_size);
	m_data = (void*)dptr;
	if (hdata)
		cuMemcpyHtoD(dptr, hdata, m_elem_size*m_size);
	else
		cuMemsetD8(dptr, 0, m_elem_size*m_size);
}

DVVector::~DVVector()
{
	cuMemFree((CUdeviceptr)m_data);
}

void DVVector::to_host(void* hdata)
{
	cuMemcpyDtoH(hdata, (CUdeviceptr)m_data, m_elem_size*m_size);
}

