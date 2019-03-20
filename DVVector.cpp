#include <cuda_runtime.h>
#include "DVVector.h"

DVVector::DVVector(TRTCContext& ctx, const char* elem_cls, size_t size, void* hdata)
{
	m_elem_cls = elem_cls;
	m_elem_size = ctx.size_of(elem_cls);
	m_size = size;
	cudaMalloc(&m_data, m_elem_size*m_size);
	if (hdata)
		cudaMemcpy(m_data, hdata, m_elem_size*m_size, cudaMemcpyHostToDevice);
	else
		cudaMemset(m_data, 0, m_elem_size*m_size);
}

DVVector::~DVVector()
{
	cudaFree(m_data);
}

void DVVector::to_host(void* hdata)
{
	cudaMemcpy(hdata, m_data, m_elem_size*m_size, cudaMemcpyDeviceToHost);
}

