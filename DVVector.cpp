#include <cuda_runtime.h>
#include "DVVector.h"

DVVector::DVVector(TRTCContext& ctx, const char* elem_cls, int size)
{
	m_elem_cls = elem_cls;
	m_elem_size = ctx.size_of(elem_cls);
	m_size = size;
	cudaMalloc(&m_data, m_elem_size*m_size);
}

DVVector::DVVector(TRTCContext& ctx, const char* elem_cls, void* hdata, int size)
{
	m_elem_cls = elem_cls;
	m_elem_size = ctx.size_of(elem_cls);
	m_size = size;
	cudaMalloc(&m_data, m_elem_size*m_size);
	cudaMemcpy(m_data, hdata, m_elem_size*m_size, cudaMemcpyHostToDevice);
}

DVVector::~DVVector()
{
	cudaFree(m_data);
}

void DVVector::ToHost(void* hdata)
{
	cudaMemcpy(hdata, m_data, m_elem_size*m_size, cudaMemcpyDeviceToHost);
}

