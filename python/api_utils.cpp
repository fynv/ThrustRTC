#include "api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include <stdio.h>
#include <string>
#include <vector>

typedef std::vector<std::string> StrArray;
typedef std::vector<const DeviceViewable*> PtrArray;

void* n_string_array_create(unsigned long long size, const char* const* strs)
{
	StrArray* ret = new StrArray(size);
	for (size_t i = 0; i < size; i++)
		(*ret)[i] = strs[i];

	return ret;
}

void n_string_array_destroy(void* ptr_arr)
{
	StrArray* arr = (StrArray*)ptr_arr;
	delete arr;
}

void* n_pointer_array_create(unsigned long long size, const void* const* ptrs)
{
	PtrArray* ret = new PtrArray(size);
	memcpy(ret->data(), ptrs, sizeof(void*)*size);
	return ret;
}

void n_pointer_array_destroy(void* ptr_arr)
{
	PtrArray* arr = (PtrArray*)ptr_arr;
	delete arr;
}

void* n_dim3_create(unsigned x, unsigned y, unsigned z)
{
	dim_type* ret = new dim_type({ x,y,z });
	return ret;
}

void n_dim3_destroy(void* cptr)
{
	dim_type* v = (dim_type*)cptr;
	delete v;
}

