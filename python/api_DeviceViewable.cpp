#include "api.h"
#include "DeviceViewable.h"

const char* n_dv_name_view_cls(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return dv->name_view_cls().c_str();
}

void n_dv_destroy(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	delete dv;
}

void* n_dvint8_create(int v)
{
	return new DVInt8((int8_t)v);
}

int n_dvint8_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return (int)*(int8_t*)dv->view().data();
}

void* n_dvuint8_create(unsigned v)
{
	return new DVUInt8((uint8_t)v);
}

unsigned n_dvuint8_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return (unsigned)*(uint8_t*)dv->view().data();
}

void* n_dvint16_create(int v)
{
	return new DVInt16((int16_t)v);
}

int n_dvint16_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return (int)*(int16_t*)dv->view().data();
}

void* n_dvuint16_create(unsigned v)
{
	return new DVUInt16((uint16_t)v);
}

unsigned n_dvuint16_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return (unsigned)*(uint16_t*)dv->view().data();
}

void* n_dvint32_create(int v)
{
	return new DVInt32(v);
}

int n_dvint32_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(int32_t*)dv->view().data();
}

void* n_dvuint32_create(unsigned v)
{
	return new DVUInt32(v);
}

unsigned n_dvuint32_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(uint32_t*)dv->view().data();
}

void* n_dvint64_create(long long v)
{
	return new DVInt64(v);
}

long long n_dvint64_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(int64_t*)dv->view().data();
}

void* n_dvuint64_create(unsigned long long v)
{
	return new DVUInt64(v);
}

unsigned long long n_dvuint64_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(uint64_t*)dv->view().data();
}

void* n_dvfloat_create(float v)
{
	return new DVFloat(v);
}

float n_dvfloat_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(float*)dv->view().data();
}

void* n_dvdouble_create(double v)
{
	return new DVDouble(v);
}

double n_dvdouble_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(double*)dv->view().data();
}

void* n_dvbool_create(int v)
{
	return new DVBool(v!=0);
}

int n_dvbool_value(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(bool*)dv->view().data()?1:0;
}

void* n_dvcomplex64_create(float real, float imag)
{
	return new DVComplex64(real, imag);
}

float n_dvcomplex64_real(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(float*)&dv->view()[0];
}

float n_dvcomplex64_imag(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(float*)&dv->view()[sizeof(float)];
}

void* n_dvcomplex128_create(double real, double imag)
{
	return new DVComplex128(real, imag);
}

double n_dvcomplex128_real(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(double*)&dv->view()[0];
}

double n_dvcomplex128_imag(void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	return *(double*)&dv->view()[sizeof(double)];
}
