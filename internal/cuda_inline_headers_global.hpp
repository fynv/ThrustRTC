/*
These header implementations are copied from the project "jitify".
https://github.com/NVIDIA/jitify
Thanks for the work.
*/

static const char s_safe_header_climits[] =
"#pragma once\n"
"\n"
"#if defined _WIN32 || defined _WIN64\n"
" #define __WORDSIZE 32\n"
"#else\n"
" #if defined __x86_64__ && !defined __ILP32__\n"
"  #define __WORDSIZE 64\n"
" #else\n"
"  #define __WORDSIZE 32\n"
" #endif\n"
"#endif\n"
"#define MB_LEN_MAX  16\n"
"#define CHAR_BIT    8\n"
"#define SCHAR_MIN   (-128)\n"
"#define SCHAR_MAX   127\n"
"#define UCHAR_MAX   255\n"
"#ifdef __CHAR_UNSIGNED__\n"
" #define CHAR_MIN   0\n"
" #define CHAR_MAX   UCHAR_MAX\n"
"#else\n"
" #define CHAR_MIN   SCHAR_MIN\n"
" #define CHAR_MAX   SCHAR_MAX\n"
"#endif\n"
"#define SHRT_MIN    (-32768)\n"
"#define SHRT_MAX    32767\n"
"#define USHRT_MAX   65535\n"
"#define INT_MIN     (-INT_MAX - 1)\n"
"#define INT_MAX     2147483647\n"
"#define UINT_MAX    4294967295U\n"
"#if __WORDSIZE == 64\n"
" # define LONG_MAX  9223372036854775807L\n"
"#else\n"
" # define LONG_MAX  2147483647L\n"
"#endif\n"
"#define LONG_MIN    (-LONG_MAX - 1L)\n"
"#if __WORDSIZE == 64\n"
" #define ULONG_MAX  18446744073709551615UL\n"
"#else\n"
" #define ULONG_MAX  4294967295UL\n"
"#endif\n"
"#define LLONG_MAX  9223372036854775807LL\n"
"#define LLONG_MIN  (-LLONG_MAX - 1LL)\n"
"#define ULLONG_MAX 18446744073709551615ULL\n";

static const char s_safe_header_cstdint[] =
"#pragma once\n"
"#include <climits>\n"
"typedef signed char      int8_t;\n"
"typedef signed short     int16_t;\n"
"typedef signed int       int32_t;\n"
"typedef signed long long int64_t;\n"
"typedef signed char      int_fast8_t;\n"
"typedef signed short     int_fast16_t;\n"
"typedef signed int       int_fast32_t;\n"
"typedef signed long long int_fast64_t;\n"
"typedef signed char      int_least8_t;\n"
"typedef signed short     int_least16_t;\n"
"typedef signed int       int_least32_t;\n"
"typedef signed long long int_least64_t;\n"
"typedef signed long long intmax_t;\n"
"typedef signed long      intptr_t; //optional\n"
"typedef unsigned char      uint8_t;\n"
"typedef unsigned short     uint16_t;\n"
"typedef unsigned int       uint32_t;\n"
"typedef unsigned long long uint64_t;\n"
"typedef unsigned char      uint_fast8_t;\n"
"typedef unsigned short     uint_fast16_t;\n"
"typedef unsigned int       uint_fast32_t;\n"
"typedef unsigned long long uint_fast64_t;\n"
"typedef unsigned char      uint_least8_t;\n"
"typedef unsigned short     uint_least16_t;\n"
"typedef unsigned int       uint_least32_t;\n"
"typedef unsigned long long uint_least64_t;\n"
"typedef unsigned long long uintmax_t;\n"
"typedef unsigned long      uintptr_t; //optional\n"
"#define INT8_MIN    SCHAR_MIN\n"
"#define INT16_MIN   SHRT_MIN\n"
"#define INT32_MIN   INT_MIN\n"
"#define INT64_MIN   LLONG_MIN\n"
"#define INT8_MAX    SCHAR_MAX\n"
"#define INT16_MAX   SHRT_MAX\n"
"#define INT32_MAX   INT_MAX\n"
"#define INT64_MAX   LLONG_MAX\n"
"#define UINT8_MAX   UCHAR_MAX\n"
"#define UINT16_MAX  USHRT_MAX\n"
"#define UINT32_MAX  UINT_MAX\n"
"#define UINT64_MAX  ULLONG_MAX\n"
"#define INTPTR_MIN  LONG_MIN\n"
"#define INTMAX_MIN  LLONG_MIN\n"
"#define INTPTR_MAX  LONG_MAX\n"
"#define INTMAX_MAX  LLONG_MAX\n"
"#define UINTPTR_MAX ULONG_MAX\n"
"#define UINTMAX_MAX ULLONG_MAX\n"
"#define PTRDIFF_MIN INTPTR_MIN\n"
"#define PTRDIFF_MAX INTPTR_MAX\n"
"#define SIZE_MAX    UINT64_MAX\n";

static const char s_safe_header_cfloat[] =
"#pragma once\n"
"\n"
"inline __host__ __device__ float  jitify_int_as_float(int i)             "
"{ union FloatInt { float f; int i; } fi; fi.i = i; return fi.f; }\n"
"inline __host__ __device__ double jitify_longlong_as_double(long long i) "
"{ union DoubleLongLong { double f; long long i; } fi; fi.i = i; return "
"fi.f; }\n"
"#define FLT_RADIX       2\n"
"#define FLT_MANT_DIG    24\n"
"#define DBL_MANT_DIG    53\n"
"#define FLT_DIG         6\n"
"#define DBL_DIG         15\n"
"#define FLT_MIN_EXP     -125\n"
"#define DBL_MIN_EXP     -1021\n"
"#define FLT_MIN_10_EXP  -37\n"
"#define DBL_MIN_10_EXP  -307\n"
"#define FLT_MAX_EXP     128\n"
"#define DBL_MAX_EXP     1024\n"
"#define FLT_MAX_10_EXP  38\n"
"#define DBL_MAX_10_EXP  308\n"
"#define FLT_MAX         jitify_int_as_float(2139095039)\n"
"#define DBL_MAX         jitify_longlong_as_double(9218868437227405311)\n"
"#define FLT_EPSILON     jitify_int_as_float(872415232)\n"
"#define DBL_EPSILON     jitify_longlong_as_double(4372995238176751616)\n"
"#define FLT_MIN         jitify_int_as_float(8388608)\n"
"#define DBL_MIN         jitify_longlong_as_double(4503599627370496)\n"
"#define FLT_ROUNDS      1\n"
"#if defined __cplusplus && __cplusplus >= 201103L\n"
"#define FLT_EVAL_METHOD 0\n"
"#define DECIMAL_DIG     21\n"
"#endif\n";

static const char s_safe_header_cuComplex_h[] =
R"(
typedef float2 cuFloatComplex;

__host__ __device__ static __inline__ float cuCrealf (cuFloatComplex x) 
{ 
    return x.x; 
}

__host__ __device__ static __inline__ float cuCimagf (cuFloatComplex x) 
{ 
    return x.y; 
}

__host__ __device__ static __inline__ cuFloatComplex make_cuFloatComplex 
                                                             (float r, float i)
{
    cuFloatComplex res;
    res.x = r;
    res.y = i;
    return res;
}

__host__ __device__ static __inline__ cuFloatComplex cuConjf (cuFloatComplex x)
{
    return make_cuFloatComplex (cuCrealf(x), -cuCimagf(x));
}
__host__ __device__ static __inline__ cuFloatComplex cuCaddf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
    return make_cuFloatComplex (cuCrealf(x) + cuCrealf(y), 
                                cuCimagf(x) + cuCimagf(y));
}

__host__ __device__ static __inline__ cuFloatComplex cuCsubf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
        return make_cuFloatComplex (cuCrealf(x) - cuCrealf(y), 
                                    cuCimagf(x) - cuCimagf(y));
}

__host__ __device__ static __inline__ cuFloatComplex cuCmulf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
    cuFloatComplex prod;
    prod = make_cuFloatComplex  ((cuCrealf(x) * cuCrealf(y)) - 
                                 (cuCimagf(x) * cuCimagf(y)),
                                 (cuCrealf(x) * cuCimagf(y)) + 
                                 (cuCimagf(x) * cuCrealf(y)));
    return prod;
}

__host__ __device__ static __inline__ cuFloatComplex cuCdivf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
    cuFloatComplex quot;
    float s = fabsf(cuCrealf(y)) + fabsf(cuCimagf(y));
    float oos = 1.0f / s;
    float ars = cuCrealf(x) * oos;
    float ais = cuCimagf(x) * oos;
    float brs = cuCrealf(y) * oos;
    float bis = cuCimagf(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    quot = make_cuFloatComplex (((ars * brs) + (ais * bis)) * oos,
                                ((ais * brs) - (ars * bis)) * oos);
    return quot;
}


__host__ __device__ static __inline__ float cuCabsf (cuFloatComplex x)
{
    float a = cuCrealf(x);
    float b = cuCimagf(x);
    float v, w, t;
    a = fabsf(a);
    b = fabsf(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0f + t * t;
    t = v * sqrtf(t);
    if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
        t = v + w;
    }
    return t;
}

typedef double2 cuDoubleComplex;

__host__ __device__ static __inline__ double cuCreal (cuDoubleComplex x) 
{ 
    return x.x; 
}

__host__ __device__ static __inline__ double cuCimag (cuDoubleComplex x) 
{ 
    return x.y; 
}

__host__ __device__ static __inline__ cuDoubleComplex make_cuDoubleComplex 
                                                           (double r, double i)
{
    cuDoubleComplex res;
    res.x = r;
    res.y = i;
    return res;
}

__host__ __device__ static __inline__ cuDoubleComplex cuConj(cuDoubleComplex x)
{
    return make_cuDoubleComplex (cuCreal(x), -cuCimag(x));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCadd(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    return make_cuDoubleComplex (cuCreal(x) + cuCreal(y), 
                                 cuCimag(x) + cuCimag(y));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCsub(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    return make_cuDoubleComplex (cuCreal(x) - cuCreal(y), 
                                 cuCimag(x) - cuCimag(y));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex ((cuCreal(x) * cuCreal(y)) - 
                                 (cuCimag(x) * cuCimag(y)),
                                 (cuCreal(x) * cuCimag(y)) + 
                                 (cuCimag(x) * cuCreal(y)));
    return prod;
}

__host__ __device__ static __inline__ cuDoubleComplex cuCdiv(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    cuDoubleComplex quot;
    double s = (fabs(cuCreal(y))) + (fabs(cuCimag(y)));
    double oos = 1.0 / s;
    double ars = cuCreal(x) * oos;
    double ais = cuCimag(x) * oos;
    double brs = cuCreal(y) * oos;
    double bis = cuCimag(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    quot = make_cuDoubleComplex (((ars * brs) + (ais * bis)) * oos,
                                 ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

__host__ __device__ static __inline__ double cuCabs (cuDoubleComplex x)
{
    double a = cuCreal(x);
    double b = cuCimag(x);
    double v, w, t;
    a = fabs(a);
    b = fabs(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0 + t * t;
    t = v * sqrt(t);
    if ((v == 0.0) || 
        (v > 1.79769313486231570e+308) || (w > 1.79769313486231570e+308)) {
        t = v + w;
    }
    return t;
}

typedef cuFloatComplex cuComplex;
__host__ __device__ static __inline__ cuComplex make_cuComplex (float x, 
                                                                float y) 
{ 
    return make_cuFloatComplex (x, y); 
}

__host__ __device__ static __inline__ cuDoubleComplex cuComplexFloatToDouble
                                                      (cuFloatComplex c)
{
    return make_cuDoubleComplex ((double)cuCrealf(c), (double)cuCimagf(c));
}

__host__ __device__ static __inline__ cuFloatComplex cuComplexDoubleToFloat
(cuDoubleComplex c)
{
	return make_cuFloatComplex ((float)cuCreal(c), (float)cuCimag(c));
}


__host__ __device__ static __inline__  cuComplex cuCfmaf( cuComplex x, cuComplex y, cuComplex d)
{
    float real_res;
    float imag_res;
    
    real_res = (cuCrealf(x) *  cuCrealf(y)) + cuCrealf(d);
    imag_res = (cuCrealf(x) *  cuCimagf(y)) + cuCimagf(d);
            
    real_res = -(cuCimagf(x) * cuCimagf(y))  + real_res;  
    imag_res =  (cuCimagf(x) *  cuCrealf(y)) + imag_res;          
     
    return make_cuComplex(real_res, imag_res);
}

__host__ __device__ static __inline__  cuDoubleComplex cuCfma( cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d)
{
    double real_res;
    double imag_res;
    
    real_res = (cuCreal(x) *  cuCreal(y)) + cuCreal(d);
    imag_res = (cuCreal(x) *  cuCimag(y)) + cuCimag(d);
            
    real_res = -(cuCimag(x) * cuCimag(y))  + real_res;  
    imag_res =  (cuCimag(x) *  cuCreal(y)) + imag_res;     
     
    return make_cuDoubleComplex(real_res, imag_res);
}
	
)";


static int s_num_headers_global = 4;
static const char* s_name_headers_global[] = {
	"climits",
	"cstdint",
	"cfloat",
	"cuComplex.h"
};

static const char* s_content_headers_global[] = {
	s_safe_header_climits,
	s_safe_header_cstdint,
	s_safe_header_cfloat,
	s_safe_header_cuComplex_h
};

