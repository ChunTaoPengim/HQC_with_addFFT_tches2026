#include "stdint.h"
#include "string.h"

#include "ringmul.h"


static inline
uint16_t _mul_8( uint8_t a , uint8_t b )
{
    uint16_t r = 0;
    for( int i = 0 ; i < 8 ; i++ )
    {
        if( b & (1<<i) )
            r ^= ((uint16_t)a) << i;
    }
    return r;
}




/// @brief multiply two polynomials in F216[x]/ x^384
/// @param c0 [out] low  byte of F216. size : 384 bytes
/// @param c1 [out] high byte of F216. size : 384 bytes
/// @param a [in] size : 384 bytes
/// @param b [in] size : 384 bytes
void ringmul_mul_384( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b )
{
#define LEN (384)
    uint16_t t0[LEN] = {0};
    for(int i=0;i<LEN;i++)
    {
        for(int j=0;j<LEN;j++)
        {
            if( i+j >= LEN ) break;
            t0[i+j] ^= _mul_8( a[i] , b[j] );
        }
    }
    for(int i=0;i<LEN;i++)
    {
        c0[i] = t0[i] & 0xff;
        c1[i] = (t0[i] >> 8);
    }
#undef LEN
}

/// @brief multiply two polynomials in F216[x]/ x^1024
/// @param c0 [out] low  byte of F216. size : 1024 bytes
/// @param c1 [out] high byte of F216. size : 1024 bytes
/// @param a [in] size : 1024 bytes
/// @param b [in] size : 1024 bytes
void ringmul_mul_1024( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b )
{
#define LEN (1024)
    uint16_t t0[LEN] = {0};
    for(int i=0;i<LEN;i++)
    {
        for(int j=0;j<LEN;j++)
        {
            if( i+j >= LEN ) break;
            t0[i+j] ^= _mul_8( a[i] , b[j] );
        }
    }
    for(int i=0;i<LEN;i++)
    {
        c0[i] = t0[i] & 0xff;
        c1[i] = (t0[i] >> 8);
    }
#undef LEN
}



