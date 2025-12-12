#include "gf256.h"

void gf256v_add( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned len )
{
    for(unsigned i=0;i<len;i++) r[i] = a[i]^b[i];
}

void gf256v_mul( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned len )
{
    for(unsigned i=0;i<len;i++) r[i] = gf256_mul( a[i] , b[i] );
}

void gf256x2v_mul( uint8_t * c0 , uint8_t * c1 , 
    const uint8_t * a0 , const uint8_t * a1 , const uint8_t * b0 , const uint8_t * b1 , unsigned len )
{
    for(unsigned i=0;i<len;i++) {
        uint8_t r0 = gf256_mul( a0[i] , b0[i] );
        uint8_t r2 = gf256_mul( a1[i] , b1[i] );
        uint8_t r1 = gf256_mul( a0[i]^a1[i] , b0[i]^b1[i] )^r0;
        c0[i] = r0 ^ gf256_mul( r2 , 0x20 );
        c1[i] = r1;
    }
}

void gf256t4v_mul( uint8_t * r0 , uint8_t * r1 , uint8_t * r2 , uint8_t * r3 ,
    const uint8_t * a0 , const uint8_t * a1 , const uint8_t * a2 , const uint8_t * a3 ,
    const uint8_t * b0 , const uint8_t * b1 , const uint8_t * b2 , const uint8_t * b3 , unsigned len )
{
    for(unsigned i=0;i<len;i++) {
        uint16_t rr0 = gf256x2_mul( a0[i] , a1[i] , b0[i] , b1[i] );
        uint16_t rr2 = gf256x2_mul( a2[i] , a3[i] , b2[i] , b3[i] );
        uint16_t rr1 = gf256x2_mul( a0[i]^a2[i] , a1[i]^a3[i] , b0[i]^b2[i] , b1[i]^b3[i] )^rr0;

        uint8_t rr2lx0x20 = gf256_mul_0x20( rr2 & 0xff );
        uint8_t rr2hx0x20 = gf256_mul_0x20( rr2 >> 8 );
        uint8_t rr2hx0x20x0x20 = gf256_mul_0x20( rr2hx0x20 );
        r0[i] = (rr0&0xff) ^ rr2hx0x20x0x20;
        r1[i] = (rr0>>8)   ^ rr2lx0x20^rr2hx0x20;
        r2[i] = rr1 & 0xff;
        r3[i] = (rr1>>8);
    }
}


