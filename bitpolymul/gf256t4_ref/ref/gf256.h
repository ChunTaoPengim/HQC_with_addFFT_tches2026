/// @file gf256.h
/// @brief Library for GF(256) arithmetics
///

#ifndef _GF256_H_
#define _GF256_H_

#include <stdint.h>

///////////////////////////////////////////
//
//  Arithmetics for one field element
//
//////////////////////////////////////////

static inline uint8_t gf256_is_nonzero(uint8_t a) {
    unsigned a8 = a;
    unsigned r = ((unsigned) 0) - a8;
    r >>= 8;
    return r & 1;
}

// gf256 := gf2[X]/ (x^8+x^4+x^3+x+1)   // 0x11b , AES field
static inline uint8_t gf256_mul(uint8_t a, uint8_t b) {
    uint8_t r = a * (b & 1);

    a = (a << 1) ^ ((a >> 7) * 0x1b);
    r ^= a * ((b >> 1) & 1);
    a = (a << 1) ^ ((a >> 7) * 0x1b);
    r ^= a * ((b >> 2) & 1);
    a = (a << 1) ^ ((a >> 7) * 0x1b);
    r ^= a * ((b >> 3) & 1);
    a = (a << 1) ^ ((a >> 7) * 0x1b);
    r ^= a * ((b >> 4) & 1);
    a = (a << 1) ^ ((a >> 7) * 0x1b);
    r ^= a * ((b >> 5) & 1);
    a = (a << 1) ^ ((a >> 7) * 0x1b);
    r ^= a * ((b >> 6) & 1);
    a = (a << 1) ^ ((a >> 7) * 0x1b);
    r ^= a * ((b >> 7) & 1);
    return r;
}

static inline uint8_t gf256_mul_0x20(uint8_t a) {
    a = (a << 1) ^ ((a >> 7) * 0x1b); // x2
    a = (a << 1) ^ ((a >> 7) * 0x1b); // x4
    a = (a << 1) ^ ((a >> 7) * 0x1b); // x8
    a = (a << 1) ^ ((a >> 7) * 0x1b); // x10
    a = (a << 1) ^ ((a >> 7) * 0x1b); // x20
    return a;
}


// gf256x2 := gf256[X]/ (X^2+X+0x20)
static inline uint16_t gf256x2_mul(uint8_t al, uint8_t ah, uint8_t bl , uint8_t bh) {
    uint8_t r0 = gf256_mul( al , bl );
    uint8_t r2 = gf256_mul( ah , bh );
    uint8_t r1 = gf256_mul( al^ah , bl^bh )^r0;
    r0 ^= gf256_mul_0x20( r2 );
    return r0 ^ (((uint16_t)r1)<<8);
}

/// gf256t4 :=  gf256x2 [Y] / ( Y^2 + Y + 0x2000 ) /// 0x1,0001,2000
static inline uint32_t gf256t4_mul(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    uint16_t r0 = gf256x2_mul( a0 , a1 , b0 , b1 );
    uint16_t r2 = gf256x2_mul( a2 , a3 , b2 , b3 );
    uint16_t r1 = gf256x2_mul( a0^a2 , a1^a3 , b0^b2 , b1^b3 )^r0;
    uint8_t r2lx0x20 = gf256_mul_0x20( r2 & 0xff );
    uint8_t r2hx0x20 = gf256_mul_0x20( r2 >> 8 );
    uint8_t r2hx0x20x0x20 = gf256_mul_0x20( r2hx0x20 );
    r0 ^= ((uint16_t)(r2lx0x20^r2hx0x20))<<8;
    r0 ^= r2hx0x20x0x20;
    return r0 ^ (((uint32_t)r1)<<16);
}


////////////////////////////////////////
//
//  library 32 bit vectors
//
////////////////////////////////////////

// gf256 := gf2[X]/ (x^8+x^4+x^3+x+1)   // 0x11b , AES field

static inline uint32_t gf256v_mul_u32(uint32_t a, uint8_t b) {
    uint32_t a_msb;
    uint32_t a32 = a;
    uint32_t b32 = b;
    uint32_t r32 = a32 * (b32 & 1);

    a_msb = a32 & 0x80808080; // MSB, 7th bits
    a32 ^= a_msb;   // clear MSB
    a32 = (a32 << 1) ^ ((a_msb >> 7) * 0x1b);
    r32 ^= (a32) * ((b32 >> 1) & 1);

    a_msb = a32 & 0x80808080; // MSB, 7th bits
    a32 ^= a_msb;   // clear MSB
    a32 = (a32 << 1) ^ ((a_msb >> 7) * 0x1b);
    r32 ^= (a32) * ((b32 >> 2) & 1);

    a_msb = a32 & 0x80808080; // MSB, 7th bits
    a32 ^= a_msb;   // clear MSB
    a32 = (a32 << 1) ^ ((a_msb >> 7) * 0x1b);
    r32 ^= (a32) * ((b32 >> 3) & 1);

    a_msb = a32 & 0x80808080; // MSB, 7th bits
    a32 ^= a_msb;   // clear MSB
    a32 = (a32 << 1) ^ ((a_msb >> 7) * 0x1b);
    r32 ^= (a32) * ((b32 >> 4) & 1);

    a_msb = a32 & 0x80808080; // MSB, 7th bits
    a32 ^= a_msb;   // clear MSB
    a32 = (a32 << 1) ^ ((a_msb >> 7) * 0x1b);
    r32 ^= (a32) * ((b32 >> 5) & 1);

    a_msb = a32 & 0x80808080; // MSB, 7th bits
    a32 ^= a_msb;   // clear MSB
    a32 = (a32 << 1) ^ ((a_msb >> 7) * 0x1b);
    r32 ^= (a32) * ((b32 >> 6) & 1);

    a_msb = a32 & 0x80808080; // MSB, 7th bits
    a32 ^= a_msb;   // clear MSB
    a32 = (a32 << 1) ^ ((a_msb >> 7) * 0x1b);
    r32 ^= (a32) * ((b32 >> 7) & 1);

    return r32;
}

// vec_r = vec_a + vec_b
void gf256v_add( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned len );

// r =  [ a[i]*b[i] for i in range(l) ]
void gf256v_mul( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned l );

void gf256x2v_mul( uint8_t * r0 , uint8_t * r1 , 
    const uint8_t * a0 , const uint8_t * a1 , const uint8_t * b0 , const uint8_t * b1 , unsigned len );

void gf256t4v_mul( uint8_t * r0 , uint8_t * r1 , uint8_t * r2 , uint8_t * r3 ,
    const uint8_t * a0 , const uint8_t * a1 , const uint8_t * a2 , const uint8_t * a3 ,
    const uint8_t * b0 , const uint8_t * b1 , const uint8_t * b2 , const uint8_t * b3 , unsigned len );

#endif // _GF256_H_

