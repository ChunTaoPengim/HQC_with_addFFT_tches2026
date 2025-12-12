#ifndef _GF256V_GFNI_H_
#define _GF256V_GFNI_H_

#include <stdint.h>
#include "gf256.h"
#include "string.h"  // memcpy

#include <immintrin.h>


typedef struct {
    __m256i val[2];
} ymm_x2;

typedef struct {
    __m256i val[4];
} ymm_x4;



static inline
ymm_x2 _gf256x2v_mul( __m256i aa0 , __m256i aa1 , __m256i bb0 , __m256i bb1 , __m256i _0x20 )
{
    //__m256i aa0 = _mm256_loadu_si256((__m256i*)a0);
    //__m256i aa1 = _mm256_loadu_si256((__m256i*)a1);
    //__m256i bb0 = _mm256_loadu_si256((__m256i*)b0);
    //__m256i bb1 = _mm256_loadu_si256((__m256i*)b1);

    //__m256i _0x20 = _mm256_set1_epi8(0x20);
    __m256i rr2 = _mm256_gf2p8mul_epi8(aa1, bb1);
    __m256i rr0 = _mm256_gf2p8mul_epi8(aa0, bb0);
    __m256i rr1 = _mm256_gf2p8mul_epi8(aa0^aa1, bb0^bb1)^rr0;
    rr0 ^= _mm256_gf2p8mul_epi8(rr2, _0x20);
    //_mm256_storeu_si256((__m256i*)c1, rr1);
    //_mm256_storeu_si256((__m256i*)c0, rr0);
    ymm_x2 r;
    r.val[0] = rr0;
    r.val[1] = rr1;
    return r;
}


static inline
ymm_x4 _gf256t4v_mul( __m256i aa0 , __m256i aa1 , __m256i aa2 , __m256i aa3 ,
    __m256i bb0 , __m256i bb1 , __m256i bb2 , __m256i bb3 , __m256i _0x20 )
{
    __m256i c02 = _mm256_gf2p8mul_epi8(aa1, bb1);
    __m256i c00 = _mm256_gf2p8mul_epi8(aa0, bb0);
    __m256i c01 = _mm256_gf2p8mul_epi8(aa0^aa1, bb0^bb1)^c00;
    c00 ^= _mm256_gf2p8mul_epi8(c02, _0x20);

    __m256i c22 = _mm256_gf2p8mul_epi8(aa3, bb3);
    __m256i c20 = _mm256_gf2p8mul_epi8(aa2, bb2);
    __m256i c21 = _mm256_gf2p8mul_epi8(aa2^aa3, bb2^bb3)^c20;
    c20 ^= _mm256_gf2p8mul_epi8(c22, _0x20);

    __m256i ta0 = aa0 ^ aa2;
    __m256i ta1 = aa1 ^ aa3;
    __m256i tb0 = bb0 ^ bb2;
    __m256i tb1 = bb1 ^ bb3;
    __m256i c12 = _mm256_gf2p8mul_epi8(ta1, tb1);
    __m256i c10 = _mm256_gf2p8mul_epi8(ta0, tb0);
    __m256i c11 = _mm256_gf2p8mul_epi8(ta0^ta1, tb0^tb1)^c10;
    c10 ^= _mm256_gf2p8mul_epi8(c12, _0x20);

    c10 ^= c00;
    c11 ^= c01;
    // reduce (c21,c20) x 0x2000
    c02 = _mm256_gf2p8mul_epi8(c21, _0x20);
    c01 ^= _mm256_gf2p8mul_epi8(c20, _0x20)^c02;
    c00 ^= _mm256_gf2p8mul_epi8(c02, _0x20);

    ymm_x4 r;
    r.val[0] = c00;
    r.val[1] = c01;
    r.val[2] = c10;
    r.val[3] = c11;
    return r;
}

static inline
ymm_x4 _gf256t4v_mul_gf256( __m256i aa0 , __m256i aa1 , __m256i aa2 , __m256i aa3 , __m256i bb0 )
{
    ymm_x4 r;
    r.val[0] = _mm256_gf2p8mul_epi8(aa0, bb0);
    r.val[1] = _mm256_gf2p8mul_epi8(aa1, bb0);
    r.val[2] = _mm256_gf2p8mul_epi8(aa2, bb0);
    r.val[3] = _mm256_gf2p8mul_epi8(aa3, bb0);
    return r;
}

static inline
ymm_x4 _gf256t4v_mul_gf256x2( __m256i aa0 , __m256i aa1 , __m256i aa2 , __m256i aa3 , __m256i bb0 , __m256i bb1 , __m256i _0x20 )
{
    __m256i c02 = _mm256_gf2p8mul_epi8(aa1, bb1);
    __m256i c00 = _mm256_gf2p8mul_epi8(aa0, bb0);
    __m256i c01 = _mm256_gf2p8mul_epi8(aa0^aa1, bb0^bb1)^c00;
    c00 ^= _mm256_gf2p8mul_epi8(c02, _0x20);

    __m256i c12 = _mm256_gf2p8mul_epi8(aa3, bb1);
    __m256i c10 = _mm256_gf2p8mul_epi8(aa2, bb0);
    __m256i c11 = _mm256_gf2p8mul_epi8(aa2^aa3, bb0^bb1)^c10;
    c10 ^= _mm256_gf2p8mul_epi8(c12, _0x20);

    ymm_x4 r;
    r.val[0] = c00;
    r.val[1] = c01;
    r.val[2] = c10;
    r.val[3] = c11;
    return r;
}

static inline
ymm_x4 _gf256t4v_mul_gf256x3( __m256i aa0 , __m256i aa1 , __m256i aa2 , __m256i aa3 , __m256i bb0 , __m256i bb1 , __m256i bb2 , __m256i _0x20 )
{
    __m256i c02 = _mm256_gf2p8mul_epi8(aa1, bb1);
    __m256i c00 = _mm256_gf2p8mul_epi8(aa0, bb0);
    __m256i c01 = _mm256_gf2p8mul_epi8(aa0^aa1, bb0^bb1)^c00;
    c00 ^= _mm256_gf2p8mul_epi8(c02, _0x20);

    __m256i c20 = _mm256_gf2p8mul_epi8(aa2, bb2);
    __m256i c21 = _mm256_gf2p8mul_epi8(aa3, bb2);

    __m256i ta0 = aa0 ^ aa2;
    __m256i ta1 = aa1 ^ aa3;
    __m256i tb0 = bb0 ^ bb2;
    __m256i tb1 = bb1;
    __m256i c12 = _mm256_gf2p8mul_epi8(ta1, tb1);
    __m256i c10 = _mm256_gf2p8mul_epi8(ta0, tb0);
    __m256i c11 = _mm256_gf2p8mul_epi8(ta0^ta1, tb0^tb1)^c10;
    c10 ^= _mm256_gf2p8mul_epi8(c12, _0x20);

    c10 ^= c00;
    c11 ^= c01;
    // reduce (c21,c20) x 0x2000
    c02 = _mm256_gf2p8mul_epi8(c21, _0x20);
    c01 ^= _mm256_gf2p8mul_epi8(c20, _0x20)^c02;
    c00 ^= _mm256_gf2p8mul_epi8(c02, _0x20);

    ymm_x4 r;
    r.val[0] = c00;
    r.val[1] = c01;
    r.val[2] = c10;
    r.val[3] = c11;
    return r;
}


#endif

