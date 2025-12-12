#include "gf256.h"
#include "string.h"  // memcpy

#include "gf256v_gfni.h"
#include <immintrin.h>

void gf256v_add( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned len )
{
    unsigned rem = len & 31;
    if (rem) {
        if (len < 32 ) {
            uint8_t temp[32] __attribute__((aligned(32)));
            memcpy(temp, a, rem);
            __m256i aa = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b, rem);
            __m256i bb = _mm256_load_si256((__m256i*)temp);
            _mm256_store_si256((__m256i*)temp, aa^bb );
            memcpy(r, temp, rem);
            return;
        } else {
            __m256i aa = _mm256_loadu_si256((__m256i*)a);
            __m256i bb = _mm256_loadu_si256((__m256i*)b);
            __m256i r1 = _mm256_loadu_si256((__m256i*)(r+rem));
            _mm256_storeu_si256((__m256i*)r, aa^bb);
            _mm256_storeu_si256((__m256i*)(r+rem),r1);
        }
        r += rem;
        a += rem;
        b += rem;
        len -= rem;
    }
    while( len ) {
        __m256i aa = _mm256_loadu_si256((__m256i*)a);
        __m256i bb = _mm256_loadu_si256((__m256i*)b);
        _mm256_storeu_si256((__m256i*)r, aa^bb);
        r += 32;
        a += 32;
        b += 32;
        len -= 32;
    }
}


void gf256v_mul( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned len )
{
    unsigned rem = len & 31;
    if (rem) {
        if (len < 32 ) {
            uint8_t temp[32] __attribute__((aligned(32)));
            memcpy(temp, a, rem);
            __m256i aa = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b, rem);
            __m256i bb = _mm256_load_si256((__m256i*)temp);
            __m256i rr = _mm256_gf2p8mul_epi8(aa, bb);
            _mm256_store_si256((__m256i*)temp, rr);
            memcpy(r, temp, rem);
            return;
        } else {
            __m256i aa = _mm256_loadu_si256((__m256i*)a);
            __m256i bb = _mm256_loadu_si256((__m256i*)b);
            __m256i rr = _mm256_gf2p8mul_epi8(aa, bb);
            _mm256_storeu_si256((__m256i*)r, rr);
        }
        r += rem;
        a += rem;
        b += rem;
        len -= rem;
    }
    while( len ) {
        __m256i aa = _mm256_loadu_si256((__m256i*)a);
        __m256i bb = _mm256_loadu_si256((__m256i*)b);
        __m256i rr = _mm256_gf2p8mul_epi8(aa, bb);
        _mm256_storeu_si256((__m256i*)r, rr);
        r += 32;
        a += 32;
        b += 32;
        len -= 32;
    }
}

void gf256x2v_mul( uint8_t * c0 , uint8_t * c1 , 
    const uint8_t * a0 , const uint8_t * a1 , const uint8_t * b0 , const uint8_t * b1 , unsigned len )
{
    __m256i _0x20 = _mm256_set1_epi8(0x20);
    unsigned rem = len & 31;
    if (rem) {
        if (len < 32 ) {
            uint8_t temp[32] __attribute__((aligned(32)));
            memcpy(temp, a0, rem);
            __m256i aa0 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, a1, rem);
            __m256i aa1 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b0, rem);
            __m256i bb0 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b1, rem);
            __m256i bb1 = _mm256_load_si256((__m256i*)temp);

            __m256i rr2 = _mm256_gf2p8mul_epi8(aa1, bb1);
            __m256i rr0 = _mm256_gf2p8mul_epi8(aa0, bb0);
            __m256i rr1 = _mm256_gf2p8mul_epi8(aa0^aa1, bb0^bb1)^rr0;
            rr0 ^= _mm256_gf2p8mul_epi8(rr2, _0x20);
            _mm256_store_si256((__m256i*)temp, rr1);
            memcpy(c1, temp, rem);
            _mm256_store_si256((__m256i*)temp, rr0);
            memcpy(c0, temp, rem);
            return;
        } else {
            __m256i aa0 = _mm256_loadu_si256((__m256i*)a0);
            __m256i aa1 = _mm256_loadu_si256((__m256i*)a1);
            __m256i bb0 = _mm256_loadu_si256((__m256i*)b0);
            __m256i bb1 = _mm256_loadu_si256((__m256i*)b1);

            __m256i rr2 = _mm256_gf2p8mul_epi8(aa1, bb1);
            __m256i rr0 = _mm256_gf2p8mul_epi8(aa0, bb0);
            __m256i rr1 = _mm256_gf2p8mul_epi8(aa0^aa1, bb0^bb1)^rr0;
            rr0 ^= _mm256_gf2p8mul_epi8(rr2, _0x20);
            _mm256_storeu_si256((__m256i*)c1, rr1);
            _mm256_storeu_si256((__m256i*)c0, rr0);
        }
        c0 += rem;
        c1 += rem;
        a0 += rem;
        a1 += rem;
        b0 += rem;
        b1 += rem;
        len -= rem;
    }
    while( len ) {
        __m256i aa0 = _mm256_loadu_si256((__m256i*)a0);
        __m256i aa1 = _mm256_loadu_si256((__m256i*)a1);
        __m256i bb0 = _mm256_loadu_si256((__m256i*)b0);
        __m256i bb1 = _mm256_loadu_si256((__m256i*)b1);

        __m256i rr2 = _mm256_gf2p8mul_epi8(aa1, bb1);
        __m256i rr0 = _mm256_gf2p8mul_epi8(aa0, bb0);
        __m256i rr1 = _mm256_gf2p8mul_epi8(aa0^aa1, bb0^bb1)^rr0;
        rr0 ^= _mm256_gf2p8mul_epi8(rr2, _0x20);
        _mm256_storeu_si256((__m256i*)c1, rr1);
        _mm256_storeu_si256((__m256i*)c0, rr0);
        c0 += 32;
        c1 += 32;
        a0 += 32;
        a1 += 32;
        b0 += 32;
        b1 += 32;
        len -= 32;
    }
}

void gf256t4v_mul( uint8_t * c0 , uint8_t * c1 , uint8_t * c2 , uint8_t * c3 ,
    const uint8_t * a0 , const uint8_t * a1 , const uint8_t * a2 , const uint8_t * a3 ,
    const uint8_t * b0 , const uint8_t * b1 , const uint8_t * b2 , const uint8_t * b3 , unsigned len )
{
    __m256i _0x20 = _mm256_set1_epi8(0x20);
    unsigned rem = len & 31;
    if (rem) {
        if (len < 32 ) {
            uint8_t temp[32] __attribute__((aligned(32)));
            memcpy(temp, a0, rem); __m256i aa0 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, a1, rem); __m256i aa1 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, a2, rem); __m256i aa2 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, a3, rem); __m256i aa3 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b0, rem); __m256i bb0 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b1, rem); __m256i bb1 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b2, rem); __m256i bb2 = _mm256_load_si256((__m256i*)temp);
            memcpy(temp, b3, rem); __m256i bb3 = _mm256_load_si256((__m256i*)temp);
            ymm_x4 rr = _gf256t4v_mul( aa0 , aa1 , aa2 , aa3 , bb0 , bb1 , bb2 , bb3 , _0x20 );
            _mm256_store_si256((__m256i*)temp, rr.val[0]); memcpy(c0, temp, rem);
            _mm256_store_si256((__m256i*)temp, rr.val[1]); memcpy(c1, temp, rem);
            _mm256_store_si256((__m256i*)temp, rr.val[2]); memcpy(c2, temp, rem);
            _mm256_store_si256((__m256i*)temp, rr.val[3]); memcpy(c3, temp, rem);
            return;
        } else {
            __m256i aa0 = _mm256_loadu_si256((__m256i*)a0);
            __m256i aa1 = _mm256_loadu_si256((__m256i*)a1);
            __m256i aa2 = _mm256_loadu_si256((__m256i*)a2);
            __m256i aa3 = _mm256_loadu_si256((__m256i*)a3);
            __m256i bb0 = _mm256_loadu_si256((__m256i*)b0);
            __m256i bb1 = _mm256_loadu_si256((__m256i*)b1);
            __m256i bb2 = _mm256_loadu_si256((__m256i*)b2);
            __m256i bb3 = _mm256_loadu_si256((__m256i*)b3);
            ymm_x4 rr = _gf256t4v_mul( aa0 , aa1 , aa2 , aa3 , bb0 , bb1 , bb2 , bb3 , _0x20 );
            _mm256_storeu_si256((__m256i*)c0, rr.val[0]);
            _mm256_storeu_si256((__m256i*)c1, rr.val[1]);
            _mm256_storeu_si256((__m256i*)c2, rr.val[2]);
            _mm256_storeu_si256((__m256i*)c3, rr.val[3]);
        }
        c0 += rem;
        c1 += rem;
        c2 += rem;
        c3 += rem;
        a0 += rem;
        a1 += rem;
        a2 += rem;
        a3 += rem;
        b0 += rem;
        b1 += rem;
        b2 += rem;
        b3 += rem;
        len -= rem;
    }
    while( len ) {
        __m256i aa0 = _mm256_loadu_si256((__m256i*)a0);
        __m256i aa1 = _mm256_loadu_si256((__m256i*)a1);
        __m256i aa2 = _mm256_loadu_si256((__m256i*)a2);
        __m256i aa3 = _mm256_loadu_si256((__m256i*)a3);
        __m256i bb0 = _mm256_loadu_si256((__m256i*)b0);
        __m256i bb1 = _mm256_loadu_si256((__m256i*)b1);
        __m256i bb2 = _mm256_loadu_si256((__m256i*)b2);
        __m256i bb3 = _mm256_loadu_si256((__m256i*)b3);
        ymm_x4 rr = _gf256t4v_mul( aa0 , aa1 , aa2 , aa3 , bb0 , bb1 , bb2 , bb3 , _0x20 );
        _mm256_storeu_si256((__m256i*)c0, rr.val[0]);
        _mm256_storeu_si256((__m256i*)c1, rr.val[1]);
        _mm256_storeu_si256((__m256i*)c2, rr.val[2]);
        _mm256_storeu_si256((__m256i*)c3, rr.val[3]);
        c0 += 32;
        c1 += 32;
        c2 += 32;
        c3 += 32;
        a0 += 32;
        a1 += 32;
        a2 += 32;
        a3 += 32;
        b0 += 32;
        b1 += 32;
        b2 += 32;
        b3 += 32;
        len -= 32;
    }
}

