#include "gf256.h"
#include "string.h"  // memcpy

#include <arm_neon.h>

void gf256v_add( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned len )
{
    unsigned rem = len & 15;
    if (rem) {
        if (len < 16 ) {
            for(unsigned i=0;i<len;i++) {
                r[i] = a[i]^b[i];
            }
            return;
        } else {
            uint8x16_t aa = vld1q_u8(a);
            uint8x16_t bb = vld1q_u8(b);
            uint8x16_t r1 = vld1q_u8(r+rem);
            vst1q_u8(r, veorq_u8(aa,bb));
            vst1q_u8(r+rem, r1);
        }
        r += rem;
        a += rem;
        b += rem;
        len -= rem;
    }
    while( len ) {
        uint8x16_t aa = vld1q_u8(a);
        uint8x16_t bb = vld1q_u8(b);
        vst1q_u8(r, veorq_u8(aa,bb));
        r += 16;
        a += 16;
        b += 16;
        len -= 16;
    }
}

#include "gf256v_neon.h"


void gf256v_mul( uint8_t * r , const uint8_t * a , const uint8_t * b , unsigned len )
{
    //uint8x16_t mask_f = vdupq_n_u8(0x0f);
    //uint8x16_t tab_rd0 = vld1q_u8(__gf256_bit8_11_reduce);
    //uint8x16_t tab_rd1 = vld1q_u8(__gf256_bit12_15_reduce);
    unsigned rem = len & 15;
    if (rem) {
        if (len < 16 ) {
            uint8_t temp[16] __attribute__((aligned(16)));
            memcpy(temp, a, rem);
            uint8x16_t aa = vld1q_u8(temp);
            memcpy(temp, b, rem);
            uint8x16_t bb = vld1q_u8(temp);
            uint8x16_t rr = _gf256v_mul(aa, bb);
            vst1q_u8(temp, rr);
            memcpy(r, temp, rem);
            return;
        } else {
            uint8x16_t aa = vld1q_u8(a);
            uint8x16_t bb = vld1q_u8(b);
            uint8x16_t r1 = vld1q_u8(r+rem);
            uint8x16_t rr = _gf256v_mul(aa, bb);
            vst1q_u8(r, rr);
            vst1q_u8(r+rem, r1);
        }
        r += rem;
        a += rem;
        b += rem;
        len -= rem;
    }
    while( len ) {
        uint8x16_t aa = vld1q_u8(a);
        uint8x16_t bb = vld1q_u8(b);
        uint8x16_t rr = _gf256v_mul(aa, bb);
        vst1q_u8(r, rr);
        r += 16;
        a += 16;
        b += 16;
        len -= 16;
    }
}

void gf256x2v_mul( uint8_t * c0 , uint8_t * c1 , 
    const uint8_t * a0 , const uint8_t * a1 , const uint8_t * b0 , const uint8_t * b1 , unsigned len )
{
    //uint8x16_t mask_f = vdupq_n_u8(0x0f);
    //uint8x16_t tab_rd0 = vld1q_u8(__gf256_bit8_11_reduce);
    //uint8x16_t tab_rd1 = vld1q_u8(__gf256_bit12_15_reduce);
    //uint8x16_t tab_rd2 = vld1q_u8(__gf256_bit16_19_reduce);
    unsigned rem = len & 15;
    if (rem) {
        if (len < 16 ) {
            uint8_t temp[16] __attribute__((aligned(16)));
            memcpy(temp, a0, rem);
            uint8x16_t aa0 = vld1q_u8(temp);
            memcpy(temp, a1, rem);
            uint8x16_t aa1 = vld1q_u8(temp);
            memcpy(temp, b0, rem);
            uint8x16_t bb0 = vld1q_u8(temp);
            memcpy(temp, b1, rem);
            uint8x16_t bb1 = vld1q_u8(temp);
            uint8x16x2_t rr = _gf256x2v_mul(aa0, aa1, bb0, bb1);
            vst1q_u8(temp, rr.val[0]);
            memcpy(c0, temp, rem);
            vst1q_u8(temp, rr.val[1]);
            memcpy(c1, temp, rem);
            return;
        } else {
            uint8x16_t cc0 = vld1q_u8(c0+rem);
            uint8x16_t cc1 = vld1q_u8(c1+rem);
            uint8x16_t aa0 = vld1q_u8(a0);
            uint8x16_t aa1 = vld1q_u8(a1);
            uint8x16_t bb0 = vld1q_u8(b0);
            uint8x16_t bb1 = vld1q_u8(b1);
            uint8x16x2_t rr = _gf256x2v_mul(aa0, aa1, bb0, bb1);
            vst1q_u8(c0, rr.val[0]);
            vst1q_u8(c1, rr.val[1]);
            vst1q_u8(c0+rem, cc0);
            vst1q_u8(c1+rem, cc1);
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
        uint8x16_t aa0 = vld1q_u8(a0);
        uint8x16_t aa1 = vld1q_u8(a1);
        uint8x16_t bb0 = vld1q_u8(b0);
        uint8x16_t bb1 = vld1q_u8(b1);
        uint8x16x2_t rr = _gf256x2v_mul(aa0, aa1, bb0, bb1);
        vst1q_u8(c0, rr.val[0]);
        vst1q_u8(c1, rr.val[1]);
        c0 += 16;
        c1 += 16;
        a0 += 16;
        a1 += 16;
        b0 += 16;
        b1 += 16;
        len -= 16;
    }
}
