#ifndef _GF256V_NEON_H_
#define _GF256V_NEON_H_

#include <stdint.h>
#include <arm_neon.h>


static inline
uint8x16_t polymul8_low(uint8x16_t a, uint8x16_t b)
{
    poly8x8_t a0 = vget_low_p8(vreinterpretq_p8_u8(a));
    poly8x8_t b0 = vget_low_p8(vreinterpretq_p8_u8(b));
    return vreinterpretq_u8_p16( vmull_p8(a0, b0) );
}

static inline
uint8x16_t polymul8_high(uint8x16_t a, uint8x16_t b)
{
    return vreinterpretq_u8_p16( vmull_high_p8(vreinterpretq_p8_u8(a), vreinterpretq_p8_u8(b)) );
}

static const unsigned char __gf256_bit8_11_reduce[16] __attribute__((aligned(16))) = {
    0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41, 0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99
};

static const unsigned char __gf256_bit12_15_reduce[16] __attribute__((aligned(16))) = {
    0x00, 0xab, 0x4d, 0xe6, 0x9a, 0x31, 0xd7, 0x7c, 0x2f, 0x84, 0x62, 0xc9, 0xb5, 0x1e, 0xf8, 0x53
};

static const unsigned char __gf256_bit16_19_reduce[16] __attribute__((aligned(16))) = {
    0, 0x5e, 0xbc, 0xe2, 0x63, 0x3d, 0xdf, 0x81, 0xc6, 0x98, 0x7a, 0x24, 0xa5, 0xfb, 0x19, 0x47
};

static inline
uint8x16_t _gf256v_mul_core( uint8x16_t aa , uint8x16_t bb , uint8x16_t mask_f , uint8x16_t tab_rd0, uint8x16_t tab_rd1)
{
    // multiply
    uint8x16_t ab0 = polymul8_low(aa, bb);
    uint8x16_t ab1 = polymul8_high(aa, bb);
    // reorder data
    uint8x16_t abl = vuzp1q_u8(ab0, ab1);
    uint8x16_t abh = vuzp2q_u8(ab0, ab1);
    // reduce
    uint8x16_t rr = abl ^ vqtbl1q_u8( tab_rd0, abh & mask_f ) ^ vqtbl1q_u8( tab_rd1, vshrq_n_u8(abh, 4) );
    return rr;
}

static inline
uint8x16_t _gf256v_mul( uint8x16_t aa , uint8x16_t bb )
{
    uint8x16_t mask_f = vdupq_n_u8(0x0f);
    uint8x16_t tab_rd0 = vld1q_u8(__gf256_bit8_11_reduce);
    uint8x16_t tab_rd1 = vld1q_u8(__gf256_bit12_15_reduce);

    return _gf256v_mul_core(aa,bb,mask_f,tab_rd0,tab_rd1);
}

static inline
uint8x16x2_t _gf256x2v_mul_core( uint8x16_t aa0 , uint8x16_t aa1 , uint8x16_t bb0 , uint8x16_t bb1 ,
    uint8x16_t mask_f , uint8x16_t tab_rd0, uint8x16_t tab_rd1, uint8x16_t tab_rd2 )
{
    // multiply
    uint8x16_t ab0l = polymul8_low(aa0, bb0);
    uint8x16_t ab0h = polymul8_high(aa0, bb0);
    uint16x8_t ab2l = vreinterpretq_u16_u8(polymul8_low(aa1, bb1));
    uint16x8_t ab2h = vreinterpretq_u16_u8(polymul8_high(aa1, bb1));
    uint8x16_t a0_a1 = aa0^aa1;
    uint8x16_t b0_b1 = bb0^bb1;
    uint8x16_t ab1l = polymul8_low(a0_a1, b0_b1)^ab0l;
    uint8x16_t ab1h = polymul8_high(a0_a1, b0_b1)^ab0h;
    ab0l ^= vreinterpretq_u8_u16(vshlq_n_u16( ab2l , 5 ));  // ab2l x 0x20
    ab0h ^= vreinterpretq_u8_u16(vshlq_n_u16( ab2h , 5 ));  // ab2h x 0x20
    // still need to reduce bit 11-14 from ab2 
    // reorder data
    uint8x16_t c0l = vuzp1q_u8(ab0l, ab0h);
    uint8x16_t c0h = vuzp2q_u8(ab0l, ab0h);
    uint8x16_t c1l = vuzp1q_u8(ab1l, ab1h);
    uint8x16_t c1h = vuzp2q_u8(ab1l, ab1h);
    uint8x16_t c2r = vshrq_n_u8(vuzp2q_u8(ab2l, ab2h),3); // drop bit 8-10 for reducing bit 11-14
    // reduce
    uint8x16_t r0 = c0l ^ vqtbl1q_u8( tab_rd0, c0h & mask_f ) ^ vqtbl1q_u8( tab_rd1, vshrq_n_u8(c0h, 4) ) ^ vqtbl1q_u8( tab_rd2 , c2r );
    uint8x16_t r1 = c1l ^ vqtbl1q_u8( tab_rd0, c1h & mask_f ) ^ vqtbl1q_u8( tab_rd1, vshrq_n_u8(c1h, 4) );

    uint8x16x2_t rr;
    rr.val[0] = r0;
    rr.val[1] = r1;
    return rr;
}

static inline
uint8x16x2_t _gf256x2v_mul( uint8x16_t aa0 , uint8x16_t aa1 , uint8x16_t bb0 , uint8x16_t bb1 )
{
    uint8x16_t mask_f = vdupq_n_u8(0x0f);
    uint8x16_t tab_rd0 = vld1q_u8(__gf256_bit8_11_reduce);
    uint8x16_t tab_rd1 = vld1q_u8(__gf256_bit12_15_reduce);
    uint8x16_t tab_rd2 = vld1q_u8(__gf256_bit16_19_reduce);

    return _gf256x2v_mul_core(aa0,aa1,bb0,bb1,mask_f,tab_rd0,tab_rd1,tab_rd2);
}

#endif

