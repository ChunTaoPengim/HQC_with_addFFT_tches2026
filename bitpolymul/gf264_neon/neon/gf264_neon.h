
#ifndef _GF264_NEON_H_
#define _GF264_NEON_H_


#include <stdint.h>

#include <arm_neon.h>

static inline
uint64x2_t _vmull_p64( uint64x2_t a , uint64x2_t b )
{
  poly64x2_t pa = vreinterpretq_p64_u64(a);
  poly64x2_t pb = vreinterpretq_p64_u64(b);
  return vreinterpretq_u64_p128( vmull_p64((poly64_t)vget_low_p64(pa),(poly64_t)vget_low_p64(pb)) );
}

static inline
uint64x2_t _vmull_high_p64( uint64x2_t a , uint64x2_t b )
{
  poly64x2_t pa = vreinterpretq_p64_u64(a);
  poly64x2_t pb = vreinterpretq_p64_u64(b);
  return vreinterpretq_u64_p128( vmull_high_p64(pa,pb) );
}

static inline
uint64x2_t _vmulq_p8( uint64x2_t a , uint64x2_t b )
{
  poly8x16_t pa = vreinterpretq_p8_u64(a);
  poly8x16_t pb = vreinterpretq_p8_u64(b);
  return vreinterpretq_u64_p8( vmulq_p8(pa,pb) );
}


/// X^64 + X^4 + X^3 + X + 1
/// 0x1b
//static const uint64_t _gf264_reducer[2] __attribute__((aligned(32)))  = {0x1bULL,0x1bULL};
//static uint64x2_t mask_0x1b = vdupq_n_u64(0x1b);

static inline
uint64x2_t get_mask_0x1b(void) { return vdupq_n_u64(0x1b); }

static inline
uint64x2_t _gf264_reduce_neon( uint64x2_t l0 , uint64x2_t h0 , uint64x2_t mask_0x1b)
{
  uint64x2_t xr0 = _vmull_high_p64(l0,mask_0x1b);  // X,XXXXXXXX XXXXXXXX  <-- 3bits in the high 64bits
  uint64x2_t yr0 = _vmull_high_p64(h0,mask_0x1b);  // X,XXXXXXXX XXXXXXXX  <-- 3bits in the high 64bits

  uint64x2_t x1 = l0 ^ xr0;  // only low 64 bits is meaningful
  uint64x2_t y1 = h0 ^ yr0;  // only low 64 bits is meaningful

  uint64x2_t yrxr = vzip2q_u64(xr0,yr0);
  uint64x2_t yx = vzip1q_u64(x1,y1);

  return yx ^ _vmulq_p8(yrxr,mask_0x1b);
}

static inline
uint64x2_t _gf264_reduce_x1_neon( uint64x2_t l0 , uint64x2_t mask_0x1b)
{
  uint64x2_t xr0 = _vmull_high_p64(l0,mask_0x1b);  // X,XXXXXXXX XXXXXXXX  <-- 3bits in the high 64bits
  uint64x2_t xr1 = _vmulq_p8(vtrn2q_u64(xr0,xr0),mask_0x1b); // trn2q_u64 to duplicate the high 64bits to low 64bits
  return l0 ^ xr0 ^ xr1;  // only low 64 bits is meaningful, high 64-71 bits are garbage
}

static inline
uint64x2_t _gf264_mul_neon( uint64x2_t a , uint64x2_t b , uint64x2_t mask_0x1b)
{
  //uint64x2_t mask_0x1b = vdupq_n_u64(0x1b);
  uint64x2_t x0 = _vmull_p64( a , b );
  uint64x2_t y0 = _vmull_high_p64(a,b);
  return _gf264_reduce_neon( x0 , y0 , mask_0x1b);
}

static inline
uint64_t _gf264_mulx1_neon( uint64_t a , uint64_t b )
{
  uint64x2_t mask_0x1b = get_mask_0x1b();

  uint64x2_t ab0 = vreinterpretq_u64_p128( vmull_p64((poly64_t)a,(poly64_t)b) );
  // reduce
  uint64x2_t ab1r = _vmull_high_p64(ab0,mask_0x1b);  // X,XXXXXXXX XXXXXXXX  <-- 3bits in the high 64bits
  uint64x2_t r0 = ab0 ^ ab1r ^ _vmulq_p8(vtrn2q_u64(ab1r,ab1r),mask_0x1b); // only low 64 bits is meaningful
  return vgetq_lane_u64(r0,0);
}



#endif


