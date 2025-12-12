
#include "bc_1.h"

#include "arm_neon.h"

static inline
uint8x16_t _vmulq_p8(uint8x16_t a,uint8x16_t b) { return vreinterpretq_u8_p8(vmulq_p8(vreinterpretq_p8_u8(a),vreinterpretq_p8_u8(b))); }

void bc_1_256( void *_poly , unsigned n_256bit )
{
  uint8_t *poly = (uint8_t*)_poly;
  static uint8_t _div_s2_s1_h[16] = {0x00,0x02,0x04,0x06, 0x28,0x2a,0x2c,0x2e, 0x72,0x70,0x76,0x74, 0x5a,0x58, 0x5e, 0x5c};
  static uint8_t _div_x2_x_diff[16] = {0,0,0,0,  2,2,2,2,  6,6,6,6,  4,4,4,4};
  uint8x16_t div_s2_s1_h   = vld1q_u8(_div_s2_s1_h);
  uint8x16_t div_x2_x_diff = vld1q_u8(_div_x2_x_diff);

  // 0,8,4,12, 2,10,6,14,  1,9,5,13,  3,11,7,15
  static uint8_t _recovery_idx[16] = { 0,8,4,12, 2,10,6,14, 1,9,5,13, 3,11,7,15 };
  uint8x16_t recovery_idx = vld1q_u8(_recovery_idx);
  uint8x16_t mask_0x16 = vdupq_n_u8(0x16);
  uint8x16_t mask_15 = vdupq_n_u8(15);
  uint8x16_t zero = vdupq_n_u8(0);

  for(int i=n_256bit;i>0;i--) {
    uint8x16_t p0 = vld1q_u8(poly);     // 0, 1
    uint8x16_t p1 = vld1q_u8(poly+16);  // 2, 3

//layer 13: [256] s4^8:(128,8) suggest unit:1
    p1 ^= vextq_u8(p1,zero,15);
    p0 ^= vextq_u8(zero,p1,15);

//layer 14: [128] s4^4:(64,4) suggest unit:1
    uint64x2_t t64_0 = vuzp1q_u64(vreinterpretq_u64_u8(p0),vreinterpretq_u64_u8(p1));  // 0,2
    uint64x2_t t64_1 = vuzp2q_u64(vreinterpretq_u64_u8(p0),vreinterpretq_u64_u8(p1));  // 1,3

    t64_1 ^= vshrq_n_u64(t64_1,60);
    t64_0 ^= vshlq_n_u64(t64_1,4);

//layer 15: [64] s4^2:(32,2) suggest unit:1
    uint32x4_t t32_0 = vuzp1q_u32(vreinterpretq_u32_u64(t64_0),vreinterpretq_u32_u64(t64_1)); // 0,4,2,6
    uint32x4_t t32_1 = vuzp2q_u32(vreinterpretq_u32_u64(t64_0),vreinterpretq_u32_u64(t64_1)); // 1,5,3,7

    t32_1 ^= vshrq_n_u32(t32_1,30);
    t32_0 ^= vshlq_n_u32(t32_1,2);

//layer 16: [32] s4^1:(16,1) suggest unit:1
    uint16x8_t t16_0 = vuzp1q_u16(vreinterpretq_u16_u32(t32_0),vreinterpretq_u16_u32(t32_1));  // 0,8,4,12, 2,10,6,14
    uint16x8_t t16_1 = vuzp2q_u16(vreinterpretq_u16_u32(t32_0),vreinterpretq_u16_u32(t32_1));  // 1,9,5,13, 3,11,7,15

    t16_1 ^= vshrq_n_u16(t16_1,15);
    t16_0 ^= vshlq_n_u16(t16_1,1);

    uint8x16_t t8_0 = vuzp1q_u8(vreinterpretq_u8_u16(t16_0),vreinterpretq_u8_u16(t16_1));  // 0,16,8,24, 4,20,12,28,  2,18,10,26,  6,22,14,30 <- 0,1,16,17, 8,9,24,25, 4,5,20,21, 12,13,28,29
    uint8x16_t t8_1 = vuzp2q_u8(vreinterpretq_u8_u16(t16_0),vreinterpretq_u8_u16(t16_1));  // 1,17,9,25, 5,21,13,29,  3,19,11,27,  7,23,15,31 <- 2,3,18,19, 10,11,26,27, 6,7,22,23, 14,15,30,31
//  [16] s3^1:  s3 = x^8 + x^4 + x^2 + x
    uint8x16_t mm1 = _vmulq_p8(mask_0x16, vshrq_n_u8(t8_1,4));
    t8_1 ^= vshrq_n_u8(mm1,4);
    t8_0 ^= _vmulq_p8(mask_0x16, t8_1);
//layer 22: [8] s2^1:(4,1) suggest unit:1
    t8_1 ^= vqtbl1q_u8( div_s2_s1_h ,vshrq_n_u8(t8_1,4));
    t8_0 ^= vqtbl1q_u8( div_s2_s1_h ,vshrq_n_u8(t8_0,4));
//layer 24: [4] s1^1:(2,1) suggest unit:1
    t8_1 ^= vqtbl1q_u8( div_x2_x_diff ,t8_1&mask_15);
    t8_0 ^= vqtbl1q_u8( div_x2_x_diff ,t8_0&mask_15);

//layer 17: [256] s2^2:(128,32) suggest unit:16
    uint8x16_t q8_0 = vqtbl1q_u8(t8_0,recovery_idx);  // 0,2,4,6,  8,10,12,14, 16,18,20,22, 24,26,28,30
    uint8x16_t q8_1 = vqtbl1q_u8(t8_1,recovery_idx);  // 1,3,5,7,  9,11,13,15, 17,19,21,23, 25,27,29,31

    uint8x16_t r8_0 = vzip1q_u8( q8_0 , q8_1 );
    uint8x16_t r8_1 = vzip2q_u8( q8_0 , q8_1 );

    r8_1 ^= vextq_u8(r8_1,zero,12);
    r8_0 ^= vextq_u8(zero,r8_1,12);
//layer 18: [128] s2^1:(64,16) suggest unit:16
    uint64x2_t r64_0 = vuzp1q_u64(vreinterpretq_u64_u8(r8_0),vreinterpretq_u64_u8(r8_1));  // 0,2
    uint64x2_t r64_1 = vuzp2q_u64(vreinterpretq_u64_u8(r8_0),vreinterpretq_u64_u8(r8_1));  // 1,3

    r64_1 ^= vshrq_n_u64(r64_1,48);
    r64_0 ^= vshlq_n_u64(r64_1,16);
//layer 20: [64] s1^1:(32,16) suggest unit:16
    uint32x4_t r32_0 = vuzp1q_u32(vreinterpretq_u32_u64(r64_0),vreinterpretq_u32_u64(r64_1));  // 0,4,2,6
    uint32x4_t r32_1 = vuzp2q_u32(vreinterpretq_u32_u64(r64_0),vreinterpretq_u32_u64(r64_1));  // 1,5,3,7

    r32_1 ^= vshrq_n_u32(r32_1,16);
    r32_0 ^= vshlq_n_u32(r32_1,16);
//layer 19: [256] s1^1:(128,64) suggest unit:64
    r64_0 = vreinterpretq_u64_u32(vzip1q_u32(r32_0,r32_1));  // 0,1,4,5 -> 0,2
    r64_1 = vreinterpretq_u64_u32(vzip2q_u32(r32_0,r32_1));  // 2,3,6,7 -> 1,3

    uint8x16_t r0 = vreinterpretq_u8_u64(vzip1q_u64(r64_0,r64_1));
    uint8x16_t r1 = vreinterpretq_u8_u64(vzip2q_u64(r64_0,r64_1));

    r1 ^= vextq_u8(r1,zero,8);
    r0 ^= vextq_u8(zero,r1,8);

    vst1q_u8(poly,   r0);
    vst1q_u8(poly+16, r1);

    poly += 32;
  }
}


void ibc_1_256( void *_poly , unsigned n_256bit )
{
  uint8_t *poly = (uint8_t*)_poly;
  uint8x16_t mask_0x16 = vdupq_n_u8(0x16);
  uint8x16_t mask_15 = vdupq_n_u8(15);
  uint8x16_t zero = vdupq_n_u8(0);

  static uint8_t _mul_x2_x_diff[16] = {0,0,0,0, 2,2,2,2, 4,4,4,4, 6,6,6,6};
  static uint8_t _mul_s1_s2_h[16] = {0x00,0x02,0x04,0x06, 0x2c,0x2e,0x28,0x2a, 0x58,0x5a,0x5c,0x5e, 0x74,0x76,0x70,0x72};
  static uint8_t _mul_0x16_high[16] = {0x0,0x1,0x2,0x3,0x5,0x4,0x7,0x6,0xb,0xa,0x9,0x8,0xe,0xf,0xc,0xd};
  uint8x16_t mul_x2_x_diff = vld1q_u8(_mul_x2_x_diff);
  uint8x16_t mul_s1_s2_h   = vld1q_u8(_mul_s1_s2_h);
  uint8x16_t mul_0x16_high = vld1q_u8(_mul_0x16_high);

  for(int i=n_256bit;i>0;i--) {
    uint8x16_t np0 = vld1q_u8(poly);    // 0, 1
    uint8x16_t np1 = vld1q_u8(poly+16);  // 2, 3

//layer 24: [4] s1^1:(2,1) suggest unit:1
    np0 ^= vqtbl1q_u8( mul_x2_x_diff ,np0&mask_15);
    np1 ^= vqtbl1q_u8( mul_x2_x_diff ,np1&mask_15);

//layer 22: [8] s2^1:(4,1) suggest unit:1
    np0 ^= vqtbl1q_u8( mul_s1_s2_h , vshrq_n_u8(np0,4) );
    np1 ^= vqtbl1q_u8( mul_s1_s2_h , vshrq_n_u8(np1,4) );

//layer ??:  [16] s3^1:  s3 = x^8 + x^4 + x^2 + x
    uint8x16_t t8_0 = vuzp1q_u8(np0,np1);
    uint8x16_t t8_1 = vuzp2q_u8(np0,np1);

    t8_0 ^= _vmulq_p8(mask_0x16, t8_1);
    //t8_1 ^= vshrq_n_u8( _vmulq_p8(mask_0x16, vshrq_n_u8(t8_1,4)) ,4);  // XXX: use tbl to do the mul_hi
    t8_1 ^= vqtbl1q_u8( mul_0x16_high , vshrq_n_u8(t8_1,4) );

    np0 = vzip1q_u8(t8_0,t8_1);
    np1 = vzip2q_u8(t8_0,t8_1);

////////////

//layer 20: [64] s1^1:(32,16) suggest unit:16
    uint32x4_t p32_0 = vuzp1q_u32(vreinterpretq_u32_u8(np0),vreinterpretq_u32_u8(np1));  // 0,1,2,3 -> 0,2,4,6
    uint32x4_t p32_1 = vuzp2q_u32(vreinterpretq_u32_u8(np0),vreinterpretq_u32_u8(np1));  // 4,5,6,7 -> 1,3,5,7

    p32_0 ^= vshlq_n_u32(p32_1,16);
    p32_1 ^= vshrq_n_u32(p32_1,16);

    np0 = vreinterpretq_u8_u32(vzip1q_u32(p32_0,p32_1));
    np1 = vreinterpretq_u8_u32(vzip2q_u32(p32_0,p32_1));

//layer 19: [256] s1^1:(128,64) suggest unit:64
    np0 ^= vextq_u8(zero,np1,8);
    np1 ^= vextq_u8(np1,zero,8);

//layer 18: [128] s2^1:(64,16) suggest unit:16
    uint64x2_t p64_0 = vuzp1q_u64(vreinterpretq_u64_u8(np0),vreinterpretq_u64_u8(np1));  // 0,2
    uint64x2_t p64_1 = vuzp2q_u64(vreinterpretq_u64_u8(np0),vreinterpretq_u64_u8(np1));  // 1,3

    p64_0 ^= vshlq_n_u64(p64_1,16);
    p64_1 ^= vshrq_n_u64(p64_1,48);

    np0 = vreinterpretq_u8_u64(vzip1q_u64(p64_0,p64_1));
    np1 = vreinterpretq_u8_u64(vzip2q_u64(p64_0,p64_1));

//layer 17: [256] s2^2:(128,32) suggest unit:16
    np0 ^= vextq_u8(zero,np1,12);
    np1 ^= vextq_u8(np1,zero,12);

//////////

//layer 16: [32] s4^1:(16,1) suggest unit:1
    uint16x8_t t16_0 = vuzp1q_u16(vreinterpretq_u16_u8(np0),vreinterpretq_u16_u8(np1));
    uint16x8_t t16_1 = vuzp2q_u16(vreinterpretq_u16_u8(np0),vreinterpretq_u16_u8(np1));

    t16_0 ^= vshlq_n_u16(t16_1,1);
    t16_1 ^= vshrq_n_u16(t16_1,15);

    np0 = vreinterpretq_u8_u16(vzip1q_u16(t16_0,t16_1));
    np1 = vreinterpretq_u8_u16(vzip2q_u16(t16_0,t16_1));

//layer 15: [64] s4^2:(32,2) suggest unit:1
    uint32x4_t t32_0 = vuzp1q_u32(vreinterpretq_u32_u8(np0),vreinterpretq_u32_u8(np1));
    uint32x4_t t32_1 = vuzp2q_u32(vreinterpretq_u32_u8(np0),vreinterpretq_u32_u8(np1));

    t32_0 ^= vshlq_n_u32(t32_1,2);
    t32_1 ^= vshrq_n_u32(t32_1,30);

    np0 = vreinterpretq_u8_u32(vzip1q_u32(t32_0,t32_1));
    np1 = vreinterpretq_u8_u32(vzip2q_u32(t32_0,t32_1));

//layer 14: [128] s4^4:(64,4) suggest unit:1
    uint64x2_t t64_0 = vuzp1q_u64(vreinterpretq_u64_u8(np0),vreinterpretq_u64_u8(np1));
    uint64x2_t t64_1 = vuzp2q_u64(vreinterpretq_u64_u8(np0),vreinterpretq_u64_u8(np1));

    t64_0 ^= vshlq_n_u64(t64_1,4);
    t64_1 ^= vshrq_n_u64(t64_1,60);

    np0 = vreinterpretq_u8_u64(vzip1q_u64(t64_0,t64_1));
    np1 = vreinterpretq_u8_u64(vzip2q_u64(t64_0,t64_1));

//layer 13: [256] s4^8:(128,8) suggest unit:1
    np0 ^= vextq_u8(zero,np1,15);
    np1 ^= vextq_u8(np1,zero,15);

    vst1q_u8(poly,    np0);
    vst1q_u8(poly+16, np1);

    poly += 32;
  }
}




/////////////////////////////////////////



static inline
void _div_X256_X1( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1 = zero;
  uint8x16_t src0 = vld1q_u8( poly + 3*16 );
  uint8x16_t tmp1 = vextq_u8( src0, src1, 15 );
  uint8x16_t dest = vld1q_u8( poly + 2*16 );
  dest ^= vshrq_n_u8(tmp1,7);
  vst1q_u8( poly+2*16 , dest );

  src1 = src0;
  src0 = dest;
  dest = vld1q_u8( poly + 1*16 );
  tmp1 = vextq_u8( src0, src1, 15 );
  dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,7) , src1 , 1 );
  vst1q_u8( poly+1*16 , dest );

  src1 = src0;
  src0 = zero;
  dest = vld1q_u8( poly );
  tmp1 = vextq_u8( src0, src1, 15 );
  dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,7) , src1 , 1 );
  vst1q_u8( poly , dest );
}

static inline
void _div_X512_X2( uint8_t *poly , uint8x16_t zero) {
#define H 4
#define L 2
  uint8x16_t src1 = zero;
  uint8x16_t src0 = vld1q_u8( poly + (H*2-1)*16 );
  uint8x16_t tmp1 = vextq_u8( src0, src1, 15 );
  uint8x16_t dest = vld1q_u8( poly + H*16 );
  dest ^= vshrq_n_u8(tmp1,8-L);
  vst1q_u8( poly+H*16 , dest );

  for(int i=1;i<H;i++) {
    src1 = src0;
    src0 = vld1q_u8( poly+(H*2-1-i)*16);
    dest = vld1q_u8( poly+(H-i)*16);
    tmp1 = vextq_u8( src0, src1, 15 );
    dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,8-L) , src1 , L );
    vst1q_u8( poly+(H-i)*16 , dest );
  }
  src1 = src0;
  src0 = zero;
  dest = vld1q_u8( poly );
  tmp1 = vextq_u8( src0, src1, 15 );
  dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,8-L) , src1 , L );
  vst1q_u8( poly , dest );
#undef H
#undef L
}

static inline
void _div_X1024_X4( uint8_t *poly , uint8x16_t zero) {
#define H 8
#define L 4
  uint8x16_t src1 = zero;
  uint8x16_t src0 = vld1q_u8( poly + (H*2-1)*16 );
  uint8x16_t tmp1 = vextq_u8( src0, src1, 15 );
  uint8x16_t dest = vld1q_u8( poly + H*16 );
  dest ^= vshrq_n_u8(tmp1,8-L);
  vst1q_u8( poly+H*16 , dest );

  for(int i=1;i<H;i++) {
    src1 = src0;
    src0 = vld1q_u8( poly+(H*2-1-i)*16);
    dest = vld1q_u8( poly+(H-i)*16);
    tmp1 = vextq_u8( src0, src1, 15 );
    dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,8-L) , src1 , L );
    vst1q_u8( poly+(H-i)*16 , dest );
  }

  src1 = src0;
  src0 = zero;
  dest = vld1q_u8( poly );
  tmp1 = vextq_u8( src0, src1, 15 );
  dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,8-L) , src1 , L );
  vst1q_u8( poly , dest );
#undef H
#undef L
}

static inline
void _div_X2048_X8( uint8_t *poly , uint8x16_t zero) {
#define H 16
#define L 1
  uint8x16_t src1 = zero;
  uint8x16_t src0 = vld1q_u8( poly + (H*2-1)*16 );
  uint8x16_t dest = vld1q_u8( poly + H*16 ) ^ vextq_u8( src0, src1, 16-L);
  vst1q_u8( poly+H*16 , dest );

  for(int i=1;i<H;i++) {
    src1 = src0;
    src0 = vld1q_u8( poly+(H*2-1-i)*16);
    dest = vld1q_u8( poly+(H-i)*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+(H-i)*16 , dest );
  }

  src1 = src0;
  src0 = zero;
  dest = vld1q_u8( poly ) ^ vextq_u8( src0, src1, 16-L );
  vst1q_u8( poly , dest );
#undef H
#undef L
}

static inline
void _div_X4096_X16( uint8_t *poly , uint8x16_t zero) {
#define H 32
#define L 2
  uint8x16_t src1 = zero;
  uint8x16_t src0 = vld1q_u8( poly + (H*2-1)*16 );
  uint8x16_t dest = vld1q_u8( poly + H*16 ) ^ vextq_u8( src0, src1, 16-L);
  vst1q_u8( poly+H*16 , dest );

  for(int i=1;i<H;i++) {
    src1 = src0;
    src0 = vld1q_u8( poly+(H*2-1-i)*16);
    dest = vld1q_u8( poly+(H-i)*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+(H-i)*16 , dest );
  }

  src1 = src0;
  src0 = zero;
  dest = vld1q_u8( poly ) ^ vextq_u8( src0, src1, 16-L );
  vst1q_u8( poly , dest );
#undef H
#undef L
}

static inline
void _div_X8192_X32( uint8_t *poly , uint8x16_t zero) {
#define H 64
#define L 4
  uint8x16_t src1 = zero;
  uint8x16_t src0 = vld1q_u8( poly + (H*2-1)*16 );
  uint8x16_t dest = vld1q_u8( poly + H*16 ) ^ vextq_u8( src0, src1, 16-L);
  vst1q_u8( poly+H*16 , dest );

  for(int i=1;i<H;i++) {
    src1 = src0;
    src0 = vld1q_u8( poly+(H*2-1-i)*16);
    dest = vld1q_u8( poly+(H-i)*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+(H-i)*16 , dest );
  }

  src1 = src0;
  src0 = zero;
  dest = vld1q_u8( poly ) ^ vextq_u8( src0, src1, 16-L );
  vst1q_u8( poly , dest );
#undef H
#undef L
}

static inline
void _div_X16384_X64( uint8_t *poly , uint8x16_t zero) {
#define H 128
#define L 8
  uint8x16_t src1 = zero;
  uint8x16_t src0 = vld1q_u8( poly + (H*2-1)*16 );
  uint8x16_t dest = vld1q_u8( poly + H*16 ) ^ vextq_u8( src0, src1, 16-L);
  vst1q_u8( poly+H*16 , dest );

  for(int i=1;i<H;i++) {
    src1 = src0;
    src0 = vld1q_u8( poly+(H*2-1-i)*16);
    dest = vld1q_u8( poly+(H-i)*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+(H-i)*16 , dest );
  }

  src1 = src0;
  src0 = zero;
  dest = vld1q_u8( poly ) ^ vextq_u8( src0, src1, 16-L );
  vst1q_u8( poly , dest );
#undef H
#undef L
}

static inline
void _div_2terms_poly( uint8_t * poly , unsigned HT_deg_X128x, unsigned TT_deg_X128x ) {
  for(unsigned i=(HT_deg_X128x*2-1);i>=HT_deg_X128x;i--) {
    uint8x16_t src  = vld1q_u8(poly+i*16);
    uint8x16_t dest = vld1q_u8(poly+(i-HT_deg_X128x+TT_deg_X128x)*16);
    vst1q_u8(poly+(i-HT_deg_X128x+TT_deg_X128x)*16, src^dest);
  }
}


// s8    = x^256 - x
// s8^2  = x^512 - x^2
// s8^4  = x^1024 - x^4
// s8^8  = x^2048 - x^8
// s8^16 = x^4096 - x^16
// s8^32 = x^8192 - x^32
// s8^64 = x^16384 - x^64
// s8^128 = x^32768 - x^128



static
void repr_s8_1( uint8_t * poly , unsigned n_byte )
{
  unsigned n_128 = n_byte>>4;
  unsigned log_n_128 = __builtin_ctz( n_128 );

  for( unsigned i=log_n_128-1;i>=8;i--) {
    unsigned s_ht = 1<<i;
    unsigned s_tt = 1<<(i-8);
    unsigned len = s_ht<<1;
    // div X^{128*s_ht} - X^{128*s_tt}
    for(unsigned j=0;j<n_128;j+=len) { _div_2terms_poly( poly+j*16 , s_ht , s_tt ); }
  }

  uint8x16_t zero = vdupq_n_u8(0);
  if( log_n_128 > 7 ) { for(unsigned j=0;j<n_128;j+=256) _div_X16384_X64( poly+j*16 , zero ); }
  if( log_n_128 > 6 ) { for(unsigned j=0;j<n_128;j+=128) _div_X8192_X32( poly+j*16 , zero ); }
  if( log_n_128 > 5 ) { for(unsigned j=0;j<n_128;j+=64) _div_X4096_X16( poly+j*16 , zero ); }
  if( log_n_128 > 4 ) { for(unsigned j=0;j<n_128;j+=32) _div_X2048_X8( poly+j*16 , zero ); }
  if( log_n_128 > 3 ) { for(unsigned j=0;j<n_128;j+=16) _div_X1024_X4( poly+j*16 , zero ); }
  if( log_n_128 > 2 ) { for(unsigned j=0;j<n_128;j+=8) _div_X512_X2( poly+j*16 , zero); }
  if( log_n_128 > 1 ) { for(unsigned j=0;j<n_128;j+=4) _div_X256_X1( poly+j*16 , zero); }
}



////////////


static inline
void _idiv_X256_X1( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1,src0,tmp1,dest;
#define H 2
#define L 1
  src1 = zero;
  for(int i=0;i<H;i++) {
    src0 = src1;
    src1 = vld1q_u8( poly + (H+i)*16 );
    dest = vld1q_u8( poly + i*16 );
    tmp1 = vextq_u8( src0, src1, 15 );
    dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,8-L) , src1 , L );
    vst1q_u8( poly + i*16 , dest );
  }
  tmp1 = vextq_u8( src1, zero, 15 );
  dest = vld1q_u8( poly + H*16 );
  dest ^= vshrq_n_u8(tmp1,8-L);
  vst1q_u8( poly + H*16 , dest );
#undef H
#undef L
}

static inline
void _idiv_X512_X2( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1,src0,tmp1,dest;
#define H 4
#define L 2
  src1 = zero;
  for(int i=0;i<H;i++) {
    src0 = src1;
    src1 = vld1q_u8( poly + (H+i)*16 );
    dest = vld1q_u8( poly + i*16 );
    tmp1 = vextq_u8( src0, src1, 15 );
    dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,8-L) , src1 , L );
    vst1q_u8( poly + i*16 , dest );
  }
  tmp1 = vextq_u8( src1, zero, 15 );
  dest = vld1q_u8( poly + H*16 );
  dest ^= vshrq_n_u8(tmp1,8-L);
  vst1q_u8( poly + H*16 , dest );
#undef H
#undef L
}

static inline
void _idiv_X1024_X4( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1,src0,tmp1,dest;
#define H 8
#define L 4
  src1 = zero;
  for(int i=0;i<H;i++) {
    src0 = src1;
    src1 = vld1q_u8( poly + (H+i)*16 );
    dest = vld1q_u8( poly + i*16 );
    tmp1 = vextq_u8( src0, src1, 15 );
    dest ^= vsliq_n_u8( vshrq_n_u8(tmp1,8-L) , src1 , L );
    vst1q_u8( poly + i*16 , dest );
  }
  tmp1 = vextq_u8( src1, zero, 15 );
  dest = vld1q_u8( poly + H*16 );
  dest ^= vshrq_n_u8(tmp1,8-L);
  vst1q_u8( poly + H*16 , dest );
#undef H
#undef L
}

static inline
void _idiv_X2048_X8( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1,src0,dest;
#define H 16
#define L 1
  src1 = zero;
  for(int i=0;i<H;i++) {
    src0 = src1;
    src1 = vld1q_u8( poly+(H+i)*16);
    dest = vld1q_u8( poly+i*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+i*16 , dest );
  }
  dest = vld1q_u8( poly+(H)*16 ) ^ vextq_u8( src1, zero, 16-L );
  vst1q_u8( poly+(H)*16 , dest );
#undef H
#undef L
}

static inline
void _idiv_X4096_X16( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1,src0,dest;
#define H 32
#define L 2
  src1 = zero;
  for(int i=0;i<H;i++) {
    src0 = src1;
    src1 = vld1q_u8( poly+(H+i)*16);
    dest = vld1q_u8( poly+i*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+i*16 , dest );
  }
  dest = vld1q_u8( poly+(H)*16 ) ^ vextq_u8( src1, zero, 16-L );
  vst1q_u8( poly+(H)*16 , dest );
#undef H
#undef L
}

static inline
void _idiv_X8192_X32( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1,src0,dest;
#define H 64
#define L 4
  src1 = zero;
  for(int i=0;i<H;i++) {
    src0 = src1;
    src1 = vld1q_u8( poly+(H+i)*16);
    dest = vld1q_u8( poly+i*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+i*16 , dest );
  }
  dest = vld1q_u8( poly+(H)*16 ) ^ vextq_u8( src1, zero, 16-L );
  vst1q_u8( poly+(H)*16 , dest );
#undef H
#undef L
}

static inline
void _idiv_X16384_X64( uint8_t *poly , uint8x16_t zero) {
  uint8x16_t src1,src0,dest;
#define H 128
#define L 8
  src1 = zero;
  for(int i=0;i<H;i++) {
    src0 = src1;
    src1 = vld1q_u8( poly+(H+i)*16);
    dest = vld1q_u8( poly+i*16)  ^ vextq_u8( src0, src1, 16-L);
    vst1q_u8( poly+i*16 , dest );
  }
  dest = vld1q_u8( poly+(H)*16 ) ^ vextq_u8( src1, zero, 16-L );
  vst1q_u8( poly+(H)*16 , dest );
#undef H
#undef L
}

static inline
void _idiv_2terms_poly( uint8_t * poly , unsigned HT_deg_X128x, unsigned TT_deg_X128x ) {
  for(unsigned i=HT_deg_X128x;i<HT_deg_X128x*2;i++) {
    uint8x16_t src  = vld1q_u8(poly+i*16);
    uint8x16_t dest = vld1q_u8(poly+(i-HT_deg_X128x+TT_deg_X128x)*16);
    vst1q_u8(poly+(i-HT_deg_X128x+TT_deg_X128x)*16, src^dest);
  }
}


static
void irepr_s8_1( uint8_t * poly , unsigned n_byte )
{
  unsigned n_128 = n_byte>>4;
  unsigned log_n_128 = __builtin_ctz( n_128 );

  uint8x16_t zero = vdupq_n_u8(0);
  if( log_n_128 > 1 ) { for(unsigned j=0;j<n_128;j+=4) _idiv_X256_X1( poly+j*16 , zero); }
  if( log_n_128 > 2 ) { for(unsigned j=0;j<n_128;j+=8) _idiv_X512_X2( poly+j*16 , zero); }
  if( log_n_128 > 3 ) { for(unsigned j=0;j<n_128;j+=16) _idiv_X1024_X4( poly+j*16 , zero ); }
  if( log_n_128 > 4 ) { for(unsigned j=0;j<n_128;j+=32) _idiv_X2048_X8( poly+j*16 , zero ); }
  if( log_n_128 > 5 ) { for(unsigned j=0;j<n_128;j+=64) _idiv_X4096_X16( poly+j*16 , zero ); }
  if( log_n_128 > 6 ) { for(unsigned j=0;j<n_128;j+=128) _idiv_X8192_X32( poly+j*16 , zero ); }
  if( log_n_128 > 7 ) { for(unsigned j=0;j<n_128;j+=256) _idiv_X16384_X64( poly+j*16 , zero ); }
  for( unsigned i=8;i<log_n_128;i++) {
    unsigned s_ht = 1<<i;
    unsigned s_tt = 1<<(i-8);
    unsigned len = s_ht<<1;
    // idiv X^{128*s_ht} - X^{128*s_tt}
    for(unsigned j=0;j<n_128;j+=len) { _idiv_2terms_poly( poly+j*16 , s_ht , s_tt ); }
  }
}


/////////////////////////////////////////

#include "bc_256.h"


void bc_1( void * poly , unsigned n_byte )
{
  repr_s8_1( poly , n_byte );
  bc_1_256( poly , n_byte>>5 );
  bc_256( poly , n_byte>>5 );
}



void ibc_1( void * poly , unsigned n_byte )
{
  ibc_256( poly , n_byte>>5 );
  ibc_1_256( poly , n_byte>>5 );
  irepr_s8_1( poly , n_byte );
}







