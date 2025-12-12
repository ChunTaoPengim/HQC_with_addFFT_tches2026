
#include "bc_8.h"




static inline
int choose_si( int loglen )
{
  int si = 1<<0;
  for( int i=1; (1<<i) < loglen ; i++ ) {
    si = 1<<i;
  }
  return si;
}

//////////////  ref code for small cases //////////////////////////////////

static inline
void div_blk( uint8_t *poly, int si_h, int si_l, int polylen )
{
  int deg_diff = si_h-si_l;
  for(int i=polylen-1;i>=si_h;i--) {  poly[(i-deg_diff)] ^= poly[i]; }
}

static
void rep_in_si( uint8_t *data, int datalen, int logsize_blk, int polyloglen_blk,  int si  )
{
  for(int i=polyloglen_blk-1;i>=si;i--) {
    int polylen = (1<<(i+logsize_blk+1));
    int si_h = (1<<(i+logsize_blk));
    int si_l = (1<<(i+logsize_blk-si));
    for(int j=0;j<datalen;j+=polylen) div_blk( data+j , si_h, si_l, polylen );
  }
}

static
void cvt( uint8_t *data, int datalen, int logsize_blk, int polyloglen_blk )
{
  if( 1 >= polyloglen_blk ) return;
  int si = choose_si(polyloglen_blk);
  rep_in_si( data, datalen , logsize_blk , polyloglen_blk , si );

  cvt( data , datalen , logsize_blk , si );
  cvt( data , datalen , logsize_blk+si , polyloglen_blk-si );
}

////////////

static inline
void idiv_blk( uint8_t *poly, int si_h, int si_l, int polylen )
{
  int deg_diff = si_h-si_l;
  for(int i=si_h;i<polylen;i++) { poly[(i-deg_diff)] ^= poly[i]; }
}

static
void irep_in_si( uint8_t *data, int datalen, int logsize_blk, int polyloglen_blk,  int si  )
{
  for(int i=si;i<polyloglen_blk;i++) {
    int polylen = (1<<(i+logsize_blk+1));
    int si_h = (1<<(i+logsize_blk));
    int si_l = (1<<(i+logsize_blk-si));
    for(int j=0;j<datalen;j+=polylen) idiv_blk( data+j , si_h, si_l, polylen );
  }
}

static
void icvt( uint8_t *data, int datalen, int logsize_blk, int polyloglen_blk )
{
  if( 1 >= polyloglen_blk ) return;
  int si = choose_si(polyloglen_blk);

  icvt( data , datalen , logsize_blk+si , polyloglen_blk-si );
  icvt( data , datalen , logsize_blk , si );

  irep_in_si( data, datalen , logsize_blk , polyloglen_blk , si );
}

///////////////////////////////////////////

#include <arm_neon.h>


static inline void bc_8_128( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8(poly);

    // poly has 16 terms
    // div s2^2 = ( x^8 - x^2 )
    // __m256i p0_h = _mm256_srli_si256( p0 , 8 ) ^ _mm256_srli_si256( p0 , 14 );
    uint8x16_t p0_shr8 = vextq_u8(p0, vdupq_n_u8(0), 8);
    uint8x16_t p0_shr14 = vextq_u8(p0, vdupq_n_u8(0), 14);
    uint8x16_t p0_h = veorq_u8(p0_shr8, p0_shr14);
    // __m256i p1   = p0 ^ _mm256_slli_si256(p0_h,2);
    uint8x16_t p0_h_shl2 = vextq_u8(vdupq_n_u8(0), p0_h, 16 - 2);
    uint8x16_t p1 = veorq_u8(p0, p0_h_shl2);

    // 2 x div s2 = x^4 - x
    // __m256i p1_h  = _mm256_srli_epi64(p1,32) ^ _mm256_srli_epi64(p1,56);
    uint64x2_t p1_u64 = vreinterpretq_u64_u8(p1);
    uint64x2_t p1_shr32_epi64 = vshrq_n_u64(p1_u64, 32);
    uint64x2_t p1_shr56_epi64 = vshrq_n_u64(p1_u64, 56);
    uint64x2_t p1_h_u64 = veorq_u64(p1_shr32_epi64, p1_shr56_epi64);
    // __m256i p2 = p1 ^ _mm256_slli_epi64(p1_h,8);
    uint64x2_t p1_h_shl8_epi64 = vshlq_n_u64(p1_h_u64, 8);
    uint64x2_t p2_u64 = veorq_u64(p1_u64, p1_h_shl8_epi64);
    uint8x16_t p2 = vreinterpretq_u8_u64(p2_u64);

    // div s1 = x^2 - x , each coef has 4 terms
    // __m256i p2_h = _mm256_srli_si256(p2, 8) ^ _mm256_srli_si256(p2, 12);
    uint8x16_t p2_shr8 = vextq_u8(p2, vdupq_n_u8(0), 8);
    uint8x16_t p2_shr12 = vextq_u8(p2, vdupq_n_u8(0), 12);
    uint8x16_t p2_h = veorq_u8(p2_shr8, p2_shr12);
        
    // __m256i p3 = p2 ^ _mm256_slli_si256(p2_h, 4);
    uint8x16_t p2_h_shl4 = vextq_u8(vdupq_n_u8(0), p2_h, 16 - 4);
    uint8x16_t p3 = veorq_u8(p2, p2_h_shl4);

    // 4 x div s1 = x^2 - x
    // __m256i p3_h = _mm256_srli_epi32(p3, 16) ^ _mm256_srli_epi32(p3, 24);
    uint32x4_t p3_u32 = vreinterpretq_u32_u8(p3);
    uint32x4_t p3_shr16_epi32 = vshrq_n_u32(p3_u32, 16);
    uint32x4_t p3_shr24_epi32 = vshrq_n_u32(p3_u32, 24);
    uint32x4_t p3_h_u32 = veorq_u32(p3_shr16_epi32, p3_shr24_epi32);
    // __m256i p4 = p3 ^ _mm256_slli_epi32(p3_h, 8);
    uint32x4_t p3_h_shl8_epi32 = vshlq_n_u32(p3_h_u32, 8);
    uint32x4_t p4_u32 = veorq_u32(p3_u32, p3_h_shl8_epi32);
    uint8x16_t p4 = vreinterpretq_u8_u32(p4_u32);

    vst1q_u8(poly,p4);
}

static inline void ibc_8_128( uint8_t * poly )
{
    uint8x16_t p4 = vld1q_u8(poly);
    // poly has 16 terms
    // 4 x div s1 = x^2 - x
    // __m256i p4_h = _mm256_srli_epi32(p4,16);
    // __m256i p3 = p4 ^ _mm256_slli_epi32(p4_h,8);
    uint32x4_t p4_u32 = vreinterpretq_u32_u8(p4);
    uint32x4_t p4_h_u32 = vshrq_n_u32(p4_u32, 16);         // srli_epi32(p4, 16)
    uint32x4_t temp_shift = vshlq_n_u32(p4_h_u32, 8);      // slli_epi32(p4_h, 8)
    uint32x4_t p3_u32 = veorq_u32(p4_u32, temp_shift);       // p4 ^ shift
    uint8x16_t p3 = vreinterpretq_u8_u32(p3_u32);
    
    // div s1 = x^2 - x , each coef has 4 terms
    // __m256i p3_h = _mm256_srli_si256(p3,8);
    // __m256i p2 = p3 ^ _mm256_slli_si256(p3_h,4);
    uint8x16_t p3_h = vextq_u8(p3, vdupq_n_u8(0), 8);       // srli_si128(p3, 8)
    uint8x16_t temp_shift_byte = vextq_u8(vdupq_n_u8(0), p3_h, 16 - 4); // slli_si128(p3_h, 4)
    uint8x16_t p2 = veorq_u8(p3, temp_shift_byte);          // p3 ^ shift
    
    // 2 x div s2 = x^4 - x
    // __m256i p2_h  = _mm256_srli_epi64(p2,32);
    // __m256i p1 = p2 ^ _mm256_slli_epi64(p2_h,8);
    uint64x2_t p2_u64 = vreinterpretq_u64_u8(p2);
    uint64x2_t p2_h_u64 = vshrq_n_u64(p2_u64, 32);         // srli_epi64(p2, 32)
    uint64x2_t temp_shift64 = vshlq_n_u64(p2_h_u64, 8);     // slli_epi64(p2_h, 8)
    uint64x2_t p1_u64 = veorq_u64(p2_u64, temp_shift64);     // p2 ^ shift
    uint8x16_t p1 = vreinterpretq_u8_u64(p1_u64);

    // div s2^2 = ( x^8 - x^2 )
    // __m256i p1_h = _mm256_srli_si256( p1 , 8 );
    // __m256i p0   = p1 ^ _mm256_slli_si256(p1_h,2);
    uint8x16_t p1_h = vextq_u8(p1, vdupq_n_u8(0), 8);       // srli_si128(p1, 8)
    uint8x16_t temp_shift_byte2 = vextq_u8(vdupq_n_u8(0), p1_h, 16 - 2); // slli_si128(p1_h, 2)
    uint8x16_t p0 = veorq_u8(p1, temp_shift_byte2);         // p1 ^ shift
    
    vst1q_u8(poly,p0);
}



static inline
void div_blk_128( uint8_t *poly, int si_h, int si_l, int polylen )
{
  int deg_diff = si_h-si_l;
  for(int i=polylen-1;i>=si_h;i--) {
      uint8x16_t d = vld1q_u8(poly+(i-deg_diff)*16);
      uint8x16_t p = vld1q_u8(poly+i*16);
      d ^= p;
      vst1q_u8( poly+(i-deg_diff)*16 , d );
      //poly[i-deg_diff] ^= poly[i];
  }
}

static inline
void idiv_blk_128( uint8_t *poly, int si_h, int si_l, int polylen )
{
  int deg_diff = si_h-si_l;
  for(int i=si_h;i<polylen;i++) {
      uint8x16_t d = vld1q_u8(poly+(i-deg_diff)*16);
      uint8x16_t p = vld1q_u8(poly+i*16);
      d ^= p;
      vst1q_u8( poly+(i-deg_diff)*16 , d );
      //poly[i-deg_diff] ^= poly[i];
  }
}

// s4 = x^16-x
static inline void div_s4_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));

    // __m128i r1 = p1 ^ _mm_srli_si128(p1, 15);
    uint8x16_t p1_shr15 = vextq_u8(p1, vdupq_n_u8(0), 15);
    uint8x16_t r1 = veorq_u8(p1, p1_shr15);
    // __m128i r0 = p0 ^ _mm_slli_si128(r1, 1);
    uint8x16_t r1_shl1 = vextq_u8(vdupq_n_u8(0), r1, 16 - 1);
    uint8x16_t r0 = veorq_u8(p0, r1_shl1);

    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
}

static inline void idiv_s4_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));

    // __m128i r0 = p0 ^ _mm_slli_si128(p1, 1);
    uint8x16_t p1_shl1 = vextq_u8(vdupq_n_u8(0), p1, 16 - 1);
    uint8x16_t r0 = veorq_u8(p0, p1_shl1);
    // __m128i r1 = p1 ^ _mm_srli_si128(p1, 15);
    uint8x16_t p1_shr15 = vextq_u8(p1, vdupq_n_u8(0), 15);
    uint8x16_t r1 = veorq_u8(p1, p1_shr15);

    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
}

// s4sq = x^32-x2
static inline void div_s4sq_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));
    uint8x16_t p2 = vld1q_u8((uint8_t const*)(poly + 32));
    uint8x16_t p3 = vld1q_u8((uint8_t const*)(poly + 48));
    
    // __m128i r2 = p2 ^ _mm_srli_si128(p3, 14);
    uint8x16_t p3_shr14 = vextq_u8(p3, vdupq_n_u8(0), 14);
    uint8x16_t r2 = veorq_u8(p2, p3_shr14);
    
    // __m128i r1 = p1 ^ _mm_alignr_epi8(p3, r2, 14);
    uint8x16_t aligned_p3_r2 = vextq_u8(r2, p3, 14);
    uint8x16_t r1 = veorq_u8(p1, aligned_p3_r2);
    
    // __m128i r0 = p0 ^ _mm_slli_si128(r2, 2);
    uint8x16_t r2_shl2 = vextq_u8(vdupq_n_u8(0), r2, 16 - 2);
    uint8x16_t r0 = veorq_u8(p0, r2_shl2);
    
    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
    vst1q_u8((uint8_t*)(poly + 32), r2);
}

// s4sq = x^32-x2
static inline void idiv_s4sq_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));
    uint8x16_t p2 = vld1q_u8((uint8_t const*)(poly + 32));
    uint8x16_t p3 = vld1q_u8((uint8_t const*)(poly + 48));

    // __m128i r0 = p0 ^ _mm_slli_si128(p2, 2);
    uint8x16_t p2_shl2 = vextq_u8(vdupq_n_u8(0), p2, 16 - 2);
    uint8x16_t r0 = veorq_u8(p0, p2_shl2);
    // __m128i r1 = p1 ^ _mm_alignr_epi8(p3, p2, 14);
    uint8x16_t aligned_p3_p2 = vextq_u8(p2, p3, 14);
    uint8x16_t r1 = veorq_u8(p1, aligned_p3_p2);
    // __m128i r2 = p2 ^ _mm_srli_si128(p3, 14);
    uint8x16_t p3_shr14 = vextq_u8(p3, vdupq_n_u8(0), 14); // p3 右移 14 bytes
    uint8x16_t r2 = veorq_u8(p2, p3_shr14);

    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
    vst1q_u8((uint8_t*)(poly + 32), r2);
}

// s4tr = x^64-x4
static inline void div_s4tr_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));
    uint8x16_t p2 = vld1q_u8((uint8_t const*)(poly + 32));
    uint8x16_t p3 = vld1q_u8((uint8_t const*)(poly + 48));
    uint8x16_t p4 = vld1q_u8((uint8_t const*)(poly + 64));
    uint8x16_t p5 = vld1q_u8((uint8_t const*)(poly + 80));
    uint8x16_t p6 = vld1q_u8((uint8_t const*)(poly + 96));
    uint8x16_t p7 = vld1q_u8((uint8_t const*)(poly + 112));

    uint8x16_t r4 = p4 ^ vextq_u8(p7, vdupq_n_u8(0), 12); // p7 右移 12 bytes
    uint8x16_t r3 = p3 ^ vextq_u8(p6, p7, 12);
    uint8x16_t r2 = p2 ^ vextq_u8(p5, p6, 12);
    uint8x16_t r1 = p1 ^ vextq_u8(r4, p5, 12);
    uint8x16_t r0 = p0 ^ vextq_u8(vdupq_n_u8(0), r4, 12);

    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
    vst1q_u8((uint8_t*)(poly + 32), r2);
    vst1q_u8((uint8_t*)(poly + 48), r3);
    vst1q_u8((uint8_t*)(poly + 64), r4);
}

// s4tr = x^64-x4
static inline void idiv_s4tr_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));
    uint8x16_t p2 = vld1q_u8((uint8_t const*)(poly + 32));
    uint8x16_t p3 = vld1q_u8((uint8_t const*)(poly + 48));
    uint8x16_t p4 = vld1q_u8((uint8_t const*)(poly + 64));
    uint8x16_t p5 = vld1q_u8((uint8_t const*)(poly + 80));
    uint8x16_t p6 = vld1q_u8((uint8_t const*)(poly + 96));
    uint8x16_t p7 = vld1q_u8((uint8_t const*)(poly + 112));

    uint8x16_t r0 = p0 ^ vextq_u8(vdupq_n_u8(0), p4, 12);
    uint8x16_t r1 = p1 ^ vextq_u8(p4, p5, 12);
    uint8x16_t r2 = p2 ^ vextq_u8(p5, p6, 12);
    uint8x16_t r3 = p3 ^ vextq_u8(p6, p7, 12);
    uint8x16_t r4 = p4 ^ vextq_u8(p7, vdupq_n_u8(0), 12);

    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
    vst1q_u8((uint8_t*)(poly + 32), r2);
    vst1q_u8((uint8_t*)(poly + 48), r3);
    vst1q_u8((uint8_t*)(poly + 64), r4);
}

// s4qu = x^128-x8
static inline void div_s4qu_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));
    uint8x16_t p2 = vld1q_u8((uint8_t const*)(poly + 32));
    uint8x16_t p3 = vld1q_u8((uint8_t const*)(poly + 48));
    uint8x16_t p4 = vld1q_u8((uint8_t const*)(poly + 64));
    uint8x16_t p5 = vld1q_u8((uint8_t const*)(poly + 80));
    uint8x16_t p6 = vld1q_u8((uint8_t const*)(poly + 96));
    uint8x16_t p7 = vld1q_u8((uint8_t const*)(poly + 112));
    uint8x16_t p8 = vld1q_u8((uint8_t const*)(poly + 128));
    uint8x16_t p9 = vld1q_u8((uint8_t const*)(poly + 144));
    uint8x16_t p10 = vld1q_u8((uint8_t const*)(poly + 160));
    uint8x16_t p11 = vld1q_u8((uint8_t const*)(poly + 176));
    uint8x16_t p12 = vld1q_u8((uint8_t const*)(poly + 192));
    uint8x16_t p13 = vld1q_u8((uint8_t const*)(poly + 208));
    uint8x16_t p14 = vld1q_u8((uint8_t const*)(poly + 224));
    uint8x16_t p15 = vld1q_u8((uint8_t const*)(poly + 240));

    uint8x16_t r8 = p8 ^ vextq_u8(p15, vdupq_n_u8(0), 8);
    uint8x16_t r7 = p7 ^ vextq_u8(p14, p15, 8);
    uint8x16_t r6 = p6 ^ vextq_u8(p13, p14, 8);
    uint8x16_t r5 = p5 ^ vextq_u8(p12, p13, 8);
    uint8x16_t r4 = p4 ^ vextq_u8(p11, p12, 8);
    uint8x16_t r3 = p3 ^ vextq_u8(p10, p11, 8);
    uint8x16_t r2 = p2 ^ vextq_u8(p9, p10, 8);
    uint8x16_t r1 = p1 ^ vextq_u8(r8, p9, 8);
    uint8x16_t r0 = p0 ^ vextq_u8(vdupq_n_u8(0), r8, 8);

    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
    vst1q_u8((uint8_t*)(poly + 32), r2);
    vst1q_u8((uint8_t*)(poly + 48), r3);
    vst1q_u8((uint8_t*)(poly + 64), r4);
    vst1q_u8((uint8_t*)(poly + 80), r5);
    vst1q_u8((uint8_t*)(poly + 96), r6);
    vst1q_u8((uint8_t*)(poly + 112), r7);
    vst1q_u8((uint8_t*)(poly + 128), r8);
}

// s4qu = x^128-x8
static inline void idiv_s4qu_byte( uint8_t * poly )
{
    uint8x16_t p0 = vld1q_u8((uint8_t const*)poly);
    uint8x16_t p1 = vld1q_u8((uint8_t const*)(poly + 16));
    uint8x16_t p2 = vld1q_u8((uint8_t const*)(poly + 32));
    uint8x16_t p3 = vld1q_u8((uint8_t const*)(poly + 48));
    uint8x16_t p4 = vld1q_u8((uint8_t const*)(poly + 64));
    uint8x16_t p5 = vld1q_u8((uint8_t const*)(poly + 80));
    uint8x16_t p6 = vld1q_u8((uint8_t const*)(poly + 96));
    uint8x16_t p7 = vld1q_u8((uint8_t const*)(poly + 112));
    uint8x16_t p8 = vld1q_u8((uint8_t const*)(poly + 128));
    uint8x16_t p9 = vld1q_u8((uint8_t const*)(poly + 144));
    uint8x16_t p10 = vld1q_u8((uint8_t const*)(poly + 160));
    uint8x16_t p11 = vld1q_u8((uint8_t const*)(poly + 176));
    uint8x16_t p12 = vld1q_u8((uint8_t const*)(poly + 192));
    uint8x16_t p13 = vld1q_u8((uint8_t const*)(poly + 208));
    uint8x16_t p14 = vld1q_u8((uint8_t const*)(poly + 224));
    uint8x16_t p15 = vld1q_u8((uint8_t const*)(poly + 240));

    uint8x16_t r0 = p0 ^ vextq_u8(vdupq_n_u8(0), p8, 8);
    uint8x16_t r1 = p1 ^ vextq_u8(p8, p9, 8);
    uint8x16_t r2 = p2 ^ vextq_u8(p9, p10, 8);
    uint8x16_t r3 = p3 ^ vextq_u8(p10, p11, 8);
    uint8x16_t r4 = p4 ^ vextq_u8(p11, p12, 8);
    uint8x16_t r5 = p5 ^ vextq_u8(p12, p13, 8);
    uint8x16_t r6 = p6 ^ vextq_u8(p13, p14, 8);
    uint8x16_t r7 = p7 ^ vextq_u8(p14, p15, 8);
    uint8x16_t r8 = p8 ^ vextq_u8(p15, vdupq_n_u8(0), 8);

    vst1q_u8((uint8_t*)poly, r0);
    vst1q_u8((uint8_t*)(poly + 16), r1);
    vst1q_u8((uint8_t*)(poly + 32), r2);
    vst1q_u8((uint8_t*)(poly + 48), r3);
    vst1q_u8((uint8_t*)(poly + 64), r4);
    vst1q_u8((uint8_t*)(poly + 80), r5);
    vst1q_u8((uint8_t*)(poly + 96), r6);
    vst1q_u8((uint8_t*)(poly + 112), r7);
    vst1q_u8((uint8_t*)(poly + 128), r8);
}


// s4 = x^16 - x
// div   s4^i , s4^(i-1) , ... , s4^1
static inline void repr_s4( uint8_t * poly , unsigned n_8 )
{
  int log_n = __builtin_ctz( n_8 );
  // no processing log_n <= 5, n_8 <= 32
  int i = log_n-4-1;
  for(;i>=4;i-- ){
    unsigned s_h_128 = 1<<(i+4-4);
    unsigned s_l_128 = 1<<(i-4);
    unsigned len_128 = 1<<(i+5-4);
    unsigned len = 1<<(i+5);
    // div X^{s_h=16*s_l} - X^{s_l}
    for(unsigned j=0;j<n_8;j+=len) { div_blk_128( poly+j , s_h_128 , s_l_128 , len_128 ); }
  }
  if (i>=3) { for(unsigned j=0;j<n_8;j+=256) { div_s4qu_byte( poly+j); } }
  if (i>=2) { for(unsigned j=0;j<n_8;j+=128) { div_s4tr_byte( poly+j); } }
  if (i>=1) { for(unsigned j=0;j<n_8;j+=64) { div_s4sq_byte( poly+j); } }
  for(unsigned j=0;j<n_8;j+=32) { div_s4_byte( poly+j); }
}

static inline void irepr_s4( uint8_t * poly , unsigned n_8 )
{
  int log_n = __builtin_ctz( n_8 );
  // no processing log_n <= 5, n_8 <= 32
  for(unsigned j=0;j<n_8;j+=32) { idiv_s4_byte( poly+j); }
  if (log_n>5) { for(unsigned j=0;j<n_8;j+=64) { idiv_s4sq_byte( poly+j); } }
  if (log_n>6) { for(unsigned j=0;j<n_8;j+=128) { idiv_s4tr_byte( poly+j); } }
  if (log_n>7) { for(unsigned j=0;j<n_8;j+=256) { idiv_s4qu_byte( poly+j); } }
  for(int i=4;i<log_n-4;i++) {
    unsigned s_h_128 = 1<<(i+4-4);
    unsigned s_l_128 = 1<<(i-4);
    unsigned len_128 = 1<<(i+5-4);
    unsigned len = 1<<(i+5);
    // div X^{s_h=16*s_l} - X^{s_l}
    for(unsigned j=0;j<n_8;j+=len) { idiv_blk_128( poly+j , s_h_128 , s_l_128 , len_128 ); }
  }
}

#include "bc_128.h"

void bc_8( uint8_t * poly , unsigned n_8 )
{
  if(16>n_8) { cvt( poly , n_8 , 0 , __builtin_ctz(n_8) ); return; }
  if(32>n_8) { bc_8_128( poly ); return; }

  repr_s4( poly , n_8 );
  for(unsigned i=0;i<n_8;i+=16) { bc_8_128( poly + i ); }
  bc_128( poly , n_8>>4 );
}

////////


void ibc_8( uint8_t * poly , unsigned n_8 )
{
  //icvt( poly , n_8 , 0 , __builtin_ctz(n_8) ); return;
  if(16>n_8) { icvt( poly , n_8 , 0 , __builtin_ctz(n_8) ); return; }
  if(32>n_8) { ibc_8_128( poly ); return; }

  ibc_128( poly , n_8>>4 );
  for(unsigned i=0;i<n_8;i+=16) { ibc_8_128( poly + i ); }
  irepr_s4( poly , n_8 );
}


