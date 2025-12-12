
#include "bc_1.h"


#include <emmintrin.h>
#include <immintrin.h>

///
/// XXX: use un-alinged load/store to support unaligned memory access.
///

static inline
void bc_1_256x4( __m256i *poly )
{
  __m256i r0 = _mm256_loadu_si256( poly+0 );
  __m256i r1 = _mm256_loadu_si256( poly+1 );
  __m256i r2 = _mm256_loadu_si256( poly+2 );
  __m256i r3 = _mm256_loadu_si256( poly+3 );

//layer 13: [256] s4^8:(128,8) suggest unit:1
  {
    __m256i r0r1_h = _mm256_permute2x128_si256( r0, r1, 0x31 );
    __m256i r2r3_h = _mm256_permute2x128_si256( r2, r3, 0x31 );
    __m256i r0r1_red_h = _mm256_srli_si256(r0r1_h,15);
    __m256i r0r1_red_l = _mm256_slli_si256(r0r1_red_h^r0r1_h,1); // some terms reduce twice
    __m256i r2r3_red_h = _mm256_srli_si256(r2r3_h,15);
    __m256i r2r3_red_l = _mm256_slli_si256(r2r3_red_h^r2r3_h,1); // some terms reduce twice

    r0 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x20 );
    r1 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x31 );
    r2 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x20 );
    r3 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x31 );
  }
//layer 14: [128] s4^4:(64,4) suggest unit:1
  {
    __m256i mask = _mm256_set_epi32( 0xf0000000,0,0,0, 0xf0000000,0,0,0 );
    r0 ^= _mm256_srli_epi64( mask&r0 , 60 );
    r1 ^= _mm256_srli_epi64( mask&r1 , 60 );
    r2 ^= _mm256_srli_epi64( mask&r2 , 60 );
    r3 ^= _mm256_srli_epi64( mask&r3 , 60 );
    r0 ^= _mm256_slli_epi64( _mm256_srli_si256(r0,8) , 4 );
    r1 ^= _mm256_slli_epi64( _mm256_srli_si256(r1,8) , 4 );
    r2 ^= _mm256_slli_epi64( _mm256_srli_si256(r2,8) , 4 );
    r3 ^= _mm256_slli_epi64( _mm256_srli_si256(r3,8) , 4 );
  }
//layer 15: [64] s4^2:(32,2) suggest unit:1
  {
    __m256i mask = _mm256_set_epi32( 0xc0000000,0, 0xc0000000,0, 0xc0000000,0, 0xc0000000,0 );
    r0 ^= _mm256_srli_epi32( mask&r0 , 30 );
    r1 ^= _mm256_srli_epi32( mask&r1 , 30 );
    r2 ^= _mm256_srli_epi32( mask&r2 , 30 );
    r3 ^= _mm256_srli_epi32( mask&r3 , 30 );
    r0 ^= _mm256_slli_epi32( _mm256_srli_epi64(r0,32) , 2 );
    r1 ^= _mm256_slli_epi32( _mm256_srli_epi64(r1,32) , 2 );
    r2 ^= _mm256_slli_epi32( _mm256_srli_epi64(r2,32) , 2 );
    r3 ^= _mm256_slli_epi32( _mm256_srli_epi64(r3,32) , 2 );
  }
//layer 16: [32] s4^1:(16,1) suggest unit:1
  {
    __m256i mask = _mm256_set1_epi32( 0x80000000 );
    r0 ^= _mm256_srli_epi16( mask&r0 , 15 );
    r1 ^= _mm256_srli_epi16( mask&r1 , 15 );
    r2 ^= _mm256_srli_epi16( mask&r2 , 15 );
    r3 ^= _mm256_srli_epi16( mask&r3 , 15 );
    r0 ^= _mm256_slli_epi16( _mm256_srli_epi32(r0,16) , 1 );
    r1 ^= _mm256_slli_epi16( _mm256_srli_epi32(r1,16) , 1 );
    r2 ^= _mm256_slli_epi16( _mm256_srli_epi32(r2,16) , 1 );
    r3 ^= _mm256_slli_epi16( _mm256_srli_epi32(r3,16) , 1 );
  }
//layer 17: [256] s2^2:(128,32) suggest unit:16
  {
    __m256i r0r1_h = _mm256_permute2x128_si256( r0, r1, 0x31 );
    __m256i r2r3_h = _mm256_permute2x128_si256( r2, r3, 0x31 );
    __m256i r0r1_red_h = _mm256_srli_si256(r0r1_h,12);
    __m256i r0r1_red_l = _mm256_slli_si256(r0r1_red_h^r0r1_h,4); // some terms reduce twice
    __m256i r2r3_red_h = _mm256_srli_si256(r2r3_h,12);
    __m256i r2r3_red_l = _mm256_slli_si256(r2r3_red_h^r2r3_h,4); // some terms reduce twice

    r0 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x20 );
    r1 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x31 );
    r2 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x20 );
    r3 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x31 );
  }
//layer 18: [128] s2^1:(64,16) suggest unit:16
  {
    __m256i mask = _mm256_set_epi32( 0xffff0000,0,0,0, 0xffff0000,0,0,0 );
    r0 ^= _mm256_srli_epi64( mask&r0 , 48 );
    r1 ^= _mm256_srli_epi64( mask&r1 , 48 );
    r2 ^= _mm256_srli_epi64( mask&r2 , 48 );
    r3 ^= _mm256_srli_epi64( mask&r3 , 48 );
    r0 ^= _mm256_slli_epi64( _mm256_srli_si256(r0,8) , 16 );
    r1 ^= _mm256_slli_epi64( _mm256_srli_si256(r1,8) , 16 );
    r2 ^= _mm256_slli_epi64( _mm256_srli_si256(r2,8) , 16 );
    r3 ^= _mm256_slli_epi64( _mm256_srli_si256(r3,8) , 16 );
  }
//layer 19: [256] s1^1:(128,64) suggest unit:64
  {
    __m256i r0r1_h = _mm256_permute2x128_si256( r0, r1, 0x31 );
    __m256i r2r3_h = _mm256_permute2x128_si256( r2, r3, 0x31 );
    __m256i r0r1_red_h = _mm256_srli_si256(r0r1_h,8);
    __m256i r0r1_red_l = _mm256_slli_si256(r0r1_red_h^r0r1_h,8); // some terms reduce twice
    __m256i r2r3_red_h = _mm256_srli_si256(r2r3_h,8);
    __m256i r2r3_red_l = _mm256_slli_si256(r2r3_red_h^r2r3_h,8); // some terms reduce twice

    r0 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x20 );
    r1 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x31 );
    r2 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x20 );
    r3 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x31 );
  }
//layer 20: [64] s1^1:(32,16) suggest unit:16
  {
    __m256i mask = _mm256_set_epi32( 0xffff0000,0, 0xffff0000,0, 0xffff0000,0, 0xffff0000,0 );
    r0 ^= _mm256_srli_epi32( mask&r0 , 16 );
    r1 ^= _mm256_srli_epi32( mask&r1 , 16 );
    r2 ^= _mm256_srli_epi32( mask&r2 , 16 );
    r3 ^= _mm256_srli_epi32( mask&r3 , 16 );
    r0 ^= _mm256_slli_epi32( _mm256_srli_epi64(r0,32) , 16 );
    r1 ^= _mm256_slli_epi32( _mm256_srli_epi64(r1,32) , 16 );
    r2 ^= _mm256_slli_epi32( _mm256_srli_epi64(r2,32) , 16 );
    r3 ^= _mm256_slli_epi32( _mm256_srli_epi64(r3,32) , 16 );
  }
// [16]: s3: s3 = x^8 + x^4 + x^2 + x   // 0x116
  {
    __m256i mask = _mm256_set1_epi16(0xff00);
    __m256i mat10 = _mm256_set1_epi64x(0xd173e61d3a74e8ULL);
    __m256i mat11 = _mm256_set1_epi64x(0xd1a2448810204080ULL);

    r0 = _mm256_andnot_si256(mask,r0) ^ _mm256_gf2p8affine_epi64_epi8( r0&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r0,8) , mat10 ,0);
    r1 = _mm256_andnot_si256(mask,r1) ^ _mm256_gf2p8affine_epi64_epi8( r1&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r1,8) , mat10 ,0);
    r2 = _mm256_andnot_si256(mask,r2) ^ _mm256_gf2p8affine_epi64_epi8( r2&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r2,8) , mat10 ,0);
    r3 = _mm256_andnot_si256(mask,r3) ^ _mm256_gf2p8affine_epi64_epi8( r3&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r3,8) , mat10 ,0);
  }
  // s2 , s1
  {
    __m256i mat0 = _mm256_set1_epi64x(0x1fe6c4890e0c080ULL);
    r0 = _mm256_gf2p8affine_epi64_epi8( r0 , mat0 ,0);
    r1 = _mm256_gf2p8affine_epi64_epi8( r1 , mat0 ,0);
    r2 = _mm256_gf2p8affine_epi64_epi8( r2 , mat0 ,0);
    r3 = _mm256_gf2p8affine_epi64_epi8( r3 , mat0 ,0);
  }

  _mm256_storeu_si256( poly+0 , r0 );
  _mm256_storeu_si256( poly+1 , r1 );
  _mm256_storeu_si256( poly+2 , r2 );
  _mm256_storeu_si256( poly+3 , r3 );
}


static inline
void ibc_1_256x4( __m256i *poly )
{
  __m256i r0 = _mm256_loadu_si256( poly+0 );
  __m256i r1 = _mm256_loadu_si256( poly+1 );
  __m256i r2 = _mm256_loadu_si256( poly+2 );
  __m256i r3 = _mm256_loadu_si256( poly+3 );
  // s1, s2
  {
    __m256i mat0 = _mm256_set1_epi64x(0x1166cc89060c080ULL);
    r0 = _mm256_gf2p8affine_epi64_epi8( r0 , mat0 ,0);
    r1 = _mm256_gf2p8affine_epi64_epi8( r1 , mat0 ,0);
    r2 = _mm256_gf2p8affine_epi64_epi8( r2 , mat0 ,0);
    r3 = _mm256_gf2p8affine_epi64_epi8( r3 , mat0 ,0);
  }
// [16]: s3: s3 = x^8 + x^4 + x^2 + x   // 0x116
  {
    __m256i mask = _mm256_set1_epi16(0xff00);
    __m256i mat10 = _mm256_set1_epi64x(0x103060d1a3468ULL);
    __m256i mat11 = _mm256_set1_epi64x(0xd1a2448810204080ULL);
    r0 = _mm256_andnot_si256(mask,r0) ^ _mm256_gf2p8affine_epi64_epi8( r0&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r0,8) , mat10 ,0);
    r1 = _mm256_andnot_si256(mask,r1) ^ _mm256_gf2p8affine_epi64_epi8( r1&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r1,8) , mat10 ,0);
    r2 = _mm256_andnot_si256(mask,r2) ^ _mm256_gf2p8affine_epi64_epi8( r2&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r2,8) , mat10 ,0);
    r3 = _mm256_andnot_si256(mask,r3) ^ _mm256_gf2p8affine_epi64_epi8( r3&mask , mat11 ,0) ^ _mm256_gf2p8affine_epi64_epi8( _mm256_srli_epi16(r3,8) , mat10 ,0);
  }
//layer 3: [64] s1^1:(32,16) suggest unit:16
  {
    __m256i mask = _mm256_set_epi32( 0xffffffff,0, 0xffffffff,0, 0xffffffff,0, 0xffffffff,0  );
    r0 ^= _mm256_srli_epi64( mask&r0 , 16 );
    r1 ^= _mm256_srli_epi64( mask&r1 , 16 );
    r2 ^= _mm256_srli_epi64( mask&r2 , 16 );
    r3 ^= _mm256_srli_epi64( mask&r3 , 16 );
  }
//layer 2: [256] s1^1:(128,64) suggest unit:64
  {
    __m256i r0r1_h = _mm256_permute2x128_si256( r0, r1, 0x31 );
    __m256i r2r3_h = _mm256_permute2x128_si256( r2, r3, 0x31 );
    __m256i r0r1_red_h = _mm256_srli_si256(r0r1_h,8);
    __m256i r0r1_red_l = _mm256_slli_si256(r0r1_h,8);
    __m256i r2r3_red_h = _mm256_srli_si256(r2r3_h,8);
    __m256i r2r3_red_l = _mm256_slli_si256(r2r3_h,8);

    r0 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x20 );
    r1 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x31 );
    r2 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x20 );
    r3 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x31 );
  }
//layer 1: [128] s2^1:(64,16) suggest unit:16
  {
    __m256i mask = _mm256_set_epi32( 0xffffffff,0xffffffff,0,0, 0xffffffff,0xffffffff,0,0  );
    r0 ^= _mm256_srli_si256( mask&r0 , 6 );
    r1 ^= _mm256_srli_si256( mask&r1 , 6 );
    r2 ^= _mm256_srli_si256( mask&r2 , 6 );
    r3 ^= _mm256_srli_si256( mask&r3 , 6 );
  }
//layer 0: [256] s2^2:(128,32) suggest unit:16
  {
    __m256i r0r1_h = _mm256_permute2x128_si256( r0, r1, 0x31 );
    __m256i r2r3_h = _mm256_permute2x128_si256( r2, r3, 0x31 );
    __m256i r0r1_red_h = _mm256_srli_si256(r0r1_h,12);
    __m256i r0r1_red_l = _mm256_slli_si256(r0r1_h,4);
    __m256i r2r3_red_h = _mm256_srli_si256(r2r3_h,12);
    __m256i r2r3_red_l = _mm256_slli_si256(r2r3_h,4);

    r0 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x20 );
    r1 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x31 );
    r2 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x20 );
    r3 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x31 );
  }

//layer 16: [32] s4^1:(16,1) suggest unit:1
  {
    __m256i mask = _mm256_set1_epi32( 0xffff0000 );
    r0 ^= _mm256_srli_epi32( mask&r0 , 15 );
    r1 ^= _mm256_srli_epi32( mask&r1 , 15 );
    r2 ^= _mm256_srli_epi32( mask&r2 , 15 );
    r3 ^= _mm256_srli_epi32( mask&r3 , 15 );
  }
//layer 15: [64] s4^2:(32,2) suggest unit:1
  {
    __m256i mask = _mm256_set_epi32( 0xffffffff,0, 0xffffffff,0, 0xffffffff,0, 0xffffffff,0  );
    r0 ^= _mm256_srli_epi64( mask&r0 , 30 );
    r1 ^= _mm256_srli_epi64( mask&r1 , 30 );
    r2 ^= _mm256_srli_epi64( mask&r2 , 30 );
    r3 ^= _mm256_srli_epi64( mask&r3 , 30 );
  }
//layer 14: [128] s4^4:(64,4) suggest unit:1
  {
    __m256i mask = _mm256_set_epi32( 0xffffffff,0xffffffff,0,0, 0xffffffff,0xffffffff,0,0  );
    __m256i r0h = mask&r0;
    __m256i r1h = mask&r1;
    __m256i r2h = mask&r2;
    __m256i r3h = mask&r3;

    r0 ^= _mm256_srli_epi64( r0h , 60 ) ^ _mm256_slli_epi64(_mm256_srli_si256(r0,8),4);
    r1 ^= _mm256_srli_epi64( r1h , 60 ) ^ _mm256_slli_epi64(_mm256_srli_si256(r1,8),4);
    r2 ^= _mm256_srli_epi64( r2h , 60 ) ^ _mm256_slli_epi64(_mm256_srli_si256(r2,8),4);
    r3 ^= _mm256_srli_epi64( r3h , 60 ) ^ _mm256_slli_epi64(_mm256_srli_si256(r3,8),4);
  }
//layer 13: [256] s4^8:(128,8) suggest unit:1
  {
    __m256i r0r1_h = _mm256_permute2x128_si256( r0, r1, 0x31 );
    __m256i r2r3_h = _mm256_permute2x128_si256( r2, r3, 0x31 );
    __m256i r0r1_red_h = _mm256_srli_si256(r0r1_h,15);
    __m256i r0r1_red_l = _mm256_slli_si256(r0r1_h,1);
    __m256i r2r3_red_h = _mm256_srli_si256(r2r3_h,15);
    __m256i r2r3_red_l = _mm256_slli_si256(r2r3_h,1);

    r0 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x20 );
    r1 ^= _mm256_permute2x128_si256( r0r1_red_l , r0r1_red_h , 0x31 );
    r2 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x20 );
    r3 ^= _mm256_permute2x128_si256( r2r3_red_l , r2r3_red_h , 0x31 );
  }

  _mm256_storeu_si256( poly+0 , r0 );
  _mm256_storeu_si256( poly+1 , r1 );
  _mm256_storeu_si256( poly+2 , r2 );
  _mm256_storeu_si256( poly+3 , r3 );
}






//////////////////////////////////////////


void bc_1_256( void * poly , unsigned n_256bit )
{
	__m256i * fx_256 = (__m256i*) poly;
  while( n_256bit >= 4 ) {
    bc_1_256x4( fx_256 );
    fx_256 += 4;
    n_256bit -= 4;
  }
  if( 0 < n_256bit ) {
    __m256i tmp[4];
    for(unsigned i=0;i<n_256bit;i++) tmp[i] = _mm256_loadu_si256( fx_256+i );
    for(unsigned i=n_256bit;i<4;i++) tmp[i] = _mm256_setzero_si256();
    bc_1_256x4( tmp );
    for(unsigned i=0;i<n_256bit;i++) _mm256_storeu_si256( fx_256+i , tmp[i] );
  }
}


void ibc_1_256( void * poly , unsigned n_256bit )
{
	__m256i * fx_256 = (__m256i*) poly;
  while( n_256bit >= 4 ) {
    ibc_1_256x4( fx_256 );
    fx_256 += 4;
    n_256bit -= 4;
  }
  if( 0 < n_256bit ) {
    __m256i tmp[4];
    for(unsigned i=0;i<n_256bit;i++) tmp[i] = _mm256_loadu_si256( fx_256+i );
    for(unsigned i=n_256bit;i<4;i++) tmp[i] = _mm256_setzero_si256();
    ibc_1_256x4( tmp );
    for(unsigned i=0;i<n_256bit;i++) _mm256_storeu_si256( fx_256+i , tmp[i] );
  }
}









//////////////////////////////////////////////////////////////////





static inline
void div_s8_p128_256x256bit( __m256i * poly )
{
  // [65536]/(32768,128)
  #define max_idx  255
  #define width    128
  ///
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256( poly+width , _mm256_loadu_si256(poly+width)^_maxYmmH );
  __m256i m0_1;
  for(int i=(width-4);i>=0;i-=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );
    __m256i m01 = _mm256_loadu_si256( poly+width-1+i );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    m0_1 = _mm256_permute2x128_si256( m01 , m0 , 0x21 );

    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_1 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_1 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_1 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_1 );
  }
  ///
  _mm256_storeu_si256(poly,_mm256_loadu_si256(poly)^_mm256_blend_epi32(_mm256_setzero_si256(),m0_1,0xf));
  #undef  max_idx
  #undef  width
}

static inline
void div_s8_p64_128x256bit( __m256i * poly )
{
  // [32768]/(16384,64)
  #define max_idx 127
  #define width   64
  #define sr_byte 8
  ///
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  //poly[width] ^= _mm256_srli_si256( _maxYmmH , sr_byte );
  _mm256_storeu_si256( poly+width , _mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte) );
  __m256i m0_2;
  for(int i=(width-4);i>=0;i-=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );
    __m256i m01 = _mm256_loadu_si256( poly+width-1+i );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m0_1 = _mm256_permute2x128_si256( m01 , m0 , 0x21 );
    m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );
  }
  ///
  _mm256_storeu_si256(poly,_mm256_loadu_si256(poly)^_mm256_blend_epi32(_mm256_setzero_si256(),m0_2,0x3));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void div_s8_p32_64x256bit( __m256i * poly )
{
  // [16384]/(8192,32)
  #define max_idx 63
  #define width   32
  #define sr_byte 12
  ///
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte));
  __m256i m0_2;
  for(int i=(width-4);i>=0;i-=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );
    __m256i m01 = _mm256_loadu_si256( poly+width-1+i );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m0_1 = _mm256_permute2x128_si256( m01 , m0 , 0x21 );
    m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );
  }
  ///
  _mm256_storeu_si256(poly,_mm256_loadu_si256(poly)^_mm256_blend_epi32(_mm256_setzero_si256(),m0_2,1));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void div_s8_p16_32x256bit( __m256i * poly )
{
  // [8192]/(4096,16)
  #define max_idx  31
  #define width    16
  #define sr_byte  14
  ///
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte));
  __m256i m0_2;
  for(int i=(width-4);i>=0;i-=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );
    __m256i m01 = _mm256_loadu_si256( poly+width-1+i );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m0_1 = _mm256_permute2x128_si256( m01 , m0 , 0x21 );
    m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );

  }
  ///
  _mm256_storeu_si256(poly,_mm256_loadu_si256(poly)^(m0_2&_mm256_set_epi32(0,0,0,0, 0,0,0,0xffff)));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void div_s8_p8_16x256bit( __m256i * poly )
{
  // [4096]/(2048,8)
  #define max_idx 15
  #define width   8
  #define sr_byte 15
  ///
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte));
  __m256i m0_2;
  for(int i=(width-4);i>=0;i-=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );
    __m256i m01 = _mm256_loadu_si256( poly+width-1+i );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m0_1 = _mm256_permute2x128_si256( m01 , m0 , 0x21 );
    m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );
  }
  ///
  _mm256_storeu_si256(poly,_mm256_loadu_si256(poly)^(m0_2&_mm256_set_epi32(0,0,0,0, 0,0,0,0xff)));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void s8div_bit_8x256bit( __m256i * poly )
{
  // only perform 3 layers:
  __m256i m0 = _mm256_loadu_si256( poly+0 );
  __m256i m1 = _mm256_loadu_si256( poly+1 );
  __m256i m2 = _mm256_loadu_si256( poly+2 );
  __m256i m3 = _mm256_loadu_si256( poly+3 );
  __m256i m4 = _mm256_loadu_si256( poly+4 );
  __m256i m5 = _mm256_loadu_si256( poly+5 );
  __m256i m6 = _mm256_loadu_si256( poly+6 );
  __m256i m7 = _mm256_loadu_si256( poly+7 );

  // [2048]/(1024,4)
#if 1
  {
    __m256i _m7h = _mm256_permute2x128_si256( m7, m7 , 0xf1 );
    m4 ^= _mm256_srli_epi16( _mm256_srli_si256(_m7h,15)  , 4);

    __m256i m7l_m6h = _mm256_permute2x128_si256( m6 , m7 , 0x21 );
    __m256i m7_m6_sr8 = _mm256_alignr_epi8( m7 , m7l_m6h , 15 );
    m3 ^= _mm256_slli_epi16(m7,4)|_mm256_srli_epi16(m7_m6_sr8,4);

    __m256i m6l_m5h = _mm256_permute2x128_si256( m5 , m6 , 0x21 );
    __m256i m6_m5_sr8 = _mm256_alignr_epi8( m6 , m6l_m5h , 15 );
    m2 ^= _mm256_slli_epi16(m6,4)|_mm256_srli_epi16(m6_m5_sr8,4);

    __m256i m5l_m4h = _mm256_permute2x128_si256( m4 , m5 , 0x21 );
    __m256i m5_m4_sr8 = _mm256_alignr_epi8( m5 , m5l_m4h , 15 );
    m1 ^= _mm256_slli_epi16(m5,4)|_mm256_srli_epi16(m5_m4_sr8,4);

    __m256i m4l_ = _mm256_permute2x128_si256( m4 , m4 , 0x2f );
    __m256i m4_sr8 = _mm256_alignr_epi8( m4 , m4l_ , 15 );
    m0 ^= _mm256_slli_epi16(m4,4)|_mm256_srli_epi16(m4_sr8,4);
  }
  // [1025]/(512,2)
  {
    __m256i _m7h = _mm256_permute2x128_si256( m7, m7 , 0xf1 );
    m6 ^= _mm256_srli_epi16( _mm256_srli_si256(_m7h,15)  , 6);

    __m256i m7l_m6h = _mm256_permute2x128_si256( m6 , m7 , 0x21 );
    __m256i m7_m6_sr8 = _mm256_alignr_epi8( m7 , m7l_m6h , 15 );
    m5 ^= _mm256_slli_epi16(m7,2)|_mm256_srli_epi16(m7_m6_sr8,6);

    __m256i m6l_ = _mm256_permute2x128_si256( m6 , m6 , 0x2f );
    __m256i m6_sr8 = _mm256_alignr_epi8( m6 , m6l_ , 15 );
    m4 ^= _mm256_slli_epi16(m6,2)|_mm256_srli_epi16(m6_sr8,6);

    __m256i _m3h = _mm256_permute2x128_si256( m3, m3 , 0xf1 );
    m2 ^= _mm256_srli_epi16( _mm256_srli_si256(_m3h,15)  , 6);

    __m256i m3l_m2h = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_m2_sr8 = _mm256_alignr_epi8( m3 , m3l_m2h , 15 );
    m1 ^= _mm256_slli_epi16(m3,2)|_mm256_srli_epi16(m3_m2_sr8,6);

    __m256i m2l_ = _mm256_permute2x128_si256( m2 , m2 , 0x2f );
    __m256i m2_sr8 = _mm256_alignr_epi8( m2 , m2l_ , 15 );
    m0 ^= _mm256_slli_epi16(m2,2)|_mm256_srli_epi16(m2_sr8,6);
  }
#endif
  // [512]/(256,1)
  {
    __m256i _m7h = _mm256_permute2x128_si256( m7, m7 , 0xf1 );
    m7 ^= _mm256_srli_epi16( _mm256_srli_si256(_m7h,15)  , 7);
    __m256i m7l_ = _mm256_permute2x128_si256( m7 , m7 , 0x0f );
    __m256i m7_sr8 = _mm256_alignr_epi8( m7 , m7l_ , 15 );
    m6 ^= _mm256_slli_epi16(m7,1)|_mm256_srli_epi16(m7_sr8,7);

    __m256i _m5h = _mm256_permute2x128_si256( m5, m5 , 0xf1 );
    m5 ^= _mm256_srli_epi16( _mm256_srli_si256(_m5h,15)  , 7);
    __m256i m5l_ = _mm256_permute2x128_si256( m5 , m5 , 0x0f );
    __m256i m5_sr8 = _mm256_alignr_epi8( m5 , m5l_ , 15 );
    m4 ^= _mm256_slli_epi16(m5,1)|_mm256_srli_epi16(m5_sr8,7);

    __m256i _m3h = _mm256_permute2x128_si256( m3, m3 , 0xf1 );
    m3 ^= _mm256_srli_epi16( _mm256_srli_si256(_m3h,15)  , 7);
    __m256i m3l_ = _mm256_permute2x128_si256( m3 , m3 , 0x0f );
    __m256i m3_sr8 = _mm256_alignr_epi8( m3 , m3l_ , 15 );
    m2 ^= _mm256_slli_epi16(m3,1)|_mm256_srli_epi16(m3_sr8,7);

    __m256i _m1h = _mm256_permute2x128_si256( m1, m1 , 0xf1 );
    m1 ^= _mm256_srli_epi16( _mm256_srli_si256(_m1h,15)  , 7);
    __m256i m1l_ = _mm256_permute2x128_si256( m1 , m1 , 0x0f );
    __m256i m1_sr8 = _mm256_alignr_epi8( m1 , m1l_ , 15 );
    m0 ^= _mm256_slli_epi16(m1,1)|_mm256_srli_epi16(m1_sr8,7);
  }

  _mm256_storeu_si256( poly+0 , m0 );
  _mm256_storeu_si256( poly+1 , m1 );
  _mm256_storeu_si256( poly+2 , m2 );
  _mm256_storeu_si256( poly+3 , m3 );
  _mm256_storeu_si256( poly+4 , m4 );
  _mm256_storeu_si256( poly+5 , m5 );
  _mm256_storeu_si256( poly+6 , m6 );
  _mm256_storeu_si256( poly+7 , m7 );
}


/////////////////////////////////////////////////////////////////


static inline
void __xor_down_256( __m256i * poly , unsigned dest_idx , unsigned src_idx , unsigned len )
{
	for(unsigned i=len;i>0;){
		i--;
    _mm256_storeu_si256(poly+dest_idx+i,_mm256_loadu_si256(poly+dest_idx+i)^_mm256_loadu_si256(poly+src_idx+i));
	}
}

static inline
void __xor_down_256_2( __m256i * poly , unsigned len , unsigned l_st ){
	__xor_down_256( poly , l_st , len , len );
//	for( int i=len-1;i>=0;i--) poly[l_st+i] ^= poly[len+i];
}

static inline
void __xor_up_256( __m256i * poly , unsigned dest_idx , unsigned src_idx , unsigned len )
{
	for(unsigned i=0;i<len;i++){
    _mm256_storeu_si256(poly+dest_idx+i,_mm256_loadu_si256(poly+dest_idx+i)^_mm256_loadu_si256(poly+src_idx+i));
	}
}

static inline
void __xor_up_256_2( __m256i * poly , unsigned len , unsigned l_st ){
	__xor_up_256( poly , l_st , len , len );
}



//////////////////////////////////////////////////////////////////


 // at least 8x256bits = 256byts
static
void varsub_x256( __m256i* poly256 , unsigned n_256 )
{
	if( 1 >= n_256 ) return;
	unsigned log_n = __builtin_ctz( n_256 );

	while( log_n > 8 ) {
		unsigned unit = 1<<log_n;
		unsigned num = n_256/unit;
		unsigned unit_2 = unit>>1;
		for(unsigned j=0;j<num;j++) __xor_down_256_2( poly256+j*unit , unit_2 , (1<<(log_n-9)) );
		log_n--;
	}

	if( n_256 >= 256 ) { for(unsigned i=0;i<n_256;i+=256) { div_s8_p128_256x256bit( poly256+i ); } }
	if( n_256 >= 128 ) { for(unsigned i=0;i<n_256;i+=128) { div_s8_p64_128x256bit( poly256+i ); } }
	if( n_256 >= 64 ) { for(unsigned i=0;i<n_256;i+=64) { div_s8_p32_64x256bit( poly256+i ); } }
	if( n_256 >= 32 ) { for(unsigned i=0;i<n_256;i+=32) { div_s8_p16_32x256bit( poly256+i ); } }
	if( n_256 >= 16 ) { for(unsigned i=0;i<n_256;i+=16) { div_s8_p8_16x256bit( poly256+i ); } }
	for(unsigned i=0;i<n_256;i+=8) { s8div_bit_8x256bit( poly256+i ); }
}



static inline
void idiv_s8_p128_256x256bit( __m256i * poly )
{
  // [65536]/(32768,128)
  #define max_idx  255
  #define width    128
  ///

  __m256i m3_l = _mm256_setzero_si256();
  for(int i=0;i<width;i+=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m0_1 = _mm256_permute2x128_si256( m3_l , m0 , 0x21 );

    m3_l = m3;
    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_1 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_1 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_1 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_1 );
  }
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_maxYmmH);
  #undef  max_idx
  #undef  width
}

static inline
void idiv_s8_p64_128x256bit( __m256i * poly )
{
  // [32768]/(16384,64)
  #define max_idx 127
  #define width   64
  #define sr_byte 8
  ///
  __m256i m3_l = _mm256_setzero_si256();
  for(int i=0;i<width;i+=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );

    __m256i m0_1 = _mm256_permute2x128_si256( m3_l , m0 , 0x21 );
    __m256i m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    m3_l = m3;
    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );
  }
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void idiv_s8_p32_64x256bit( __m256i * poly )
{
  // [16384]/(8192,32)
  #define max_idx 63
  #define width   32
  #define sr_byte 12
  ///
  __m256i m3_l = _mm256_setzero_si256();
  for(int i=0;i<width;i+=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );

    __m256i m0_1 = _mm256_permute2x128_si256( m3_l , m0 , 0x21 );
    __m256i m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    m3_l = m3;
    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );
  }
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void idiv_s8_p16_32x256bit( __m256i * poly )
{
  // [8192]/(4096,16)
  #define max_idx 31
  #define width   16
  #define sr_byte 14
  ///
  __m256i m3_l = _mm256_setzero_si256();
  for(int i=0;i<width;i+=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );

    __m256i m0_1 = _mm256_permute2x128_si256( m3_l , m0 , 0x21 );
    __m256i m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    m3_l = m3;
    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );
  }
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void idiv_s8_p8_16x256bit( __m256i * poly )
{
  // [4096]/(2048,8)
  #define max_idx 15
  #define width   8
  #define sr_byte 15
  ///
  __m256i m3_l = _mm256_setzero_si256();
  for(int i=0;i<width;i+=4) {
    __m256i m3 = _mm256_loadu_si256( poly+width+i+3 );
    __m256i m2 = _mm256_loadu_si256( poly+width+i+2 );
    __m256i m1 = _mm256_loadu_si256( poly+width+i+1 );
    __m256i m0 = _mm256_loadu_si256( poly+width+i );

    __m256i m0_1 = _mm256_permute2x128_si256( m3_l , m0 , 0x21 );
    __m256i m0_2 = _mm256_alignr_epi8( m0 , m0_1 , sr_byte );

    __m256i m1_1 = _mm256_permute2x128_si256( m0 , m1 , 0x21 );
    __m256i m1_2 = _mm256_alignr_epi8( m1 , m1_1 , sr_byte );

    __m256i m2_1 = _mm256_permute2x128_si256( m1 , m2 , 0x21 );
    __m256i m2_2 = _mm256_alignr_epi8( m2 , m2_1 , sr_byte );

    __m256i m3_1 = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_2 = _mm256_alignr_epi8( m3 , m3_1 , sr_byte );

    m3_l = m3;
    _mm256_storeu_si256( poly+i+3 , _mm256_loadu_si256(poly+i+3)^m3_2 );
    _mm256_storeu_si256( poly+i+2 , _mm256_loadu_si256(poly+i+2)^m2_2 );
    _mm256_storeu_si256( poly+i+1 , _mm256_loadu_si256(poly+i+1)^m1_2 );
    _mm256_storeu_si256( poly+i+0 , _mm256_loadu_si256(poly+i+0)^m0_2 );
  }
  __m256i _maxYmmH = _mm256_permute2x128_si256( _mm256_loadu_si256(poly+max_idx) , _mm256_loadu_si256(poly+max_idx) , 0xf1 );
  _mm256_storeu_si256(poly+width,_mm256_loadu_si256(poly+width)^_mm256_srli_si256(_maxYmmH,sr_byte));
  #undef  max_idx
  #undef  width
  #undef  sr_byte
}

static inline
void i_s8div_bit_8x256bit( __m256i * poly )
{
  // only perform 3 layers:
  __m256i m0 = _mm256_loadu_si256( poly+0 );
  __m256i m1 = _mm256_loadu_si256( poly+1 );
  __m256i m2 = _mm256_loadu_si256( poly+2 );
  __m256i m3 = _mm256_loadu_si256( poly+3 );
  __m256i m4 = _mm256_loadu_si256( poly+4 );
  __m256i m5 = _mm256_loadu_si256( poly+5 );
  __m256i m6 = _mm256_loadu_si256( poly+6 );
  __m256i m7 = _mm256_loadu_si256( poly+7 );

  // [512]/(256,1)
  {
    __m256i m7l_ = _mm256_permute2x128_si256( m7 , m7 , 0x0f );
    __m256i m7_sr8 = _mm256_alignr_epi8( m7 , m7l_ , 15 );
    m6 ^= _mm256_slli_epi16(m7,1)|_mm256_srli_epi16(m7_sr8,7);
    __m256i _m7h = _mm256_permute2x128_si256( m7, m7 , 0xf1 );
    m7 ^= _mm256_srli_epi16( _mm256_srli_si256(_m7h,15)  , 7);

    __m256i m5l_ = _mm256_permute2x128_si256( m5 , m5 , 0x0f );
    __m256i m5_sr8 = _mm256_alignr_epi8( m5 , m5l_ , 15 );
    m4 ^= _mm256_slli_epi16(m5,1)|_mm256_srli_epi16(m5_sr8,7);
    __m256i _m5h = _mm256_permute2x128_si256( m5, m5 , 0xf1 );
    m5 ^= _mm256_srli_epi16( _mm256_srli_si256(_m5h,15)  , 7);

    __m256i m3l_ = _mm256_permute2x128_si256( m3 , m3 , 0x0f );
    __m256i m3_sr8 = _mm256_alignr_epi8( m3 , m3l_ , 15 );
    m2 ^= _mm256_slli_epi16(m3,1)|_mm256_srli_epi16(m3_sr8,7);
    __m256i _m3h = _mm256_permute2x128_si256( m3, m3 , 0xf1 );
    m3 ^= _mm256_srli_epi16( _mm256_srli_si256(_m3h,15)  , 7);

    __m256i m1l_ = _mm256_permute2x128_si256( m1 , m1 , 0x0f );
    __m256i m1_sr8 = _mm256_alignr_epi8( m1 , m1l_ , 15 );
    m0 ^= _mm256_slli_epi16(m1,1)|_mm256_srli_epi16(m1_sr8,7);
    __m256i _m1h = _mm256_permute2x128_si256( m1, m1 , 0xf1 );
    m1 ^= _mm256_srli_epi16( _mm256_srli_si256(_m1h,15)  , 7);
  }
  // [1025]/(512,2)
  {
    __m256i m6l_ = _mm256_permute2x128_si256( m6 , m6 , 0x2f );
    __m256i m6_sr8 = _mm256_alignr_epi8( m6 , m6l_ , 15 );
    m4 ^= _mm256_slli_epi16(m6,2)|_mm256_srli_epi16(m6_sr8,6);

    __m256i m7l_m6h = _mm256_permute2x128_si256( m6 , m7 , 0x21 );
    __m256i m7_m6_sr8 = _mm256_alignr_epi8( m7 , m7l_m6h , 15 );
    m5 ^= _mm256_slli_epi16(m7,2)|_mm256_srli_epi16(m7_m6_sr8,6);

    __m256i _m7h = _mm256_permute2x128_si256( m7, m7 , 0xf1 );
    m6 ^= _mm256_srli_epi16( _mm256_srli_si256(_m7h,15)  , 6);

    __m256i m2l_ = _mm256_permute2x128_si256( m2 , m2 , 0x2f );
    __m256i m2_sr8 = _mm256_alignr_epi8( m2 , m2l_ , 15 );
    m0 ^= _mm256_slli_epi16(m2,2)|_mm256_srli_epi16(m2_sr8,6);

    __m256i m3l_m2h = _mm256_permute2x128_si256( m2 , m3 , 0x21 );
    __m256i m3_m2_sr8 = _mm256_alignr_epi8( m3 , m3l_m2h , 15 );
    m1 ^= _mm256_slli_epi16(m3,2)|_mm256_srli_epi16(m3_m2_sr8,6);

    __m256i _m3h = _mm256_permute2x128_si256( m3, m3 , 0xf1 );
    m2 ^= _mm256_srli_epi16( _mm256_srli_si256(_m3h,15)  , 6);

  }
  // [2048]/(1024,4)
  {
    __m256i m4l_ = _mm256_permute2x128_si256( m4 , m4 , 0x2f );
    __m256i m4_sr8 = _mm256_alignr_epi8( m4 , m4l_ , 15 );
    m0 ^= _mm256_slli_epi16(m4,4)|_mm256_srli_epi16(m4_sr8,4);

    __m256i m5l_m4h = _mm256_permute2x128_si256( m4 , m5 , 0x21 );
    __m256i m5_m4_sr8 = _mm256_alignr_epi8( m5 , m5l_m4h , 15 );
    m1 ^= _mm256_slli_epi16(m5,4)|_mm256_srli_epi16(m5_m4_sr8,4);

    __m256i m6l_m5h = _mm256_permute2x128_si256( m5 , m6 , 0x21 );
    __m256i m6_m5_sr8 = _mm256_alignr_epi8( m6 , m6l_m5h , 15 );
    m2 ^= _mm256_slli_epi16(m6,4)|_mm256_srli_epi16(m6_m5_sr8,4);

    __m256i m7l_m6h = _mm256_permute2x128_si256( m6 , m7 , 0x21 );
    __m256i m7_m6_sr8 = _mm256_alignr_epi8( m7 , m7l_m6h , 15 );
    m3 ^= _mm256_slli_epi16(m7,4)|_mm256_srli_epi16(m7_m6_sr8,4);

    __m256i _m7h = _mm256_permute2x128_si256( m7, m7 , 0xf1 );
    m4 ^= _mm256_srli_epi16( _mm256_srli_si256(_m7h,15)  , 4);
  }

  _mm256_storeu_si256( poly+0 , m0 );
  _mm256_storeu_si256( poly+1 , m1 );
  _mm256_storeu_si256( poly+2 , m2 );
  _mm256_storeu_si256( poly+3 , m3 );
  _mm256_storeu_si256( poly+4 , m4 );
  _mm256_storeu_si256( poly+5 , m5 );
  _mm256_storeu_si256( poly+6 , m6 );
  _mm256_storeu_si256( poly+7 , m7 );
}





static
void i_varsub_x256( __m256i* poly256 , unsigned n_256 )
{
	if( 1 >= n_256 ) return;
	unsigned log_n = __builtin_ctz( n_256 );

	for(unsigned i=0;i<n_256;i+=8) { i_s8div_bit_8x256bit( poly256+i ); }
	if( n_256 >= 16 ) { for(unsigned i=0;i<n_256;i+=16) { idiv_s8_p8_16x256bit( poly256+i ); } }
	if( n_256 >= 32 ) { for(unsigned i=0;i<n_256;i+=32) { idiv_s8_p16_32x256bit( poly256+i ); } }
	if( n_256 >= 64 ) { for(unsigned i=0;i<n_256;i+=64) { idiv_s8_p32_64x256bit( poly256+i ); } }
	if( n_256 >= 128 ) { for(unsigned i=0;i<n_256;i+=128) { idiv_s8_p64_128x256bit( poly256+i ); } }
	if( n_256 >= 256 ) { for(unsigned i=0;i<n_256;i+=256) { idiv_s8_p128_256x256bit( poly256+i ); } }

	for(unsigned i=9;i<=log_n ; i++ ) {
		unsigned unit = 1<<i;
		unsigned num = n_256/unit;
		unsigned unit_2 = unit>>1;
		for(unsigned j=0;j<num;j++) __xor_up_256_2( poly256+j*unit , unit_2 , (1<<(i-9)) );
	}
}







//////////////////////////////////////////////////////////////////


#include "bc_256.h"


void bc_1( void * poly , unsigned n_byte )
{
	__m256i * poly256 = (__m256i*) poly;
	unsigned n_256 = n_byte >>5;

	varsub_x256( poly256 , n_256 );  // at least 8x256bits = 256byts

	bc_256( poly256 , n_256 );

	bc_1_256( (uint32_t*)poly256 , n_256 );  // at least 4x256bits = 128byts
}


void ibc_1( void * poly , unsigned n_byte )
{
	__m256i * poly256 = (__m256i*) poly;
	unsigned n_256 = n_byte >>5;

	ibc_1_256( (uint32_t*)poly256 , n_256 );

	ibc_256( poly256 , n_256 );

	i_varsub_x256( poly256 , n_256 );
}

