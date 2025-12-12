#ifndef _CIDX_TO_GF256X2_GFNI_H_
#define _CIDX_TO_GF256X2_GFNI_H_


#include <stdint.h>
#include <immintrin.h>

#ifdef  __cplusplus
extern  "C" {
#endif

#define MAX_BTFY_LEN  (8192)

// for s5(x). 1ymm processes 32 elements. 1btfy unit has 64 elements. 8192/64 = 128
#define SIZE_TBL_S5  (MAX_BTFY_LEN/64)

extern const uint8_t cidx_to_gf256x2_2x[SIZE_TBL_S5]; // s5 , if SIZE_TBL_S5 > 128, some values will overflow uint8_t
extern const uint16_t cidx_to_gf256x2_4x[SIZE_TBL_S5]; // s4
extern const uint16_t cidx_to_gf256x2_8x[SIZE_TBL_S5]; // s3
extern const uint16_t cidx_to_gf256x2_16x[SIZE_TBL_S5]; // s2
extern const uint16_t cidx_to_gf256x2_32x[SIZE_TBL_S5]; // s1
extern const uint16_t cidx_to_gf256x2_64x[SIZE_TBL_S5]; // s0

static inline
__m256i cidx_to_gf256( __m256i cidx )
{
    __m256i mat = _mm256_set1_epi64x( 0x170de7e26122492ULL );
    return _mm256_gf2p8affine_epi64_epi8( cidx , mat , 0 );
}

static inline
__m256i cidx_to_gf256x2_l( __m256i cidx_l , __m256i cidx_h )
{
    __m256i mat10 = _mm256_set1_epi64x( 0xd99b2e8f29bb01ULL );
    return _mm256_gf2p8affine_epi64_epi8( cidx_h , mat10 , 0 ) ^ cidx_to_gf256(cidx_l);
}

static inline
__m256i cidx_to_gf256x2_h( __m256i cidx_h ) { return cidx_to_gf256(cidx_h); }

#ifdef  __cplusplus
}
#endif


#endif
