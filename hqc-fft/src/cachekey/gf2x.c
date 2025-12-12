/**
 * \file gf2x.c
 * \brief Implementation of multiplication of two polynomials
 */

#include "gf2x.h"

#include "string.h"

#include "parameters.h"

#include "polymul.h"

#if 17669 == PARAM_N
// 280
#define N_PAD_U64  (17920/64)
#define N_MOD_64   (5)
#elif 35851 == PARAM_N
// 576
#define N_PAD_U64  (36864/64)
#define N_MOD_64   (11)
#elif 57637 == PARAM_N
#define N_PAD_U64  (1024)
#define N_MOD_64   (37)
#else
error here.
#endif

void ring_to_fftform( uint8_t *v_fft , const uint8_t *v )
{
    uint64_t tmp1[N_PAD_U64];
#if 17669 == PARAM_N
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { tmp1[i] = 0; }
    memcpy( tmp1 , v , VEC_N_SIZE_BYTES );
    polymul_17920_input( v_fft , tmp1 );
#elif 35851 == PARAM_N
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { tmp1[i] = 0; }
    memcpy( tmp1 , v , VEC_N_SIZE_BYTES );
    polymul_36864_input( v_fft , tmp1 );
#elif 57637 == PARAM_N
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { tmp1[i] = 0; }
    memcpy( tmp1 , v , VEC_N_SIZE_BYTES );
    polymul_input_transform( v_fft , tmp1 , N_PAD_U64 );
#else
error -- no matched implementation for PARAM_N 
#endif
}

void ring_to_fftform_rlonly( uint8_t *v_fft , const uint8_t *v )
{
    uint64_t tmp1[N_PAD_U64];
#if 17669 == PARAM_N
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { tmp1[i] = 0; }
    memcpy( tmp1 , v , VEC_N_SIZE_BYTES );
    polymul_280U64_input( v_fft , tmp1 );
#elif 35851 == PARAM_N
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { tmp1[i] = 0; }
    memcpy( tmp1 , v , VEC_N_SIZE_BYTES );
    polymul_576U64_input( v_fft , tmp1 );
#elif 57637 == PARAM_N
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { tmp1[i] = 0; }
    memcpy( tmp1 , v , VEC_N_SIZE_BYTES );
    polymul_input_transform( v_fft , tmp1 , N_PAD_U64 );
#else
error -- no matched implementation for PARAM_N 
#endif
}


static inline
void ring_reduce(uint8_t * _c , const uint64_t *temp_c)
{
    uint64_t c[VEC_N_SIZE_64];
    for(int i=0;i<VEC_N_SIZE_64;i++) {
        c[i] = temp_c[i] ^ ((temp_c[VEC_N_SIZE_64-1+i]>>N_MOD_64) | (temp_c[VEC_N_SIZE_64+i]<<(64-N_MOD_64)));
    }
    c[VEC_N_SIZE_64-1] &= BITMASK(PARAM_N, 64);
    memcpy( _c , c , VEC_N_SIZE_64*8 );
}

void ring_mul_fftformx2( uint8_t *c, const uint8_t *a_fft, const uint8_t *b_fft )
{
    uint64_t temp_c[N_PAD_U64*2];
#if 17669 == PARAM_N
    polymul_17920_mul( temp_c , a_fft , b_fft );
#elif 35851 == PARAM_N
    polymul_36864_mul( temp_c , a_fft , b_fft );
#elif 57637 == PARAM_N
    polymul_output( temp_c , a_fft , b_fft , N_PAD_U64 );
#else
error -- no matched implementation for PARAM_N 
#endif
    ring_reduce(c, temp_c);
}

void ring_mul_fftform( uint8_t *c, const uint8_t *a, const uint8_t *b_fft )
{
    uint64_t a_fft[R_FFTFORM_BYTES/8];
    ring_to_fftform( a_fft , a );
    ring_mul_fftformx2( c , a_fft , b_fft );
}

void ring_mul( uint8_t *c, const uint8_t *a, const uint8_t *b )
{
    uint64_t a_fft[R_FFTFORM_BYTES/8];
    uint64_t b_fft[R_FFTFORM_BYTES/8];
    ring_to_fftform( a_fft , a );
    ring_to_fftform( b_fft , b );
    ring_mul_fftformx2( c , a_fft , b_fft );
}

void ring_mul_x2( uint8_t * c1 , uint8_t * c2 , const uint8_t * a , const uint8_t * b1 , const uint8_t * b2 )
{
    uint64_t a_fft[R_FFTFORM_BYTES/8];
    uint64_t b1_fft[R_FFTFORM_BYTES/8];
    uint64_t b2_fft[R_FFTFORM_BYTES/8];
    ring_to_fftform( a_fft , a );
    ring_to_fftform( b1_fft , b1 );
    ring_to_fftform( b2_fft , b2 );
    ring_mul_fftformx2( c1 , a_fft , b1_fft );
    ring_mul_fftformx2( c2 , a_fft , b2_fft );
}

#if 1
static inline void ring_mul_fftformx2_2( uint8_t *c, const uint8_t *a_fft, const uint8_t *s_fft , const uint8_t * s )
{
    uint64_t temp_c[N_PAD_U64*2];
#if 17669 == PARAM_N
    //polymul_17920_mul( temp_c , a_fft , b_fft );
    uint64_t ss[N_PAD_U64]; // 280
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { ss[i] = 0; }
    memcpy( ss , s , VEC_N_SIZE_BYTES );
    polymul_280U64_mul( temp_c , a_fft + POLYMUL_280U64_FFTSIZE_U64*8 , ss , a_fft , s_fft );
#elif 35851 == PARAM_N
    //polymul_36864_mul( temp_c , a_fft , b_fft );
    uint64_t ss[N_PAD_U64]; // 576
    for(int i=VEC_N_SIZE_BYTES/8;i<N_PAD_U64;i++) { ss[i] = 0; }
    memcpy( ss , s , VEC_N_SIZE_BYTES );
    polymul_576U64_mul( temp_c , a_fft + POLYMUL_576U64_FFTSIZE_U64*8 , ss , a_fft , s_fft );
#elif 57637 == PARAM_N
    (void) s;
    polymul_output( temp_c , a_fft , s_fft , N_PAD_U64 );
#else
error -- no matched implementation for PARAM_N 
#endif
    ring_reduce(c, temp_c);
}

void ring_mul_x2_fftform( uint8_t * c1 , uint8_t * c2 , const uint8_t * a , const uint8_t * b1_fft , const uint8_t * s_fft , const uint8_t * s )
{
    uint64_t a_fft[R_FFTFORM_BYTES/8];
    ring_to_fftform( a_fft , a );
    ring_mul_fftformx2( c1 , a_fft , b1_fft );
    ring_mul_fftformx2_2( c2 , a_fft , s_fft , s );
}
#else
void ring_mul_x2_fftform( uint8_t * c1 , uint8_t * c2 , const uint8_t * a , const uint8_t * b1_fft , const uint8_t * b2_fft )
{
    uint64_t a_fft[R_FFTFORM_BYTES/8];
    ring_to_fftform( a_fft , a );
    ring_mul_fftformx2( c1 , a_fft , b1_fft );
    ring_mul_fftformx2( c2 , a_fft , b2_fft );
}
#endif

void vect_mul(uint64_t *o, const uint64_t *v1, const uint64_t *v2) { ring_mul((uint8_t*)o,(uint8_t*)v1,(uint8_t*)v2); }


