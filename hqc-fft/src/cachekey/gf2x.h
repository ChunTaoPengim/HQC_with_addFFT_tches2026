/**
 * \file gf2x.h
 * \brief Header file for gf2x.c
 */

#ifndef HQC_GF2X_H
#define HQC_GF2X_H


#include "parameters.h"

#include "polymul.h"

#if 17669 == PARAM_N
#define R_FFTFORM_BYTES  POLYMUL_17920_FFTSIZE_BYTE
#define R_FFTFORM_RLONLY_BYTES  (POLYMUL_280U64_FFTSIZE_U64*8)
#elif 35851 == PARAM_N
#define R_FFTFORM_BYTES  POLYMUL_36864_FFTSIZE_BYTE
#define R_FFTFORM_RLONLY_BYTES  (POLYMUL_576U64_FFTSIZE_U64*8)
#elif 57637 == PARAM_N
#define R_FFTFORM_BYTES  (1024*8*POLYMUL_EXP_RATIO)
#define R_FFTFORM_RLONLY_BYTES  (R_FFTFORM_BYTES)
#else
error here.
#endif


#include "stdint.h"

///
/// Arithmetic operations for the quotient ring :=  gf2[x]/(x^N-1)
///

void ring_to_fftform( uint8_t *v_fft , const uint8_t *v );

void ring_to_fftform_rlonly( uint8_t *v_fft , const uint8_t *v );

void ring_mul_fftformx2( uint8_t *c, const uint8_t *a_fft, const uint8_t *b_fft );

void ring_mul_fftform( uint8_t *c, const uint8_t *a, const uint8_t *b_fft );

void ring_mul( uint8_t *c, const uint8_t *a, const uint8_t *b );

void ring_mul_x2( uint8_t * c1 , uint8_t * c2 , const uint8_t * a , const uint8_t * b1 , const uint8_t * b2 );

//void ring_mul_x2_fftform( uint8_t * c1 , uint8_t * c2 , const uint8_t * a , const uint8_t * b1_fft , const uint8_t * b2_fft );
// assert( sizeof( s ) ==  VEC_N_SIZE_BYTES )
void ring_mul_x2_fftform( uint8_t * c1 , uint8_t * c2 , const uint8_t * a , const uint8_t * b1_fft , const uint8_t * s_fft , const uint8_t * s );


void vect_mul(uint64_t *o, const uint64_t *v1, const uint64_t *v2);


#endif

