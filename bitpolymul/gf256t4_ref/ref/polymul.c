#include "stdint.h"
#include "string.h"

#include "polymul.h"

#include "btfy.h"
#include "gf256.h"
#include "dencoder_gf256t4.h"
#include "bc_1.h"

static inline
unsigned log_floor(unsigned a)
{
    for(unsigned i=0;i<32;i++){
        if(0==a) return i-1;
        a &=  ~(1<<i);
    }
    return 32;
}

// sizeof a_fft = sizeof a * POLYMUL_EXP_RATIO
// n_u64: size of the bit polynomial a in u64 
void polymul_input_transform( uint64_t * a_fft , const uint64_t * a , unsigned n_u64 )
{
    uint64_t buffer[POLYMUL_MAX_INPUT_U64*2] __attribute__ ((aligned (32)));

    unsigned n = n_u64*8;
    unsigned n_u32 = n_u64*2;
    unsigned loglen = log_floor(n_u32);
    uint8_t *a0 = ((uint8_t*)a_fft);
    uint8_t *a1 = ((uint8_t*)a_fft) + n_u32*2;
    uint8_t *a2 = ((uint8_t*)a_fft) + n_u32*2*2;
    uint8_t *a3 = ((uint8_t*)a_fft) + n_u32*2*3;

    // input transform
    memcpy( buffer , a , n );
    bc_1( buffer , n );
    encode_gf256t4(a0,a1,a2,a3,n_u32*2,(uint32_t*)buffer,16);
    btfy_gf256t4(a0,a1,a2,a3,loglen+1,1ULL<<(16+loglen+1));
}

// c = a_fft * b_fft
// n_u64: size of the bit polynomial a in u64 
void polymul_output( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft , unsigned n_u64 )
{
    uint64_t buffer[POLYMUL_MAX_INPUT_U64*2] __attribute__ ((aligned (32)));

    unsigned n = n_u64*8;
    unsigned n_u32 = n_u64*2;
    unsigned loglen = log_floor(n_u32);
    uint8_t *a0 = ((uint8_t*)a_fft);
    uint8_t *a1 = ((uint8_t*)a_fft) + n_u32*2;
    uint8_t *a2 = ((uint8_t*)a_fft) + n_u32*2*2;
    uint8_t *a3 = ((uint8_t*)a_fft) + n_u32*2*3;
    uint8_t *b0 = ((uint8_t*)b_fft);
    uint8_t *b1 = ((uint8_t*)b_fft) + n_u32*2;
    uint8_t *b2 = ((uint8_t*)b_fft) + n_u32*2*2;
    uint8_t *b3 = ((uint8_t*)b_fft) + n_u32*2*3;
    uint8_t *c0 = ((uint8_t*)buffer);
    uint8_t *c1 = ((uint8_t*)buffer) + n_u32*2;
    uint8_t *c2 = ((uint8_t*)buffer) + n_u32*2*2;
    uint8_t *c3 = ((uint8_t*)buffer) + n_u32*2*3;

    gf256t4v_mul( c0 , c1 , c2 , c3 , a0 , a1 , a2 , a3 , b0 , b1 , b2 , b3 , n_u32*2 );

    // output transform
    ibtfy_gf256t4(c0,c1,c2,c3,loglen+1,1ULL<<(16+loglen+1));
    decode_gf256t4((uint32_t*)c,c0,c1,c2,c3,n_u32*2);
    ibc_1( c , n*2 );
}

// c = a * b
// n_u64: size of the bit polynomial a in u64 
void polymul_fafft( uint64_t * c , const uint64_t * a , const uint64_t * b, unsigned n_u64 )
{
    uint64_t a1[POLYMUL_MAX_INPUT_U64*2] __attribute__ ((aligned (32)));
    uint64_t b1[POLYMUL_MAX_INPUT_U64*2] __attribute__ ((aligned (32)));

    polymul_input_transform( a1 , a , n_u64 );
    polymul_input_transform( b1 , b , n_u64 );
    polymul_output( c , a1 , b1 , n_u64 );
}


