#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>



static inline
unsigned log_floor(unsigned a)
{
    for(unsigned i=0;i<32;i++){
        if(0==a) return i-1;
        a &=  ~(1<<i);
    }
    return 32;
}


#define MAX_BUFFLEN_U64   (2048)

#include "btfy.h"
#include "gf256.h"
#include "bitpoly_to_gf256x2.h"
#include "bc_8.h"


#include "benchmark.h"

#define _PROFILE_

#if defined(_PROFILE_)
struct benchmark bm_ptm;
struct benchmark bm_ibc;
struct benchmark bm_ibt;
#endif

static int polymul_addfft( uint64_t *c, const uint64_t * a, const uint64_t * b , unsigned n_u64 )
{
    if (0==n_u64) return 0;

    uint64_t buf0[MAX_BUFFLEN_U64] __attribute__ ((aligned (32)));
    uint64_t buf1[MAX_BUFFLEN_U64] __attribute__ ((aligned (32)));
    uint64_t buf2[MAX_BUFFLEN_U64] __attribute__ ((aligned (32)));
    uint64_t buf3[MAX_BUFFLEN_U64] __attribute__ ((aligned (32)));

    uint8_t * a0 = (uint8_t *)buf0;
    uint8_t * a1 = (uint8_t *)buf1;
    uint8_t * b0 = (uint8_t *)buf2;
    uint8_t * b1 = (uint8_t *)buf3;

    unsigned len = n_u64*8;
    unsigned loglen = log_floor(len);

    // input transform
    memcpy( a0 , a , len );
    bc_8( a0 , len );
    bitpoly_to_gf256x2_n( a0 , a1 , a0 , len );
    // first stage of btfy
    memcpy( a0+len , a0 , len );
    memcpy( a1+len , a1 , len );
    btfy_gf256x2( a0 , a1 , loglen , 0 );
    btfy_gf256x2( a0+len , a1+len , loglen , len );

    memcpy( b0 , b , len );
    bc_8( b0 , len );
    bitpoly_to_gf256x2_n( b0 , b1 , b0 , len );
    // first stage of btfy
    memcpy( b0+len , b0 , len );
    memcpy( b1+len , b1 , len );
    btfy_gf256x2( b0 , b1 , loglen , 0 );
    btfy_gf256x2( b0+len , b1+len , loglen , len );

#if defined(_PROFILE_)
BENCHMARK( bm_ptm , {
    // multiply
    gf256x2v_mul( a0 , a1 , a0 , a1 , b0 , b1 , len*2 );
});
    // output transform
    uint8_t * c0 = (uint8_t*)c;
    uint8_t * c1 = b1;
BENCHMARK( bm_ibt , {
    ibtfy_gf256x2( a0 , a1 , loglen+1 , 0 );
});
BENCHMARK( bm_ibc , {
    gf256x2_to_bitpoly_n( c0 , c1 , a0 , a1 , 2*len );
    ibc_8( c0 , 2*len );
    ibc_8( c1 , 2*len );
    gf256v_add( c0+1 , c0+1 , c1 , 2*len-1 );
});
#else
    // multiply
    gf256x2v_mul( a0 , a1 , a0 , a1 , b0 , b1 , len*2 );
    // output transform
    uint8_t * c0 = (uint8_t*)c;
    uint8_t * c1 = b1;
    ibtfy_gf256x2( a0 , a1 , loglen+1 , 0 );
    gf256x2_to_bitpoly_n( c0 , c1 , a0 , a1 , 2*len );
    ibc_8( c0 , 2*len );
    ibc_8( c1 , 2*len );
    for(unsigned i=0;i<2*len-1;i++) c0[i+1] ^= c1[i];
#endif
    return 0;
}


void benchmark_polymul( unsigned len_u64 )
{
    uint64_t a[MAX_BUFFLEN_U64] = {0};
    uint64_t b[MAX_BUFFLEN_U64] = {0};
    uint64_t c0[MAX_BUFFLEN_U64];

    if (len_u64*2 > MAX_BUFFLEN_U64) {
        printf("polynomial size > MAX_BUFFLEN_U64(%d)\n", MAX_BUFFLEN_U64 );
        return;
    }

    char mesg[256];
    struct benchmark bm_mul;
    bm_init(&bm_mul);

#if defined(_PROFILE_)
    bm_init(&bm_ptm);
    bm_init(&bm_ibc);
    bm_init(&bm_ibt);
#endif

    int fail = 0;
    for(int j = 0; j < 1000; j++)
    {
        for(unsigned i = 0; i < len_u64; i++) {
            a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
        }
        //for(int i=0;i<LEN2048*2;i++) {c0[i]=0;}
        //polymul(a,b,c0,LEN2048);
BENCHMARK( bm_mul , {
        polymul_addfft(c0,a,b,len_u64);
});
        if( fail ) break;
    }

    printf("benchmark bitpolymul %d x u64 = %d bits\n", len_u64 , len_u64*64 );
    bm_dump( mesg , sizeof(mesg) , &bm_mul );
    printf( "bitmul: %s\n" , mesg );
#if defined(_PROFILE_)
    bm_dump( mesg , sizeof(mesg) , &bm_ptm );
    printf( "pntmul: %s\n" , mesg );
    bm_dump( mesg , sizeof(mesg) , &bm_ibt );
    printf( "ibtfy : %s\n" , mesg );
    bm_dump( mesg , sizeof(mesg) , &bm_ibc );
    printf( "ibc   : %s\n" , mesg );
#endif

}



int main(void)
{

    benchmark_polymul( 256 );
    benchmark_polymul( 512 );
    benchmark_polymul( 1024 );

    return 0;
}
