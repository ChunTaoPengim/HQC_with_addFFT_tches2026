#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>


#include "polymul.h"


#define TEST_NUM (200)

static inline
void mul64_ref( uint64_t * c0 , uint64_t * c1 , uint64_t a , uint64_t b )
{
    uint64_t t0=0,t1=0;
    if( a&1 ) { t0 ^= b; }
    for(int k=1;k<64;k++){
        if ((a>>k)&1) {
            t0 ^= (b << k);
            t1 ^= (b >> (64-k));
        }
    }
    c0[0] = t0;
    c1[0] = t1;
}


static void polymul_ref( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned len )
{
    for(unsigned i=0;i<len*2;i++) c[i] = 0;
    for(unsigned i=0;i<len;i++){
        for(unsigned j=0;j<len;j++){
            uint64_t ai = a[i];
            uint64_t bj = b[j];
            uint64_t c0=0;
            uint64_t c1=0;
            mul64_ref( &c0 , &c1 , ai , bj );
            c[i+j]   ^= c0;
            c[i+j+1] ^= c1;
        }
    }
}


void test_polymul( unsigned len_u64 )
{
    uint64_t a[POLYMUL_MAX_INPUT_U64];
    uint64_t b[POLYMUL_MAX_INPUT_U64];
    uint64_t c0[POLYMUL_MAX_INPUT_U64*2];
    uint64_t c1[POLYMUL_MAX_INPUT_U64*2];

    if (len_u64 > POLYMUL_MAX_INPUT_U64) {
        printf("polynomial size > MAX_INPUT_U64(%d)\n", POLYMUL_MAX_INPUT_U64 );
        return;
    }

    int fail = 0;
    for(int j = 0; j < TEST_NUM; j++)
    {
        for(unsigned i = 0; i < len_u64; i++) {
            a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
        }

        polymul_ref(c0,a,b,len_u64);
        polymul_fafft(c1,a,b,len_u64);

        for(unsigned i = 0; i < len_u64*2; i++)
        {
            if(c0[i]!=c1[i])
            {
                printf("test FAIL [%d,%d]: ",j,i);
                printf("%lx vs %lx \n", c0[i] , c1[i]);
                if (0==i) {
                    printf("a0: %lx , b0: %lx\n", a[0], b[0]);
                }
                fail = 1;
            }
            if( fail ) break;
        }
        if (fail) break;
    }

    printf("TEST bitpolymul %d x u64 = %d bits\n", len_u64 , len_u64*64 );
    printf("TEST (%d) %s\n", TEST_NUM , (fail)?"FAIL":"PASS" );

}



#define _BENCHMARK_

#if defined(_BENCHMARK_)

#include "benchmark.h"

uint64_t bm_mul[TEST_NUM];
uint64_t bm_inp[TEST_NUM];
int len_m[1];
int len_i[1];

void benchmark_polymul( unsigned len_u64 )
{
    uint64_t a[POLYMUL_MAX_INPUT_U64];
    uint64_t b[POLYMUL_MAX_INPUT_U64];
    uint64_t a1[POLYMUL_MAX_INPUT_U64*POLYMUL_EXP_RATIO];
    uint64_t b1[POLYMUL_MAX_INPUT_U64*POLYMUL_EXP_RATIO];
    uint64_t c0[POLYMUL_MAX_INPUT_U64*2];

    if (len_u64 > POLYMUL_MAX_INPUT_U64) {
        printf("polynomial size > MAX_INPUT_U64(%d)\n", POLYMUL_MAX_INPUT_U64 );
        return;
    }

    char mesg[256];
    len_m[0] = 0;
    len_i[0] = 0;

    for(int j = 0; j < TEST_NUM; j++)
    {
        for(unsigned i = 0; i < len_u64; i++) {
            a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
        }
REC_TIMING( bm_mul , len_m , {
    polymul_input_transform(b1,b,len_u64);
    REC_TIMING( bm_inp , len_i , {
        polymul_input_transform(a1,a,len_u64);
    });
    polymul_output(c0,a1,b1,len_u64);
} );
    }

    printf("benchmark bitpolymul %d x u64 = %d bits\n", len_u64 , len_u64*64 );
    report( mesg , sizeof(mesg) , bm_mul , len_m[0] );
    printf( "bitmul: %s\n" , mesg );
    report( mesg , sizeof(mesg) , bm_inp , len_i[0] );
    printf( "inp tr: %s\n" , mesg );
}

#endif


int main(void)
{

    test_polymul( 128 );
    test_polymul( 256 );
    //test_polymul( 512 );
    //test_polymul( 1024 );

#if defined(_BENCHMARK_)
    benchmark_polymul( 256 );
    benchmark_polymul( 512 );
    benchmark_polymul( 1024 );
#endif
    return 0;
}
