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


#define TEST0_INP_U64  (280)
#define TEST0_OUT_U64  (POLYMUL_17920_FFTSIZE_BYTE/8)

static
int test_0( int test_run )
{
    uint64_t a[TEST0_INP_U64] = {0};
    uint64_t b[TEST0_INP_U64] = {0};
    uint64_t a_fft[TEST0_OUT_U64];
    uint64_t b_fft[TEST0_OUT_U64];

    uint64_t c0[TEST0_INP_U64*2];
    uint64_t c1[TEST0_INP_U64*2];
    int fail = 0;
    for(int j = 0; j < test_run; j++)
    {
        if (0==j) {
            a[0] = 0x12345678ffffffff;
            b[0] = 0x60b7acd901234567;
        } else {
            for(int i = 0; i < TEST0_INP_U64; i++) {
                a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
                b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            }
            a[TEST0_INP_U64-1] &= 0x7fffffffffffffff;
            b[TEST0_INP_U64-1] &= 0x7fffffffffffffff;
        }
        polymul_ref(c0,a,b,TEST0_INP_U64);
        polymul_17920_input( a_fft , a );
        polymul_17920_input( b_fft , b );
        polymul_17920_mul( c1 , a_fft , b_fft );

        for(int i = 0; i < TEST0_INP_U64*2 ; i++)
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
            if( fail ) { printf("\n"); break; }
        }
        if( fail ) break;
    }
    if(!fail) {
        printf("test polymul_crt( [%d]xu64 ) pass [%d].\n", TEST0_INP_U64 ,  test_run);
    }
    return !fail;
}



#define TEST1_INP_U64  (576)
#define TEST1_OUT_U64  (POLYMUL_36864_FFTSIZE_BYTE/8)

static
int test_1( int test_run )
{
    uint64_t a[TEST1_INP_U64] = {0};
    uint64_t b[TEST1_INP_U64] = {0};
    uint64_t a_fft[TEST1_OUT_U64];
    uint64_t b_fft[TEST1_OUT_U64];

    uint64_t c0[TEST1_INP_U64*2];
    uint64_t c1[TEST1_INP_U64*2];
    int fail = 0;
    for(int j = 0; j < test_run; j++)
    {
        if (0==j) {
            //a[0] = 0x12345678ffffffff;
            //b[0] = 0x1234567801234567;
            a[512] = 1;
            b[512] = 1;
        } else if (1==j) {
            a[0] = 0x12345678ffffffff;
            b[0] = 0x1234567801234567;
        } else {
            //for(int i = 0; i < TEST1_INP_U64; i++) {
            for(int i = 0; i < 512; i++) {
                a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
                b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            }
            //a[512] = 1;
            //b[512] = 1;
            a[TEST1_INP_U64-1] &= 0x7fffffffffffffff;
            b[TEST1_INP_U64-1] &= 0x7fffffffffffffff;
        }
        polymul_ref(c0,a,b,TEST1_INP_U64);
        polymul_36864_input( a_fft , a );
        polymul_36864_input( b_fft , b );
        polymul_36864_mul( c1 , a_fft , b_fft );

        for(int i = 0; i < TEST1_INP_U64*2 ; i++)
        {
            if(c0[i]!=c1[i])
            {
                printf("test FAIL [%d,%d]: [ref] %lx vs [test] %lx \n",j,i, c0[i] , c1[i]);
                if (-1==j) {
                    printf("a0: %lx , b0: %lx\n", a[0], b[0]);
                }
                fail = 1;
            }
            //if( fail ) { printf("\n"); break; }
        }
        if( fail ) break;
    }
    if(!fail) {
        printf("test polymul_crt( [%d]xu64 ) pass [%d].\n", TEST1_INP_U64 ,  test_run);
    }
    return !fail;
}

/////////////////////////////////////////////////////

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

    len_m[0] = 0;
    len_i[0] = 0;

    char mesg[256];

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




void benchmark_polymul_17920(void)
{
    uint64_t a[POLYMUL_MAX_INPUT_U64];
    uint64_t b[POLYMUL_MAX_INPUT_U64];
    uint64_t a1[POLYMUL_MAX_INPUT_U64*POLYMUL_EXP_RATIO];
    uint64_t b1[POLYMUL_MAX_INPUT_U64*POLYMUL_EXP_RATIO];
    uint64_t c0[POLYMUL_MAX_INPUT_U64*2];

    unsigned len_u64 = 17920/64;

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
    polymul_17920_input(b1,b);
    REC_TIMING( bm_inp , len_i , {
        polymul_17920_input(a1,a);
    });
    polymul_17920_mul(c0,a1,b1);
} );
    }

    printf("benchmark bitpolymul %d x u64 = %d bits\n", len_u64 , len_u64*64 );
    report( mesg , sizeof(mesg) , bm_mul , len_m[0] );
    printf( "bitmul: %s\n" , mesg );
    report( mesg , sizeof(mesg) , bm_inp , len_i[0] );
    printf( "inp tr: %s\n" , mesg );
}


void benchmark_polymul_36864(void)
{
    uint64_t a[POLYMUL_MAX_INPUT_U64];
    uint64_t b[POLYMUL_MAX_INPUT_U64];
    uint64_t a1[POLYMUL_MAX_INPUT_U64*POLYMUL_EXP_RATIO];
    uint64_t b1[POLYMUL_MAX_INPUT_U64*POLYMUL_EXP_RATIO];
    uint64_t c0[POLYMUL_MAX_INPUT_U64*2];

    unsigned len_u64 = 36864/64;

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
    polymul_36864_input(b1,b);
    REC_TIMING( bm_inp , len_i , {
        polymul_36864_input(a1,a);
    });
    polymul_36864_mul(c0,a1,b1);
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

    test_0(200);
    test_1(200);

    //test_polymul( 128 );

#if defined(_BENCHMARK_)
    bm_init(NULL);
    benchmark_polymul( 256 );
    benchmark_polymul_17920();
    benchmark_polymul( 512 );
    benchmark_polymul_36864();
    benchmark_polymul( 1024 );
#endif
    return 0;
}
