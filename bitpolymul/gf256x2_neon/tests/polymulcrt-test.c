#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>


static inline void _mull_64( uint64_t * c0 , uint64_t * c1 , uint64_t a , uint64_t b )
{
    uint64_t r0 = a & (-(b&1));
    uint64_t r1 = 0;
    for(int i=1;i<64;i++){
        uint64_t mask = -((b>>i)&1);
        uint64_t t0 = a<<i;
        uint64_t t1 = a>>(64-i);
        r0 ^= (t0&mask);
        r1 ^= (t1&mask);
    }
    c0[0] = r0;
    c1[0] = r1;
}

static void polymul_ref( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned len )
{
    for(unsigned i=0;i<len*2;i++) c[i] = 0;
    for(unsigned i=0;i<len;i++){
        for(unsigned j=0;j<len;j++){
            uint64_t c0, c1;
            _mull_64( &c0 , &c1 , a[i] , b[j] );
            c[i+j]   ^= c0;
            c[i+j+1] ^= c1;
        }
    }
}

#include "polymul.h"


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
            if( fail ) { printf("\n"); break; }
        }
        if( fail ) break;
    }
    if(!fail) {
        printf("test polymul_crt( [%d]xu64 ) pass [%d].\n", TEST1_INP_U64 ,  test_run);
    }
    return !fail;
}



int main(void)
{
    test_0(500);
    test_1(100);

    return 0;
}
