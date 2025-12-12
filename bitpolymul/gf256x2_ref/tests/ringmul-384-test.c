#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>


#include "ringmul.h"
#include "polymul.h"



static inline
uint16_t _mul_8( uint8_t a , uint8_t b )
{
    uint16_t r = 0;
    for( int i = 0 ; i < 8 ; i++ )
    {
        if( b & (1<<i) )
            r ^= ((uint16_t)a) << i;
    }
    return r;
}

static void ringmul_384_ref( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b )
{
#define LEN (384)
    uint16_t t0[LEN] = {0};
    for(int i=0;i<LEN;i++)
    {
        for(int j=0;j<LEN;j++)
        {
            if( i+j >= LEN ) break;
            t0[i+j] ^= _mul_8( a[i] , b[j] );
        }
    }
    for(int i=0;i<LEN;i++)
    {
        c0[i] = t0[i] & 0xff;
        c1[i] = (t0[i] >> 8);
    }
#undef LEN
}



static void ringmul_1024_ref( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b )
{
#define LEN (1024)
    uint16_t t0[LEN] = {0};
    for(int i=0;i<LEN;i++)
    {
        for(int j=0;j<LEN;j++)
        {
            if( i+j >= LEN ) break;
            t0[i+j] ^= _mul_8( a[i] , b[j] );
        }
    }
    for(int i=0;i<LEN;i++)
    {
        c0[i] = t0[i] & 0xff;
        c1[i] = (t0[i] >> 8);
    }
#undef LEN
}




#define TEST_NUM (100)

#define MAX_INPUT_U8 (384)

int test_ringmul_384(int test_num)
{
    uint8_t a0[MAX_INPUT_U8] = {0};
    uint8_t b0[MAX_INPUT_U8] = {0};

    uint8_t c0[MAX_INPUT_U8];
    uint8_t c1[MAX_INPUT_U8];
    uint8_t d0[MAX_INPUT_U8];
    uint8_t d1[MAX_INPUT_U8];

    printf("test_ringmul_384: test poly-ring (mod x^384) of input size (%d) x 1\n" , MAX_INPUT_U8 );
    int fail = 0;
    for(int j = 0; j < test_num; j++)
    {
        if(0==j) {
            for(unsigned i = 0; i < 128; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 128; i++) { b0[i] = rand()&0xff; }
        } else {
            for(unsigned i = 0; i < 384; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 384; i++) { b0[i] = rand()&0xff; }
        }

        ringmul_384_ref( c0 , c1 , a0 , b0 );
        ringmul_mul_384( d0 , d1 , a0 , b0 );

        for(unsigned i = 0; i < 384; i++)
        {
            if( (c0[i]!=d0[i])||(c1[i]!=d1[i]))
            {
                printf("test FAIL [%d,%d]: [ref] %x,%x vs [test] %x,%x \n",j,i, c1[i] , c0[i] , d1[i] , d0[i] );
                fail = 1;
            }
            //if( fail ) { printf("\n"); break; }
        }
        if( fail ) break;
    }
    printf("TEST (%d) %s\n", test_num , (fail)?"FAIL":"PASS" );
    return !fail;
}


#define MAX_INPUT1024_U8 (1024)

int test_ringmul_1024(int test_num)
{
    uint8_t a0[MAX_INPUT1024_U8] = {0};
    uint8_t b0[MAX_INPUT1024_U8] = {0};

    uint8_t c0[MAX_INPUT1024_U8];
    uint8_t c1[MAX_INPUT1024_U8];
    uint8_t d0[MAX_INPUT1024_U8];
    uint8_t d1[MAX_INPUT1024_U8];


    printf("test_ringmul_1024: test poly-ring (mod x^1024) of input size (%d) x 1\n" , MAX_INPUT1024_U8 );
    int fail = 0;
    for(int j = 0; j < test_num; j++)
    {
        if(0==j) {
            for(unsigned i = 0; i < 128; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 128; i++) { b0[i] = rand()&0xff; }
        } else {
            for(unsigned i = 0; i < 1024; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 1024; i++) { b0[i] = rand()&0xff; }
        }

        ringmul_1024_ref( c0 , c1 , a0 , b0 );

        ringmul_mul_1024( d0 , d1 , a0 , b0 );

        for(unsigned i = 0; i < 1024; i++)
        {
            if( (c0[i]!=d0[i])||(c1[i]!=d1[i]))
            {
                printf("test FAIL [%d,%d]: [ref] %x,%x vs [test] %x,%x \n",j,i, c1[i] , c0[i] , d1[i] , d0[i] );
                fail = 1;
            }
            //if( fail ) { printf("\n"); break; }
        }
        if( fail ) break;
    }
    printf("TEST (%d) %s\n", test_num , (fail)?"FAIL":"PASS" );
    return !fail;
}



#define _DO_BENCHMARK_

#ifdef _DO_BENCHMARK_
#include "benchmark.h"

void benchmark_ringmul_384(void)
{
    uint8_t a0[MAX_INPUT_U8] = {0};
    uint8_t b0[MAX_INPUT_U8] = {0};

    uint8_t d0[MAX_INPUT_U8];
    uint8_t d1[MAX_INPUT_U8];
    int test_num = TEST_NUM;
    uint64_t rec_m[TEST_NUM]; int len_m[1] = {0};

    printf("benchmark_ringmul_384: test poly-ring (mod x^384) of input size (%d) x 1\n" , MAX_INPUT_U8 );
    int fail = 0;
    for(int j = 0; j < test_num; j++)
    {
        if(0==j) {
            for(unsigned i = 0; i < 128; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 128; i++) { b0[i] = rand()&0xff; }
        } else {
            for(unsigned i = 0; i < 384; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 384; i++) { b0[i] = rand()&0xff; }
        }

REC_TIMING( rec_m , len_m , {
        ringmul_mul_384( d0 , d1 , a0 , b0 );
});
        if( fail ) break;
    }
    char mesg[256];
    report( mesg , sizeof(mesg) , rec_m , len_m[0] );
    printf( "bitmul: %s\n" , mesg );
}



void benchmark_ringmul_1024(void)
{
    uint8_t a0[MAX_INPUT1024_U8] = {0};
    uint8_t b0[MAX_INPUT1024_U8] = {0};
    uint8_t d0[MAX_INPUT1024_U8];
    uint8_t d1[MAX_INPUT1024_U8];
    int test_num = TEST_NUM;
    uint64_t rec_m[TEST_NUM]; int len_m[1] = {0};

    printf("benchmark_ringmul_1024: test poly-ring (mod x^1024) of input size (%d) x 1\n" , MAX_INPUT1024_U8 );
    int fail = 0;
    for(int j = 0; j < test_num; j++)
    {
        if(0==j) {
            for(unsigned i = 0; i < 128; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 128; i++) { b0[i] = rand()&0xff; }
        } else {
            for(unsigned i = 0; i < 1024; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < 1024; i++) { b0[i] = rand()&0xff; }
        }
REC_TIMING( rec_m , len_m , {
        ringmul_mul_1024( d0 , d1 , a0 , b0 );
});
        if( fail ) break;
    }
    char mesg[256];
    report( mesg , sizeof(mesg) , rec_m , len_m[0] );
    printf( "bitmul: %s\n" , mesg );
}



#endif



//#define _COUNT_GF216MUL

#ifdef _COUNT_GF216MUL
uint64_t get_gf216mul(void);
void reset_gf216mul(void);
#endif

int main(void)
{

    test_ringmul_384(TEST_NUM);
#ifdef _COUNT_GF216MUL
    printf("# gf216mul: %llu\n", get_gf216mul() );
    reset_gf216mul();
#endif

    test_ringmul_1024(TEST_NUM);
#ifdef _COUNT_GF216MUL
    printf("# gf216mul: %llu\n", get_gf216mul() );
#endif


#ifdef _DO_BENCHMARK_
    benchmark_ringmul_384();
    benchmark_ringmul_1024();
#endif

    return 0;
}
