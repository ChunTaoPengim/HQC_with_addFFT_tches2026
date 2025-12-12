#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>


#include "ringmul.h"


#define TEST_NUM (1000)

#define MAX_BUFF (8192)

#define RINGMUL_S12 (4096)
#define RINGMUL_S13 (8192)

//#define RINGMUL_INP0 (2048+192)
//#define RINGMUL_INP1 (4096+384)
#define RINGMUL_INP0 (2048+192)
#define RINGMUL_INP1 (4096+512)


void test_ringmul_s12(void)
{
    uint8_t a0[MAX_BUFF];
    uint8_t b0[MAX_BUFF]; for(unsigned i = 0; i < MAX_BUFF; i++) { b0[i] = 1; }
    uint8_t b1[MAX_BUFF] = {0};
    uint8_t a0fft[MAX_BUFF];
    uint8_t a1fft[MAX_BUFF];

    uint8_t c0[MAX_BUFF];
    uint8_t c1[MAX_BUFF];


    printf("test_ringmul_s12: test poly-ring (mod s12) of input size (%d byte)\n" , RINGMUL_INP0 );
    int fail = 0;
    for(int j = 0; j < TEST_NUM; j++)
    {
        for(unsigned i = 0; i < RINGMUL_INP0; i++) { a0[i] = rand(); }
        for(unsigned i = RINGMUL_INP0; i < MAX_BUFF; i++) { a0[i] = 0; }

        ringmul_s12_input_2240( a0fft , a1fft , a0 );
        ringmul_s12_mul( c0 , c1 , a0fft , a1fft , b0 , b1 );

        for(unsigned i = 0; i < RINGMUL_S12; i++)
        {
            if((c0[i]!=a0[i])||(c1[i]!=0))
            {
                printf("test /s12 FAIL [%d,%d]: ",j,i);
                printf("c[%d]=%x,%x : a[%d]=%x \n", i , c1[i] , c0[i] , i , a0[i]);
                fail = 1;
            }
            if( fail ) break;
        }
        if( fail ) break;
    }
    printf("TEST ringmul_s12 %d x u8 -> %d bytes\n", RINGMUL_INP0 , RINGMUL_S12 );
    printf("TEST (%d) %s\n", TEST_NUM , (fail)?"FAIL":"PASS" );
}

void test_ringmul_s13(void)
{
    uint8_t a0[MAX_BUFF];
    uint8_t b0[MAX_BUFF]; for(unsigned i = 0; i < MAX_BUFF; i++) { b0[i] = 1; }
    uint8_t b1[MAX_BUFF] = {0};
    uint8_t a0fft[MAX_BUFF];
    uint8_t a1fft[MAX_BUFF];

    uint8_t c0[MAX_BUFF];
    uint8_t c1[MAX_BUFF];

    printf("test_ringmul_s13: test poly-ring (mod s13) of input size (%d byte)\n" , RINGMUL_INP1 );
    int fail = 0;
    for(int j = 0; j < TEST_NUM; j++)
    {
        for(unsigned i = 0; i < RINGMUL_INP1; i++) { a0[i] = rand(); }
        for(unsigned i = RINGMUL_INP1; i < MAX_BUFF; i++) { a0[i] = 0; }

        ringmul_s13_input_4608( a0fft , a1fft , a0 );
        ringmul_s13_mul( c0 , c1 , a0fft , a1fft , b0 , b1 );

        for(unsigned i = 0; i < RINGMUL_S13; i++)
        {
            if((c0[i]!=a0[i])||(c1[i]!=0))
            {
                printf("test /s13 FAIL [%d,%d]: ",j,i);
                printf("c[%d]=%x,%x : a[%d]=%x \n", i , c1[i] , c0[i] , i , a0[i]);
                fail = 1;
            }
            if( fail ) break;
        }
        if( fail ) break;
    }
    printf("TEST ringmul_s13 %d x u8 -> %d bytes\n", RINGMUL_INP1 , RINGMUL_S13 );
    printf("TEST (%d) %s\n", TEST_NUM , (fail)?"FAIL":"PASS" );
}


#define _BENCHMARK_

#if defined(_BENCHMARK_)

#include "benchmark.h"


void benchmark_ringmul_s12(void)
{
    uint8_t a0[MAX_BUFF];
    uint8_t b0[MAX_BUFF]; for(unsigned i = 0; i < MAX_BUFF; i++) { b0[i] = 1; }
    uint8_t b1[MAX_BUFF] = {0};
    uint8_t a0fft[MAX_BUFF];
    uint8_t a1fft[MAX_BUFF];

    uint8_t c0[MAX_BUFF];
    uint8_t c1[MAX_BUFF];

    char mesg[256];
    struct benchmark bm_mul;
    struct benchmark bm_inp;
    bm_init(&bm_mul);
    bm_init(&bm_inp);

    printf("benchmark_ringmul_s12: test poly-ring (mod s12) of input size (%d byte)\n" , RINGMUL_INP0 );
    for(int j = 0; j < TEST_NUM; j++)
    {
        for(unsigned i = 0; i < RINGMUL_INP0; i++) { a0[i] = rand(); }
        for(unsigned i = RINGMUL_INP0; i < MAX_BUFF; i++) { a0[i] = 0; }

BENCHMARK( bm_inp , {
        ringmul_s12_input_2240( a0fft , a1fft , a0 );
} );
BENCHMARK( bm_mul , {
        ringmul_s12_mul( c0 , c1 , a0fft , a1fft , b0 , b1 );
} );
    }
    bm_dump( mesg , sizeof(mesg) , &bm_inp );
    printf( "inp: %s\n" , mesg );
    bm_dump( mesg , sizeof(mesg) , &bm_mul );
    printf( "mul: %s\n" , mesg );
}

void benchmark_ringmul_s13(void)
{
    uint8_t a0[MAX_BUFF];
    uint8_t b0[MAX_BUFF]; for(unsigned i = 0; i < MAX_BUFF; i++) { b0[i] = 1; }
    uint8_t b1[MAX_BUFF] = {0};
    uint8_t a0fft[MAX_BUFF];
    uint8_t a1fft[MAX_BUFF];

    uint8_t c0[MAX_BUFF];
    uint8_t c1[MAX_BUFF];

    char mesg[256];
    struct benchmark bm_mul;
    struct benchmark bm_inp;
    bm_init(&bm_mul);
    bm_init(&bm_inp);

    printf("benchmark_ringmul_s13: test poly-ring (mod s13) of input size (%d byte)\n" , RINGMUL_INP1 );
    for(int j = 0; j < TEST_NUM; j++)
    {
        for(unsigned i = 0; i < RINGMUL_INP1; i++) { a0[i] = rand(); }
        for(unsigned i = RINGMUL_INP1; i < MAX_BUFF; i++) { a0[i] = 0; }
BENCHMARK( bm_inp , {
        ringmul_s13_input_4608( a0fft , a1fft , a0 );
} );
BENCHMARK( bm_mul , {
        ringmul_s13_mul( c0 , c1 , a0fft , a1fft , b0 , b1 );
} );
    }
    bm_dump( mesg , sizeof(mesg) , &bm_inp );
    printf( "inp: %s\n" , mesg );
    bm_dump( mesg , sizeof(mesg) , &bm_mul );
    printf( "mul: %s\n" , mesg );
}

#endif


int main(void)
{

    test_ringmul_s12();
    test_ringmul_s13();

#if defined(_BENCHMARK_)
    printf("\n\n");
    benchmark_ringmul_s12();
    benchmark_ringmul_s13();
#endif
    return 0;
}
