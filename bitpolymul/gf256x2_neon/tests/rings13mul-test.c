#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>


#include "ringmul.h"
#include "polymul.h"



static inline void _mull_8( uint8_t * c0 , uint8_t * c1 , uint8_t a , uint8_t b )
{
    uint8_t r0 = a & (-(b&1));
    uint8_t r1 = 0;
    for(int i=1;i<8;i++){
        uint8_t mask = -((b>>i)&1);
        uint8_t t0 = a<<i;
        uint8_t t1 = a>>(8-i);
        r0 ^= (t0&mask);
        r1 ^= (t1&mask);
    }
    c0[0] = r0;
    c1[0] = r1;
}

//  s13 = x^8192 + x^4096 + x^512 + x^256 + x^32 + x^16 + x^2 + x

static void ringmul_s13_ref( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b )
{
    uint8_t cc0[16384];
    uint8_t cc1[16384];
    for(unsigned i=0;i<16384;i++) cc0[i] = 0;
    for(unsigned i=0;i<16384;i++) cc1[i] = 0;

    for(unsigned i=0;i<8192;i++){
        for(unsigned j=0;j<8192;j++){
            uint8_t d0, d1;
            _mull_8( &d0 , &d1 , a[i] , b[j] );
            cc0[i+j] ^= d0;
            cc1[i+j] ^= d1;
        }
    }
    for(unsigned i=16384-1;i>=8192;i--) {
        cc0[i-8192+4096] ^= cc0[i];
        cc0[i-8192+512]  ^= cc0[i];
        cc0[i-8192+256]  ^= cc0[i];
        cc0[i-8192+32]   ^= cc0[i];
        cc0[i-8192+16]   ^= cc0[i];
        cc0[i-8192+2]    ^= cc0[i];
        cc0[i-8192+1]    ^= cc0[i];

        cc1[i-8192+4096] ^= cc1[i];
        cc1[i-8192+512]  ^= cc1[i];
        cc1[i-8192+256]  ^= cc1[i];
        cc1[i-8192+32]   ^= cc1[i];
        cc1[i-8192+16]   ^= cc1[i];
        cc1[i-8192+2]    ^= cc1[i];
        cc1[i-8192+1]    ^= cc1[i];
    }
    for(int i=0;i<8192;i++) c0[i] = cc0[i];
    for(int i=0;i<8192;i++) c1[i] = cc1[i];
}



#define TEST_NUM (30)

#define MAX_INPUT_U8 (8192)

#define RINGMUL_S13_INPUT_U8 (4096+512)
#define RINGMUL_S13_U8 (8192)


int test_ringmul_s13(void)
{
    uint8_t a0[MAX_INPUT_U8] = {0};
    uint8_t b0[MAX_INPUT_U8] = {0};
    uint8_t a0fft[MAX_INPUT_U8];
    uint8_t a1fft[MAX_INPUT_U8];
    uint8_t b0fft[MAX_INPUT_U8];
    uint8_t b1fft[MAX_INPUT_U8];

    uint8_t c0[MAX_INPUT_U8];
    uint8_t c1[MAX_INPUT_U8];
    uint8_t d0[MAX_INPUT_U8];
    uint8_t d1[MAX_INPUT_U8];


    printf("test_ringmul_s13: test poly-ring (mod s13) of input size (%d) x 1\n" , RINGMUL_S13_INPUT_U8 );
    int fail = 0;
    for(int j = 0; j < TEST_NUM; j++)
    {
        if(0==j) {
            a0[4096] = 1;
            b0[4096] = 1;
        } else {
            for(unsigned i = 0; i < RINGMUL_S13_INPUT_U8; i++) { a0[i] = rand()&0xff; }
            for(unsigned i = 0; i < RINGMUL_S13_INPUT_U8; i++) { b0[i] = rand()&0xff; }
        }

        ringmul_s13_ref( c0 , c1 , a0 , b0 );

        ringmul_s13_input_35864( a0fft , a1fft , a0 );
        ringmul_s13_input_35864( b0fft , b1fft , b0 );
        ringmul_s13_mul( d0 , d1 , a0fft , a1fft , b0fft , b1fft );

        for(unsigned i = 0; i < RINGMUL_S13_U8; i++)
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
    printf("TEST (%d) %s\n", TEST_NUM , (fail)?"FAIL":"PASS" );
    return !fail;
}



int main(void)
{

    test_ringmul_s13();

    return 0;
}
