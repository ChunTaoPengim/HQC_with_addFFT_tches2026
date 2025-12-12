#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>

#if 0
static void dump_u8( const char * mesg , const uint8_t * vec , unsigned len )
{
    printf( "%s" , mesg );
    for(int i=0;i<len;i++) {
        printf("%02x ", vec[i]);
        if( 3==(i&3)) { printf(","); }
        if( 7==(i&7)) { printf("|"); }
        if(i==len-1) break;
        if( 15==(i&15)) { printf("\n"); }
    }
    printf("\n");
}
#else
static void dump_u8( const char * mesg , const uint8_t * vec , unsigned len ) { (void)mesg; (void)vec; (void)len; }
#endif

static void polymul_ref( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned len )
{
    for(unsigned i=0;i<len*2;i++) c[i] = 0;
    for(unsigned i=0;i<len;i++){
        for(unsigned j=0;j<len;j++){
            uint64_t ai = a[i];
            uint64_t bj = b[j];
            uint64_t c0=0;
            uint64_t c1=0;
            if( ai&1 ) { c0 ^= bj; }
            for(int k=1;k<64;k++){
                if ((ai>>k)&1) {
                    c0 ^= (bj << k);
                    c1 ^= (bj >> (64-k));
                }
            }
            c[i+j]   ^= c0;
            c[i+j+1] ^= c1;
        }
    }
}

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
    dump_u8( "a0: " , a0 , len );
    bc_8( a0 , len );
    dump_u8( "bc: " , a0 , len );
    bitpoly_to_gf256x2_n( a0 , a1 , a0 , len );
    dump_u8( "gf0:" , a0 , len );
    dump_u8( "gf1:" , a1 , len );
    // first stage of btfy
    memcpy( a0+len , a0 , len );
    memcpy( a1+len , a1 , len );
    btfy_gf256x2( a0 , a1 , loglen , 0 );
    btfy_gf256x2( a0+len , a1+len , loglen , len );
    dump_u8( "btf:" , a0 , len*2 );
    dump_u8( "btf:" , a1 , len*2 );

    memcpy( b0 , b , len );
    bc_8( b0 , len );
    bitpoly_to_gf256x2_n( b0 , b1 , b0 , len );
    // first stage of btfy
    memcpy( b0+len , b0 , len );
    memcpy( b1+len , b1 , len );
    btfy_gf256x2( b0 , b1 , loglen , 0 );
    btfy_gf256x2( b0+len , b1+len , loglen , len );
    dump_u8( "b0b:" , b0 , len*2 );
    dump_u8( "b1b:" , b1 , len*2 );

    // multiply
    gf256x2v_mul( a0 , a1 , a0 , a1 , b0 , b1 , len*2 );
    dump_u8( "ptm:" , a0 , len*2 );
    dump_u8( "ptm:" , a1 , len*2 );

    // output transform
    uint8_t * c0 = (uint8_t*)c;
    uint8_t * c1 = b1;
    ibtfy_gf256x2( a0 , a1 , loglen+1 , 0 );
    dump_u8( "ibt:" , c0 , len*2 );
    dump_u8( "ibt:" , c1 , len*2 ); 
    gf256x2_to_bitpoly_n( c0 , c1 , a0 , a1 , 2*len );
    dump_u8( "pol:" , c0 , len*2 );
    dump_u8( "pol:" , c1 , len*2 ); 
    ibc_8( c0 , 2*len );
    ibc_8( c1 , 2*len );
    dump_u8( "ibc:" , c0 , len*2 );
    dump_u8( "ibc:" , c1 , len*2 );
    gf256v_add( c0+1 , c0+1 , c1 , 2*len-1 );
    //for(int i=0;i<2*len-1;i++) c0[i+1] ^= c1[i];
    return 0;
}


#define MAX_TEST_LEN   (128)

#if MAX_TEST_LEN*2 > MAX_BUFFLEN_U64
error -- no enough buff size
#endif



int main(void)
{
#define MAXLEN  (MAX_TEST_LEN)
//#define MAXLEN  (1)

    uint64_t a[MAXLEN*2] = {0};
    uint64_t b[MAXLEN*2] = {0};
    uint64_t c0[MAXLEN*2];
    uint64_t c1[MAXLEN*2];
    int fail = 0;
    for(int j = 0; j < 1000; j++)
    {
        if (0==j) {
            a[0] = 0x1;
            b[0] = 0x1;
        } else if (1==j) {
            a[1] = 0x1ffffffff;
            b[1] = 0x1ffffffff;        
        } else if (2==j) {
            a[MAXLEN-1] = 0x1ffffffff;
            b[MAXLEN-1] = 0x1ffffffff;
        } else {
            int qq = (j < MAXLEN )? j : MAXLEN;
            for(int i = 0; i < qq; i++) {
                a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
                b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            }
        }
        //for(int i=0;i<LEN2048*2;i++) {c0[i]=0;}
        //polymul(a,b,c0,LEN2048);
        polymul_ref(c0,a,b,MAXLEN);
        polymul_addfft(c1,a,b,MAXLEN);

        for(int i = 0; i < MAXLEN*2; i++)
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
        printf("test fafft_polymul( [%d]xu64 ) pass [%d].\n",MAXLEN, 1000);
    }
    return 0;
}
