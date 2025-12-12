#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<stdint.h>
#include<string.h>
#include<assert.h>


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


#include "polymul.h"

#define MAX_BUFF_U64   (POLYMUL_MAX_INPUT_U64*POLYMUL_EXP_RATIO*2)


static int tpolymul_fafft( uint64_t *c, const uint64_t * a, const uint64_t * b , unsigned n_u64 )
{
    if (0==n_u64) return 0;

    uint64_t a1[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t b1[MAX_BUFF_U64] __attribute__ ((aligned (32)));

    polymul_input_transform(a1, a , n_u64);
    polymul_input_transform(b1, b , n_u64);
    polymul_output(c, a1, b1, n_u64);

    return 0;
}


#include "btfy.h"
#include "gf256.h"
#include "dencoder_gf256t4.h"
#include "bc_1.h"

#include "benchmark.h"

#define TEST_RUN (1000)

#if 0
int profile_polymul( uint64_t * _c0 , uint64_t * a , uint64_t * b , unsigned u64len )
{
    uint64_t recs0[TEST_RUN]; unsigned t0[0] = {0};
    uint64_t recs1[TEST_RUN]; unsigned t1[0] = {0};
    uint64_t recs2[TEST_RUN]; unsigned t2[0] = {0};
    uint64_t recs3[TEST_RUN]; unsigned t3[0] = {0};

    char msg[256];

    uint64_t aa[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t bb[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t cc[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    unsigned logu64len = log_floor(u64len);

    int fail = 0;
    for(unsigned j = 0; j < TEST_RUN; j++)
    {
        for(unsigned i=0;i<u64len;i++) a[i]=0;
	    for(unsigned i=0;i<u64len;i++) b[i]=0;

        if (0==j) {
            a[0] = 0x41a7;
            b[0] = 0x60b7acd9;
        } else if (1==j) {
            a[1] = 0x1ffffffff;
            b[1] = 0x1ffffffff;
        } else if (2==j) {
            a[u64len-1] = 0x1ffffffff;
            b[u64len-1] = 0x1ffffffff;
        } else {
            int qq = (j < u64len )? j : u64len;
            for(int i = 0; i < qq; i++) {
                a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
                b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            }
        }
        polymul_input_transform(aa, a , u64len);
        polymul_input_transform(bb, b , u64len);

        unsigned n = u64len*8;
        unsigned n_u32 = u64len*2;
        unsigned loglen = log_floor(n_u32);
        uint8_t *a0 = ((uint8_t*)aa);
        uint8_t *a1 = ((uint8_t*)aa) + n_u32*2;
        uint8_t *a2 = ((uint8_t*)aa) + n_u32*2*2;
        uint8_t *a3 = ((uint8_t*)aa) + n_u32*2*3;
        uint8_t *b0 = ((uint8_t*)bb);
        uint8_t *b1 = ((uint8_t*)bb) + n_u32*2;
        uint8_t *b2 = ((uint8_t*)bb) + n_u32*2*2;
        uint8_t *b3 = ((uint8_t*)bb) + n_u32*2*3;
        uint8_t *c0 = ((uint8_t*)cc);
        uint8_t *c1 = ((uint8_t*)cc) + n_u32*2;
        uint8_t *c2 = ((uint8_t*)cc) + n_u32*2*2;
        uint8_t *c3 = ((uint8_t*)cc) + n_u32*2*3;

        REC_TIMING( recs0 , t0 ,  {
            gf256t4v_mul( c0 , c1 , c2 , c3 , a0 , a1 , a2 , a3 , b0 , b1 , b2 , b3 , n_u32*2 );
        });
        REC_TIMING( recs1 , t1 ,  {
            ibtfy_gf256t4(c0,c1,c2,c3,loglen+1,1ULL<<(16+loglen+1));
        });
        REC_TIMING( recs2 , t2 ,  {
            decode_gf256t4((uint32_t*)_c0,c0,c1,c2,c3,n_u32*2);
        });
        REC_TIMING( recs3 , t3 ,  {
            ibc_1( _c0 , n*2 );
        });
        if( fail ) break;
    }
    printf("profile fafft_polymul( [%d] bits:[%d]xu64 ) [%d].\n", u64len*64 , u64len, TEST_RUN);
    report(msg, 256, recs0, *t0);
    printf("pointmul : %s\n", msg );
    report(msg, 256, recs1, *t1);
    printf("ibtfy    : %s\n", msg );
    report(msg, 256, recs2, *t2);
    printf("decode   : %s\n", msg );
    report(msg, 256, recs3, *t3);
    printf("ibc      : %s\n", msg );

    return 0;
}
#endif

int bench_polymul( uint64_t * c0 , uint64_t * a , uint64_t * b , unsigned u64len )
{
    bm_init(NULL);
    uint64_t recs[TEST_RUN]; unsigned tt[1] = {0};

    char msg[256];

    uint64_t a1[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t b1[MAX_BUFF_U64] __attribute__ ((aligned (32)));

    int fail = 0;
    for(unsigned j = 0; j < TEST_RUN; j++)
    {
        for(unsigned i=0;i<u64len;i++) a[i]=0;
	    for(unsigned i=0;i<u64len;i++) b[i]=0;

        if (0==j) {
            a[0] = 0x41a7;
            b[0] = 0x60b7acd9;
        } else if (1==j) {
            a[1] = 0x1ffffffff;
            b[1] = 0x1ffffffff;
        } else if (2==j) {
            a[u64len-1] = 0x1ffffffff;
            b[u64len-1] = 0x1ffffffff;
        } else {
            int qq = (j < u64len )? j : u64len;
            for(int i = 0; i < qq; i++) {
                a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
                b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            }
        }
        REC_TIMING( recs , tt, {
            polymul_input_transform(a1, a , u64len);
            polymul_input_transform(b1, b , u64len);
            polymul_output(c0, a1, b1, u64len);
        });
        if( fail ) break;
    }
    printf("benchmark fafft_polymul( [%d] bits:[%d]xu64 ) [%d].\n", u64len*64 , u64len, TEST_RUN);
    report(msg, 256, recs , *tt);
    printf("benchmark: %s\n", msg );
    return 0;
}




int test_polymul( uint64_t * c0 , uint64_t * c1 , uint64_t * a , uint64_t * b , unsigned u64len )
{
    int fail = 0;
    for(unsigned j = 0; j < TEST_RUN; j++)
    {
        for(unsigned i=0;i<u64len;i++) a[i]=0;
        for(unsigned i=0;i<u64len;i++) b[i]=0;

        if (0==j) {
            a[0] = 0x41a7;
            b[0] = 0x60b7acd9;
        } else if (1==j) {
            a[1] = 0x1ffffffff;
            b[1] = 0x1ffffffff;
        } else if (2==j) {
            a[u64len-1] = 0x1ffffffff;
            b[u64len-1] = 0x1ffffffff;
        } else {
            int qq = (j < u64len )? j : u64len;
            for(int i = 0; i < qq; i++) {
                a[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
                b[i] = ((uint64_t)rand())^(((uint64_t)rand())<<32);
            }
        }
        //for(int i=0;i<LEN2048*2;i++) {c0[i]=0;}
        //polymul(a,b,c0,LEN2048);
        polymul_ref(c0,a,b,u64len);
        tpolymul_fafft(c1,a,b,u64len);

        for(unsigned i = 0; i < u64len*2; i++)
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
    if(fail) {
        printf("test fafft_polymul( [%d] bits:[%d]xu64 ) fail [%d].\n", u64len*64 , u64len, TEST_RUN);
        return -1;
    } else {
        printf("test fafft_polymul( [%d] bits:[%d]xu64 ) pass [%d].\n", u64len*64 , u64len, TEST_RUN);
        return 0;
    }
}




int main(void)
{
    uint64_t a[MAX_BUFF_U64] __attribute__ ((aligned (32))) = {0};
    uint64_t b[MAX_BUFF_U64] __attribute__ ((aligned (32))) = {0};
    uint64_t c0[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    //uint64_t c1[MAX_TEST_LEN*2];

    bench_polymul( c0 , a , b , 256);
    bench_polymul( c0 , a , b , 512);
    bench_polymul( c0 , a , b , 1024);

    //profile_polymul( c0 , a , b , 1024);

    return 0;
}
