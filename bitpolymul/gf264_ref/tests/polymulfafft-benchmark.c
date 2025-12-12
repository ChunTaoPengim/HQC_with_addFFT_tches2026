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

#include "btfy.h"
#include "gf264.h"
#include "dencoder.h"
#include "bc_1.h"

#include "benchmark.h"

#define MAX_BUFF_U64   (4096)


static int polymul_fafft( uint64_t *c, const uint64_t * a, const uint64_t * b , unsigned n_u64 )
{
    if (0==n_u64) return 0;

    uint64_t a1[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t b1[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t buff[MAX_BUFF_U64] __attribute__ ((aligned (32)));

    unsigned loglen = log_floor(n_u64);
    // input transform
    memcpy( buff , a , n_u64*8 );
    bc_1( buff , n_u64*8 );
    encode_64(a1,n_u64*2,buff,32);
    btfy_64(a1,loglen+1,1ULL<<(32+loglen+1));

    memcpy( buff , b , n_u64*8 );
    bc_1( buff , n_u64*8 );
    encode_64(b1,n_u64*2,buff,32);
    btfy_64(b1,loglen+1,1ULL<<(32+loglen+1));

    gf264v_mul( a1 , a1 , b1 , n_u64*2 );

    // output transform
    ibtfy_64(a1,loglen+1,1ULL<<(32+loglen+1));
    decode_64( buff , a1 , n_u64*2 );
    ibc_1( buff , n_u64*2*8 );

    memcpy( c , buff , n_u64*2*8 );
    return 0;
}




int bench_polymul( uint64_t * c0 , uint64_t * a , uint64_t * b , unsigned u64len )
{
    struct benchmark bm0, bm1, bm2, bm3, bm4;
    bm_init( &bm0 );
    bm_init( &bm1 );
    bm_init( &bm2 );
    bm_init( &bm3 );
    bm_init( &bm4 );

    char msg[256];

    uint64_t a1[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t b1[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    uint64_t buff[MAX_BUFF_U64] __attribute__ ((aligned (32)));
    unsigned logu64len = log_floor(u64len);

    int fail = 0;
    for(unsigned j = 0; j < 1000; j++)
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
BENCHMARK( bm0 , {
        // input transform
        memcpy( buff , a , u64len*8 );
        bc_1( buff , u64len*8 );
        encode_64(a1,u64len*2,buff,32);
        btfy_64(a1,logu64len+1,1ULL<<(32+logu64len+1));
        memcpy( buff , b , u64len*8 );
        bc_1( buff , u64len*8 );
        encode_64(b1,u64len*2,buff,32);
        btfy_64(b1,logu64len+1,1ULL<<(32+logu64len+1));
    BENCHMARK( bm1 , {
        gf264v_mul(c0,a1,b1,u64len*2);
    });
        //int r = fafft_output_transform( c , a1 , 2*n_u64 );
    BENCHMARK( bm2 , {
        ibtfy_64(a1,logu64len+1,1ULL<<(32+logu64len+1));
    } );
    BENCHMARK( bm3 , {
        decode_64(c0,a1,u64len*2);
        //memcpy( c0 , a1 , u64len*8*2 );
        //ibc_1( c0 , u64len*8*2 );
    } );
    BENCHMARK( bm4 , {
        ibc_1( c0 , u64len*2*8);
        //memcpy( c0 , a1 , u64len*8*2 );
        //ibc_1( c0 , u64len*8*2 );
    } );


        //polymul_fafft(c0,a,b,u64len);
});

        if( fail ) break;
    }
    printf("benchmark fafft_polymul( [%d] bits:[%d]xu64 ) [%d].\n", u64len*64 , u64len, 1000);
    bm_dump(msg, 256, &bm0);
    printf("benchmark: %s\n", msg );
    bm_dump(msg, 256, &bm1);
    printf("pointmul : %s\n", msg );
    bm_dump(msg, 256, &bm2);
    printf("ibtfy    : %s\n", msg );
    bm_dump(msg, 256, &bm3);
    printf("decode   : %s\n", msg );
    bm_dump(msg, 256, &bm4);
    printf("ibc      : %s\n", msg );

    return 0;
}





int test_polymul( uint64_t * c0 , uint64_t * c1 , uint64_t * a , uint64_t * b , unsigned u64len )
{
    int fail = 0;
    for(unsigned j = 0; j < 1000; j++)
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
        polymul_fafft(c1,a,b,u64len);

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
        printf("test fafft_polymul( [%d] bits:[%d]xu64 ) fail [%d].\n", u64len*64 , u64len, 1000);
        return -1;
    } else {
        printf("test fafft_polymul( [%d] bits:[%d]xu64 ) pass [%d].\n", u64len*64 , u64len, 1000);
        return 0;
    }
}


#define MAX_TEST_LEN   (1024)

#if MAX_TEST_LEN*2 > MAX_BUFF_U64
error -- no enough buff size
#endif


int main(void)
{
    uint64_t a[MAX_TEST_LEN*2] __attribute__ ((aligned (32))) = {0};
    uint64_t b[MAX_TEST_LEN*2] __attribute__ ((aligned (32))) = {0};
    uint64_t c0[MAX_TEST_LEN*2] __attribute__ ((aligned (32)));
    //uint64_t c1[MAX_TEST_LEN*2];

    bench_polymul( c0 , a , b , 256);
    bench_polymul( c0 , a , b , 512);
    bench_polymul( c0 , a , b , 1024);

    return 0;
}
