
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  // uint64_t
#include <stdlib.h>  // rand()

#include "benchmark.h"


#define TEST_RUN 1000


#include "parameters.h"
#include "gf2x.h"
// void vect_mul(__m256i *o, const __m256i *v1, const __m256i *v2);


int main(void) {
    struct benchmark bm1;
    bm_init(&bm1);

    char msg[256];

    for (unsigned i = 0; i < 256; i++) {
        msg[i] = i;
    }

    printf("benchmark for gf2x [%d] bits ->[%d] u64 \n\n", PARAM_N , VEC_N_SIZE_64 );

    uint64_t poly_a[VEC_N_SIZE_64];
    uint64_t poly_b[VEC_N_SIZE_64];
    uint64_t poly_c[VEC_N_SIZE_64];
    uint64_t poly_d[VEC_N_SIZE_64] = {0};



    printf("===========  benchmark vect_mul()  ================\n\n");
    for (unsigned i = 0; i < TEST_RUN; i++) {
        for(size_t j=0;j<(sizeof(poly_d)/sizeof(uint64_t));j++) poly_a[j] = rand();
        for(size_t j=0;j<(sizeof(poly_d)/sizeof(uint64_t));j++) poly_b[j] = rand();
        
        BENCHMARK(bm1, {
            vect_mul( (void*)poly_c , (void*)poly_a , (void*)poly_b );
        });
        for(size_t j=0;j<(sizeof(poly_d)/sizeof(uint64_t));j++) poly_d[j] ^= poly_c[j];
    }
    bm_dump(msg, 256, &bm1);
    printf("result: %s\n\n", msg );
    printf("XX: %x\n", (unsigned) (poly_d[0]&0xffff) );


    return 0;
}
