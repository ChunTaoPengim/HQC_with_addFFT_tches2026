
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  // uint64_t
#include <stdlib.h>  // rand()

#include "benchmark.h"
#include "parameters.h"
#include "reed_muller.h"
#include "reed_solomon.h"
#include "code.h"


#define TEST_RUN 1000


#include "parameters.h"
// void vect_mul(__m256i *o, const __m256i *v1, const __m256i *v2);


int main(void) {
    struct benchmark bm_rme, bm_rse, bm_rmd, bm_rsd;
    bm_init(&bm_rme);
    bm_init(&bm_rse);
    bm_init(&bm_rmd);
    bm_init(&bm_rsd);


    char msg[256];

    uint64_t v[VEC_N1N2_SIZE_64] = {0};
    uint8_t m[VEC_K_SIZE_BYTES] = {0};
    uint64_t tmp[VEC_N1_SIZE_64] = {0};
    uint8_t m2[VEC_K_SIZE_BYTES] = {0};
    
    // uint8_t m2[VEC_K_SIZE_BYTES] = {0};
    
    
    

    printf("===========  benchmark coder  ================\n\n");
    for (unsigned i = 0; i < TEST_RUN; i++) {
        for (int i = 0; i < VEC_K_SIZE_BYTES; i++) {
            m[i] = rand() % 256; // Random message
        }
        // reed_muller_encode(v, tmp);
        BENCHMARK(bm_rse, {
            reed_solomon_encode(tmp, (uint64_t*)m);
        });
        BENCHMARK(bm_rme, {
            reed_muller_encode(v, tmp);
        });
        BENCHMARK(bm_rmd, {
            reed_muller_decode(tmp, v);
        });
        BENCHMARK(bm_rsd, {
            reed_solomon_decode((uint64_t*)m2, tmp);
        });
        for(int i = 0; i < VEC_K_SIZE_BYTES; i++) {
            if(m[i] != m2[i]) {
                printf("Error: m != m2\n");
                return -1;
            }
        }
        
    }
    
    bm_dump(msg, 256, &bm_rme);
    printf("RM encode: %s\n", msg );
    bm_dump(msg, 256, &bm_rse);
    printf("RS encode: %s\n", msg );
    bm_dump(msg, 256, &bm_rmd);
    printf("RM decode: %s\n", msg );
    bm_dump(msg, 256, &bm_rsd);
    printf("RS decode: %s\n", msg );
    // printf("XX: %x\n", (unsigned) (poly_d[0]&0xffff) );


    return 0;
}

