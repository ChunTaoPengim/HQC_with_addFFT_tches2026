
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
    uint64_t rme[TEST_RUN] = {0}; unsigned len_rme[1] = {0};
    uint64_t rse[TEST_RUN] = {0}; unsigned len_rse[1] = {0};
    uint64_t rmd[TEST_RUN] = {0}; unsigned len_rmd[1] = {0};
    uint64_t rsd[TEST_RUN] = {0}; unsigned len_rsd[1] = {0};
    bm_init(NULL);

    char msg[256];

    uint64_t v[VEC_N1N2_SIZE_64] = {0};
    uint8_t m[VEC_K_SIZE_BYTES] = {0};
    uint64_t tmp[VEC_N1_SIZE_64] = {0};
    uint8_t m2[VEC_K_SIZE_BYTES] = {0};
    
    // uint8_t m2[VEC_K_SIZE_BYTES] = {0};
    
    
    

    printf("===========  benchmark code_decode()  ================\n\n");
    for (unsigned i = 0; i < TEST_RUN; i++) {
        for (int i = 0; i < VEC_K_SIZE_BYTES; i++) {
            m[i] = rand() % 256; // Random message
        }
        // reed_muller_encode(v, tmp);
        REC_TIMING(rse, len_rse, {
            reed_solomon_encode(tmp, (uint64_t*)m);
        });
        REC_TIMING(rme, len_rme, {
            reed_muller_encode(v, tmp);
        });
        REC_TIMING(rmd, len_rmd, {
            reed_muller_decode(tmp, v);
        });
        REC_TIMING(rsd, len_rsd, {
            reed_solomon_decode((uint64_t*)m2, tmp);
        });
        for(int i = 0; i < VEC_K_SIZE_BYTES; i++) {
            if(m[i] != m2[i]) {
                printf("Error: m != m2\n");
                return -1;
            }
        }
        
    }
    report(msg, 256, rme, len_rme[0]);
    printf("RM encode: %s\n", msg );
    report(msg, 256, rse, len_rse[0]);
    printf("RS encode: %s\n", msg );
    report(msg, 256, rmd, len_rmd[0]);
    printf("RM decode: %s\n", msg );
    report(msg, 256, rsd, len_rsd[0]);
    printf("RS decode: %s\n", msg );
    // printf("XX: %x\n", (unsigned) (poly_d[0]&0xffff) );


    return 0;
}

