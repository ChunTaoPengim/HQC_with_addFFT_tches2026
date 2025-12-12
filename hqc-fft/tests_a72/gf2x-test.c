
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  // uint64_t
#include <stdlib.h>  // rand()
#include <inttypes.h>
#include <stddef.h>


#define TEST_RUN 1000


#include "hal.h"
#include "gf2x.h"
#include "parameters.h"
// void vect_mul(__m256i *o, const __m256i *v1, const __m256i *v2);
static int cmp_uint64_t(const void *a, const void *b)
{
  return (int)((*((const uint64_t *)a)) - (*((const uint64_t *)b)));
}

static void print_median(const char *txt, uint64_t cyc[TEST_RUN])
{
  printf("%10s cycles = %" PRIu64 "\n", txt, cyc[TEST_RUN >> 1]);
}


int main(void) {

    enable_cyclecounter();
    printf("benchmark for gf2x [%d] bits ->[%d] u64 \n\n", PARAM_N , VEC_N_SIZE_64 );

    uint64_t poly_a[VEC_N_SIZE_64];
    uint64_t poly_b[VEC_N_SIZE_64];
    uint64_t poly_c[VEC_N_SIZE_64];
    uint64_t poly_d[VEC_N_SIZE_64] = {0};

    uint64_t cycles_mul[TEST_RUN];
    uint64_t t0, t1;





    printf("===========  benchmark vect_mul()  ================\n\n");
    for (unsigned i = 0; i < TEST_RUN; i++) {
        for(size_t j=0;j<(sizeof(poly_d)/sizeof(uint64_t));j++) poly_a[j] = rand();
        for(size_t j=0;j<(sizeof(poly_d)/sizeof(uint64_t));j++) poly_b[j] = rand();
        
        t0 = get_cyclecounter();
        vect_mul( (void*)poly_c , (void*)poly_a , (void*)poly_b );
        t1 = get_cyclecounter();
        cycles_mul[i] = t1 - t0;
        
        for(size_t j=0;j<(sizeof(poly_d)/sizeof(uint64_t));j++) poly_d[j] ^= poly_c[j];
    }
    
    qsort(cycles_mul, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
    print_median("vect_mul", cycles_mul);
    printf("XX: %x\n", (unsigned) (poly_d[0]&0xffff) );


    return 0;
}
