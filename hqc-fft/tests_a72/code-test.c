
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  // uint64_t
#include <inttypes.h>
#include <stddef.h>

#include "hal.h"
#include "parameters.h"
#include "reed_muller.h"
#include "reed_solomon.h"
#include "code.h"

#define TEST_RUN 1000

static int cmp_uint64_t(const void *a, const void *b)
{
  return (int)((*((const uint64_t *)a)) - (*((const uint64_t *)b)));
}

static void print_median(const char *txt, uint64_t cyc[TEST_RUN])
{
  printf("%10s cycles = %" PRIu64 "\n", txt, cyc[TEST_RUN >> 1]);
}

// static int percentiles[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99};

// static void print_percentile_legend(void)
// {
//   unsigned i;
//   printf("%21s", "percentile");
//   for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++)
//   {
//     printf("%7d", percentiles[i]);
//   }
//   printf("\n");
// }

// static void print_percentiles(const char *txt, uint64_t cyc[TEST_RUN])
// {
//   unsigned i;
//   printf("%10s percentiles:", txt);
//   for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++)
//   {
//     printf("%7" PRIu64, (cyc)[TEST_RUN * percentiles[i] / 100] / NITERATIONS);
//   }
//   printf("\n");
// }



#include "parameters.h"
// void vect_mul(__m256i *o, const __m256i *v1, const __m256i *v2);


int main(void) {

    enable_cyclecounter();
    uint64_t v[VEC_N1N2_SIZE_64] = {0};
    uint8_t m[VEC_K_SIZE_BYTES] = {0};
    uint64_t tmp[VEC_N1_SIZE_64] = {0};
    uint8_t m2[VEC_K_SIZE_BYTES] = {0};
    
    // uint8_t m2[VEC_K_SIZE_BYTES] = {0};

    uint64_t cycles_rme[TEST_RUN];
    uint64_t cycles_rse[TEST_RUN];
    uint64_t cycles_rmd[TEST_RUN];
    uint64_t cycles_rsd[TEST_RUN];
    uint64_t t0, t1;
    
    
    

    printf("===========  benchmark coder  ================\n\n");
    for (unsigned i = 0; i < TEST_RUN; i++) {
        for (int i = 0; i < VEC_K_SIZE_BYTES; i++) {
            m[i] = rand() % 256; // Random message
        }
        // reed_muller_encode(v, tmp);
        
        t0 = get_cyclecounter();
        reed_solomon_encode(tmp, (uint64_t*)m);
        t1 = get_cyclecounter();
        cycles_rse[i] = t1 - t0;
    
        t0 = get_cyclecounter();
        reed_muller_encode(v, tmp);
        t1 = get_cyclecounter();
        cycles_rme[i] = t1 - t0;
    
        t0 = get_cyclecounter();
        reed_muller_decode(tmp, v);
        t1 = get_cyclecounter();
        cycles_rmd[i] = t1 - t0;

        t0 = get_cyclecounter();
        reed_solomon_decode((uint64_t*)m2, tmp);
        t1 = get_cyclecounter();
        cycles_rsd[i] = t1 - t0;
        
        for(int i = 0; i < VEC_K_SIZE_BYTES; i++) {
            if(m[i] != m2[i]) {
                printf("Error: m != m2\n");
                return -1;
            }
        }
        
    }
    qsort(cycles_rmd, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
    qsort(cycles_rme, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
    qsort(cycles_rsd, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
    qsort(cycles_rse, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);

    print_median("RM encode", cycles_rme);
    print_median("RS encode", cycles_rse);
    print_median("RM decode", cycles_rmd);
    print_median("RS decode", cycles_rsd);
    disable_cyclecounter();


    return 0;
}

