#include <stdio.h>
#include <string.h>
#include <stdint.h>  // uint64_t
#include <stdlib.h>  // rand()
#include <inttypes.h>
#include <stddef.h>

#include "api.h"
#include "parameters.h"
#include "hal.h"

#if defined(TEST_CACHEKEY)
#include "cachekey.h"
#endif

int vec_equal(unsigned char *a, unsigned char *b, int len) {
    for(int i = 0 ; i < len ; ++i) {
        if(a[i] != b[i]) return 0;
    }
    return 1;
}

#define TEST_RUN (1000)
static int cmp_uint64_t(const void *a, const void *b)
{
  return (int)((*((const uint64_t *)a)) - (*((const uint64_t *)b)));
}

static void print_median(const char *txt, uint64_t cyc[TEST_RUN])
{
  printf("%10s cycles = %" PRIu64 "\n", txt, cyc[TEST_RUN >> 1]);
}

int main() {

	enable_cyclecounter();
	printf("\n");
	printf("*********************\n");
	printf("**** HQC-%d-%d ****\n", PARAM_SECURITY, PARAM_DFR_EXP);
	printf("*********************\n");

	printf("\n");
	printf("N: %d   ", PARAM_N);
	printf("N1: %d   ", PARAM_N1);
	printf("N2: %d   ", PARAM_N2);
	printf("OMEGA: %d   ", PARAM_OMEGA);
	printf("OMEGA_R: %d   ", PARAM_OMEGA_R);
	printf("Failure rate: 2^-%d   ", PARAM_DFR_EXP);
	printf("Sec: %d bits", PARAM_SECURITY);
	printf("\n");
#if defined(TEST_CACHEKEY)
    printf(" PK: %d:%d bytes", PUBLIC_KEY_BYTES, sizeof(pk_t));
	printf(" SK: %d:%d bytes", SECRET_KEY_BYTES, sizeof(sk_t));
	printf("CPK: %d bytes", CACHE_PUBLICKEYBYTES);
	printf("CSK: %d bytes", CACHE_SECRETKEYBYTES);
	unsigned char cpk[CACHE_PUBLICKEYBYTES];
	unsigned char csk[CACHE_SECRETKEYBYTES];
#endif
	printf("\n");

	unsigned char pk[PUBLIC_KEY_BYTES];
	unsigned char sk[SECRET_KEY_BYTES];
	unsigned char ct[CIPHERTEXT_BYTES];
	unsigned char key1[SHARED_SECRET_BYTES];
	unsigned char key2[SHARED_SECRET_BYTES];

	uint64_t cycles_keygen[TEST_RUN];
	uint64_t cycles_enc[TEST_RUN];
	uint64_t cycles_dec[TEST_RUN];
	uint64_t t0, t1;
#if defined(TEST_CACHEKEY)
	uint64_t cycles_ckeygen[TEST_RUN];
	uint64_t cycles_cenc[TEST_RUN];
	uint64_t cycles_cdec[TEST_RUN];
#endif

    int passed = 1;
    for(int i=0;i<TEST_RUN;i++) {
#if defined(TEST_CACHEKEY)

		t0 = get_cyclecounter();
    	cache_hqckem_keypair(pk, sk, cpk, csk);
		t1 = get_cyclecounter();
		cycles_ckeygen[i] = t1 - t0;
#endif
		t0 = get_cyclecounter();
    	crypto_kem_keypair(pk, sk);
		t1 = get_cyclecounter();
		cycles_keygen[i] = t1 - t0;

		t0 = get_cyclecounter();
        crypto_kem_enc(ct, key1, pk);
		t1 = get_cyclecounter();
		cycles_enc[i] = t1 - t0;

		t0 = get_cyclecounter();
		crypto_kem_dec(key2, ct, sk);
	    t1 = get_cyclecounter();
		cycles_dec[i] = t1 - t0;

        if(!vec_equal(key1, key2, SHARED_SECRET_BYTES)) {
            printf("[%d] Error: key1 != key2\n", i );
            passed = 0;
            break;
        }
#if defined(TEST_CACHEKEY)
        cache_hqcpke_sk( csk , sk );
        cache_hqcpke_pk( cpk , pk );
        
		t0 = get_cyclecounter();
        cache_hqckem_enc(ct, key1, cpk);
		t1 = get_cyclecounter();
		cycles_cenc[i] = t1 - t0;

		t0 = get_cyclecounter();
	    cache_hqckem_dec(key2, ct, csk);
		t1 = get_cyclecounter();
		cycles_cdec[i] = t1 - t0;


        if(!vec_equal(key1, key2, SHARED_SECRET_BYTES)) {
            printf("[%d] Error: c key1 != c key2\n", i );
            passed = 0;
            break;
        }
#endif
    }

	printf("\n\nsecret1: ");
	for(int i = 0 ; i < SHARED_SECRET_BYTES ; ++i) printf("%x", key1[i]);

	printf("\nsecret2: ");
	for(int i = 0 ; i < SHARED_SECRET_BYTES ; ++i) printf("%x", key2[i]);
	printf("\n\n");

    printf("TEST [%d] %s\n", TEST_RUN ,  passed ? "PASSED" : "FAILED");

	qsort(cycles_keygen, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
	print_median("keygen", cycles_keygen);
	qsort(cycles_enc, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
	print_median("enc", cycles_enc);
	qsort(cycles_dec, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
	print_median("dec", cycles_dec);
#if defined(TEST_CACHEKEY)
	qsort(cycles_ckeygen, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
	print_median("c_keygen", cycles_ckeygen);
	qsort(cycles_cenc, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
	print_median("c_enc", cycles_cenc);
	qsort(cycles_cdec, TEST_RUN, sizeof(uint64_t), cmp_uint64_t);
	print_median("c_dec", cycles_cdec);
#endif
	disable_cyclecounter();

	return 0;
}