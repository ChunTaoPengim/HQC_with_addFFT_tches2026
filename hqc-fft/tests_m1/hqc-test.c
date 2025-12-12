#include <stdio.h>
#include "api.h"
#include "parameters.h"
#include "benchmark.h"

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

int main() {

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

    struct benchmark bm_k, bm_e, bm_d;
    bm_init(&bm_k);
    bm_init(&bm_e);
    bm_init(&bm_d);
#if defined(TEST_CACHEKEY)
    struct benchmark bm_ck, bm_ce, bm_cd;
    bm_init(&bm_ck);
    bm_init(&bm_ce);
    bm_init(&bm_cd);
#endif
    char mesg[256];

    int passed = 1;
    for(int i=0;i<TEST_RUN;i++) {
#if defined(TEST_CACHEKEY)
        BENCHMARK( bm_ck , {
    	cache_hqckem_keypair(pk, sk, cpk, csk); });
#endif
        BENCHMARK( bm_k , {
    	crypto_kem_keypair(pk, sk); });
        BENCHMARK( bm_e , {
        crypto_kem_enc(ct, key1, pk); });
        BENCHMARK( bm_d , {
	    crypto_kem_dec(key2, ct, sk); });

        if(!vec_equal(key1, key2, SHARED_SECRET_BYTES)) {
            printf("[%d] Error: key1 != key2\n", i );
            passed = 0;
            break;
        }
#if defined(TEST_CACHEKEY)
        cache_hqcpke_sk( csk , sk );
        cache_hqcpke_pk( cpk , pk );
        BENCHMARK( bm_ce , {
        cache_hqckem_enc(ct, key1, cpk); });
        BENCHMARK( bm_cd , {
	    cache_hqckem_dec(key2, ct, csk); });


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

    bm_dump(mesg,sizeof(mesg),&bm_k);
    printf("Keygen: %s\n", mesg);
    bm_dump(mesg,sizeof(mesg),&bm_e);
    printf("Encaps: %s\n", mesg);
    bm_dump(mesg,sizeof(mesg),&bm_d);
    printf("Decaps: %s\n", mesg);

    printf("\n");
#if defined(TEST_CACHEKEY)
    bm_dump(mesg,sizeof(mesg),&bm_ck);
    printf("c Keygen: %s\n", mesg);
    bm_dump(mesg,sizeof(mesg),&bm_ce);
    printf("c Encaps: %s\n", mesg);
    bm_dump(mesg,sizeof(mesg),&bm_cd);
    printf("c Decaps: %s\n", mesg);
#endif

    printf("TEST [%d] %s\n", TEST_RUN ,  passed ? "PASSED" : "FAILED");

	return 0;
}