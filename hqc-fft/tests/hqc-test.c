#include <stdio.h>
#include "api.h"
#include "parameters.h"

// #define TEST_CACHEKEY

#define DO_BENCHMARK

#if defined(TEST_CACHEKEY)
#include "cachekey.h"
#endif

#if defined(DO_BENCHMARK)
#include "benchmark.h"
#else

#include "stdint.h"

static inline
void report(char *buf, size_t bufsize, void *recs, unsigned len) {
    sprintf(buf,"");
    (void) bufsize;
    (void) recs;
    (void) len;
}

#define REC_TIMING(recs,len,call) do { \
        call; \
    } while (0)

#endif

int vec_equal(unsigned char *a, unsigned char *b, int len) {
    for(int i = 0 ; i < len ; ++i) {
        if(a[i] != b[i]) return 0;
    }
    return 1;
}

#define TEST_RUN (1000)

int main(void) {

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
    printf("PK: %d:%ld bytes", PUBLIC_KEY_BYTES, sizeof(pk_t));
	printf(" SK: %d:%ld bytes", SECRET_KEY_BYTES, sizeof(sk_t));
    printf("\nS: %d bytes, S_FFT: %d bytes, S_FFT_RLONLE: %d bytes", VEC_N_SIZE_BYTES, R_FFTFORM_BYTES , R_FFTFORM_RLONLY_BYTES );
	printf("\nCPK: %d:%ld bytes", CACHE_PUBLICKEYBYTES, sizeof(cached_pk_t));
	printf(" CSK: %d:%ld bytes", CACHE_SECRETKEYBYTES, sizeof(cached_sk_t));
    printf("\n");
    unsigned char cpk[CACHE_PUBLICKEYBYTES];
	unsigned char csk[CACHE_SECRETKEYBYTES];
#endif
	printf("\n");

	unsigned char pk[PUBLIC_KEY_BYTES];
	unsigned char sk[SECRET_KEY_BYTES];
	unsigned char ct[CIPHERTEXT_BYTES];
	unsigned char key1[SHARED_SECRET_BYTES];
	unsigned char key2[SHARED_SECRET_BYTES];

    uint64_t rec_k[TEST_RUN] = {0}; unsigned len_k[1] = {0};
    uint64_t rec_e[TEST_RUN] = {0}; unsigned len_e[1] = {0};
    uint64_t rec_d[TEST_RUN] = {0}; unsigned len_d[1] = {0};
    uint64_t rec_ck[TEST_RUN] = {0}; unsigned len_ck[1] = {0};
    uint64_t rec_ce[TEST_RUN] = {0}; unsigned len_ce[1] = {0};
    uint64_t rec_cd[TEST_RUN] = {0}; unsigned len_cd[1] = {0};
    char mesg[256];
#if defined(DO_BENCHMARK)
    bm_init(NULL);
#endif

    int passed = 1;
    for(int i=0;i<TEST_RUN;i++) {
#if defined(TEST_CACHEKEY)
        REC_TIMING( rec_ck , len_ck , {
    	cache_hqckem_keypair(pk, sk, cpk, csk); });
#endif
        REC_TIMING( rec_k , len_k , {
    	crypto_kem_keypair(pk, sk); });
        REC_TIMING( rec_e , len_e , {
        crypto_kem_enc(ct, key1, pk); });
        REC_TIMING( rec_d , len_d , {
	    crypto_kem_dec(key2, ct, sk); });

        if(!vec_equal(key1, key2, SHARED_SECRET_BYTES)) {
            printf("[%d] Error: key1 != key2\n", i );
            passed = 0;
            break;
        }

#if defined(TEST_CACHEKEY)
        cache_hqcpke_sk( csk , sk );
        cache_hqcpke_pk( cpk , pk );
        REC_TIMING( rec_ce , len_ce , {
        cache_hqckem_enc(ct, key1, cpk); });
        REC_TIMING( rec_cd , len_cd , {
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

    report(mesg,sizeof(mesg),rec_k,len_k[0]);
    printf("Keygen: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_e,len_e[0]);
    printf("Encaps: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_d,len_d[0]);
    printf("Decaps: %s\n", mesg);
    printf("\n");
#if defined(TEST_CACHEKEY)
    report(mesg,sizeof(mesg),rec_ck,len_ck[0]);
    printf("c Keygen: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_ce,len_ce[0]);
    printf("c Encaps: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_cd,len_cd[0]);
    printf("c Decaps: %s\n", mesg);
#endif
    printf("TEST [%d] %s\n", TEST_RUN ,  passed ? "PASSED" : "FAILED");

	return 0;
}
