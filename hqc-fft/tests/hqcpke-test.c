#include <stdio.h>
#include "api.h"

#include "parameters.h"

#include "hqc.h"
#include "api_cachekey.h"

#include "benchmark.h"

int vec_equal(unsigned char *a, unsigned char *b, int len) {
    for(int i = 0 ; i < len ; ++i) {
        if(a[i] != b[i]) return 0;
    }
    return 1;
}

#define TEST_RUN (100)

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
	printf(" PK: %d bytes", PUBLIC_KEY_BYTES);
	printf(" SK: %d bytes", SECRET_KEY_BYTES);
	printf("CPK: %d bytes", CACHE_PUBLICKEYBYTES);
	printf("CSK: %d bytes", CACHE_SECRETKEYBYTES);
    printf("\n");

	unsigned char pk[PUBLIC_KEY_BYTES];
	unsigned char sk[SECRET_KEY_BYTES];
	unsigned char cpk[CACHE_PUBLICKEYBYTES];
	unsigned char csk[CACHE_SECRETKEYBYTES];
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

    int passed = 1;
    for(int i=0;i<TEST_RUN;i++) {
        uint8_t theta[SEED_BYTES] = {0};
        uint64_t m0[(VEC_K_SIZE_BYTES/8)+((VEC_K_SIZE_BYTES%8)?1:0)] = {0};
        uint64_t m1[(VEC_K_SIZE_BYTES/8)+((VEC_K_SIZE_BYTES%8)?1:0)] = {0};
        uint64_t m2[(VEC_K_SIZE_BYTES/8)+((VEC_K_SIZE_BYTES%8)?1:0)] = {0};
        uint64_t u[VEC_N_SIZE_64] = {0};
        uint64_t v[VEC_N1N2_SIZE_64] = {0};
        uint8_t sigma[PARAM_SECURITY_BYTES] = {0};

        for(int j=0;j<sizeof(m0)/sizeof(uint64_t)-1;j++) m0[j] = (i+17)*(j+19);

        REC_TIMING( rec_ck , len_ck , {
    	cache_hqcpke_keygen(pk, cpk, csk); });

        REC_TIMING( rec_k , len_k , {
    	hqc_pke_keygen(pk, sk); });
        REC_TIMING( rec_e , len_e , {
        hqc_pke_encrypt(u, v, m0, theta, pk); });
        REC_TIMING( rec_d , len_d , {
	    hqc_pke_decrypt(m1, sigma, u, v, sk); });

        cache_hqcpke_sk( csk , sk );
        cache_hqcpke_pk( cpk , pk );
        REC_TIMING( rec_ce , len_ce , {
        cache_hqcpke_encrypt(u, v, m0, theta, cpk); });
        REC_TIMING( rec_cd , len_cd , {
	    cache_hqcpke_decrypt(m2, sigma, u, v, csk); });


        if(!vec_equal(m0, m1, VEC_K_SIZE_BYTES)) {
            printf("[%d] Error: key1 != key2\n", i );
            passed = 0;
            break;
        }
        if(!vec_equal(m0, m2, VEC_K_SIZE_BYTES)) {
            printf("[%d] Error: c key1 != c key2\n", i );
            passed = 0;
            break;
        }
    }

	printf("\n\nsecret1: ");
	for(int i = 0 ; i < SHARED_SECRET_BYTES ; ++i) printf("%x", key1[i]);

	printf("\nsecret2: ");
	for(int i = 0 ; i < SHARED_SECRET_BYTES ; ++i) printf("%x", key2[i]);
	printf("\n\n");

    report(mesg,sizeof(mesg),rec_k,len_k[0]);
    printf("Keygen: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_e,len_e[0]);
    printf("Encrypt: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_d,len_d[0]);
    printf("Decrypt: %s\n", mesg);
    printf("\n");

    report(mesg,sizeof(mesg),rec_ck,len_ck[0]);
    printf("c Keygen: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_ce,len_ce[0]);
    printf("c Encrypt: %s\n", mesg);
    report(mesg,sizeof(mesg),rec_cd,len_cd[0]);
    printf("c Decrypt: %s\n", mesg);

    printf("TEST [%d] %s\n", TEST_RUN ,  passed ? "PASSED" : "FAILED");

	return 0;
}
