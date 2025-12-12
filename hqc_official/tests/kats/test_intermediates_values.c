#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "api.h"
#include "parameters.h"
#include "symmetric.h"

int test_kem_api(void);

static void init_randomness(void) {
    unsigned char entropy_input[48] = {0};
#ifdef VERBOSE
    for (int i = 0; i < 48; i++) entropy_input[i] = i;
    prng_init(entropy_input, NULL, 48, 0);
#endif
}

int test_kem_api(void) {
    init_randomness();

#ifdef VERBOSE
    printf("\n*********\n");
    printf("  %s\n", CRYPTO_ALGNAME);
    printf("*********\n");
    printf("\n");
    printf("N: %d   ", PARAM_N);
    printf("N1: %d   ", PARAM_N1);
    printf("N2: %d   ", PARAM_N2);
    printf("OMEGA: %d   ", PARAM_OMEGA);
    printf("OMEGA_R: %d   ", PARAM_OMEGA_R);
    printf("Failure rate: 2^-%d   ", PARAM_DFR_EXP);
    printf("Sec: %d bits", PARAM_SECURITY);
    printf("\n");
#endif

    unsigned char pk[PUBLIC_KEY_BYTES] = {0};
    unsigned char sk[SECRET_KEY_BYTES] = {0};
    unsigned char ct[CIPHERTEXT_BYTES] = {0};
    unsigned char ss1[SHARED_SECRET_BYTES] = {0};
    unsigned char ss2[SHARED_SECRET_BYTES] = {0};

    crypto_kem_keypair(pk, sk);
    crypto_kem_enc(ct, ss1, pk);
    crypto_kem_dec(ss2, ct, sk);

    if (memcmp(ss1, ss2, SHARED_SECRET_BYTES)) {
        printf("\n");
        for (size_t i = 0; i < SHARED_SECRET_BYTES; i++) {
            printf("%02x", ss1[i]);
        }
        printf("\n");
        for (size_t i = 0; i < SHARED_SECRET_BYTES; i++) {
            printf("%02x", ss2[i]);
        }
        printf("ERROR KEM\n");
        exit(1);
    }

#ifdef VERBOSE
    printf("\n\nsecret1: ");
    for (int i = 0; i < SHARED_SECRET_BYTES; ++i) printf("%02x", ss1[i]);

    printf("\nsecret2: ");
    for (int i = 0; i < SHARED_SECRET_BYTES; ++i) printf("%02x", ss2[i]);
    printf("\n\n");
#endif

    return 0;
}

int main(void) {
    if (test_kem_api()) {
        printf("test_kem_api failed\n");
        return 1;
    }
    return 0;
}
