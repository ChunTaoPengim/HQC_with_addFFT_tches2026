#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "api.h"
#include "parameters.h"
#include "symmetric.h"
#include "crypto_memset.h"

static void init_randomness(void) {

#ifdef VERBOSE
    unsigned char entropy_input[48] = {0};
    for (int i = 0; i < 48; i++) entropy_input[i] = (unsigned char)i;
    prng_init(entropy_input, NULL, sizeof entropy_input, 0);
#else
    unsigned char seed[32] = {0};
    size_t filled = 0;
    while (filled < sizeof seed) {
        ssize_t got = syscall(SYS_getrandom, seed + filled, sizeof seed - filled, 0);
        if (got < 0) {
            perror("getrandom");
            exit(EXIT_FAILURE);
        }
        filled += (size_t)got;
    }
    prng_init(seed, NULL, sizeof seed, 0);
#endif
}

int main(void) {

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
    unsigned char key1[SHARED_SECRET_BYTES] = {0};
    unsigned char key2[SHARED_SECRET_BYTES] = {0};

    crypto_kem_keypair(pk, sk);
    crypto_kem_enc(ct, key1, pk);
    crypto_kem_dec(key2, ct, sk);

    printf("\n\nsecret1: ");
    for (int i = 0; i < SHARED_SECRET_BYTES; ++i) printf("%02x", key1[i]);

    printf("\nsecret2: ");
    for (int i = 0; i < SHARED_SECRET_BYTES; ++i) printf("%02x", key2[i]);
    printf("\n\n");

    // Zeroize sensitive data
    memset_zero(sk, sizeof sk);
    memset_zero(key1, sizeof key1);
    memset_zero(key2, sizeof key2);

    return 0;
}
