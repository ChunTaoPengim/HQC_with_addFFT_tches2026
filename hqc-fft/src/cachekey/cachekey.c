/**
 * @file api_chachekey.c
 * @brief Implementation of api_cachekey.h
 */


#include "parameters.h"
#include "gf2x.h"
#include "parsing.h"
#include "symmetric.h"
#include "code.h"
#include "vector.h"
#include "cachekey.h"
#include "crypto_memset.h"
#include <stdint.h>
#include <string.h>
#ifdef VERBOSE
#include <stdio.h>
#endif


#include "hqc.h"


void cache_hqcpke_pk( unsigned char * cpk , const unsigned char * pk )
{
    uint64_t h[VEC_N_SIZE_64] = {0};
    uint64_t s[VEC_N_SIZE_64] = {0};

    hqc_ek_pke_from_string(h, s, pk);

    cached_pk_t * _cpk = (cached_pk_t *)cpk;
    memcpy( &(_cpk->pk), pk, sizeof(_cpk->pk) ); // pk
    ring_to_fftform_rlonly( _cpk->s_fft , s );
    ring_to_fftform( _cpk->h_fft , h );
}


void cache_hqcpke_sk( unsigned char * csk , const unsigned char * sk )
{

    uint64_t y[VEC_N_SIZE_64] = {0};
    cached_sk_t * _csk = (cached_sk_t *)csk;
    //memcpy( csk, sk, SECRET_KEY_BYTES);
    memcpy( &(_csk->sk) , sk , sizeof(_csk->sk) );

    hqc_dk_pke_from_string(y, sk + PUBLIC_KEY_BYTES);
    ring_to_fftform( _csk->y_fft , y );

    memset_zero(y, sizeof y);
    uint64_t *s = y;
    uint64_t h[VEC_N_SIZE_64] = {0};
    hqc_ek_pke_from_string(h, s, sk );  // ek_pke is the same address as sk.

    ring_to_fftform_rlonly( _csk->s_fft , s );
    ring_to_fftform( _csk->h_fft , h );

}

void cache_hqcpke_keygen(unsigned char* pk, unsigned char* sk, unsigned char * cpk, unsigned char* csk, uint8_t *seed)
{
    uint8_t keypair_seed[2 * SEED_BYTES] = {0};
    uint8_t *seed_dk = keypair_seed;
    uint8_t *seed_ek = keypair_seed + SEED_BYTES;
    shake256_xof_ctx dk_xof_ctx = {0};
    shake256_xof_ctx ek_xof_ctx = {0};

    uint64_t x[VEC_N_SIZE_64] = {0};
    uint64_t y[VEC_N_SIZE_64] = {0};
    uint64_t h[VEC_N_SIZE_64] = {0};
    uint64_t s[VEC_N_SIZE_64] = {0};

    // Derive keypair seeds
    hash_i(keypair_seed, seed);

    // Compute decryption key
    xof_init(&dk_xof_ctx, seed_dk, SEED_BYTES);
    vect_sample_fixed_weight1(&dk_xof_ctx, y, PARAM_OMEGA);
    vect_sample_fixed_weight1(&dk_xof_ctx, x, PARAM_OMEGA);

    // Compute encryption key
    xof_init(&ek_xof_ctx, seed_ek, SEED_BYTES);
    vect_set_random(&ek_xof_ctx, h);

    // Compute public key
    cached_sk_t * _csk = (cached_sk_t *)csk;
    // vect_mul(s, y, h);
    ring_to_fftform( _csk->h_fft, h );
    ring_to_fftform( _csk->y_fft, y );
    ring_mul_fftformx2( s , _csk->h_fft , _csk->y_fft );
    vect_add(s, x, s, VEC_N_SIZE_64);
    ring_to_fftform_rlonly( _csk->s_fft, s );

    // Parse encryption key to string
    pk_t * _pk = (pk_t *)pk;
    memcpy( _pk->seed_ek , seed_ek , sizeof(_pk->seed_ek) );
    memcpy( _pk->s , s , sizeof(_pk->s) );

    sk_t * _sk = (sk_t *)sk;
    memcpy(sk, pk, sizeof(pk_t) );
    memcpy( _sk->seed_dk , seed_dk , sizeof(_sk->seed_dk) );

    // cpk
    cached_pk_t * _cpk = (cached_pk_t *)cpk;
    memcpy( &(_cpk->pk) , pk , sizeof(_cpk->pk) );
    memcpy( _cpk->s_fft , _csk->s_fft , sizeof(_cpk->s_fft) );
    memcpy( _cpk->h_fft , _csk->h_fft , sizeof(_cpk->h_fft) );

    #ifdef VERBOSE
        printf("\n\nsk_seed: "); for(int i = 0 ; i < SEED_BYTES ; ++i) printf("%02x", sk_seed[i]);
        printf("\n\nsigma: "); for(int i = 0 ; i < VEC_K_SIZE_BYTES ; ++i) printf("%02x", sigma[i]);
        printf("\n\nx: "); vect_print(x, VEC_N_SIZE_BYTES);
        printf("\n\ny: "); vect_print(y, VEC_N_SIZE_BYTES);

        printf("\n\npk_seed: "); for(int i = 0 ; i < SEED_BYTES ; ++i) printf("%02x", pk_seed[i]);
        printf("\n\nh: "); vect_print(h, VEC_N_SIZE_BYTES);
        printf("\n\ns: "); vect_print(s, VEC_N_SIZE_BYTES);

        //printf("\n\nsk: "); for(int i = 0 ; i < SECRET_KEY_BYTES ; ++i) printf("%02x", sk[i]);
        printf("\n\npk: "); for(int i = 0 ; i < PUBLIC_KEY_BYTES ; ++i) printf("%02x", pk[i]);
    #endif

    // Zeroize sensitive data
    memset_zero(keypair_seed, sizeof keypair_seed);
    memset_zero(x, sizeof x);
    memset_zero(y, sizeof y);
    memset_zero(&dk_xof_ctx, sizeof dk_xof_ctx);
}

void cache_hqcpke_encrypt(ciphertext_pke_t *c_pke, const uint8_t *_cpk, const uint64_t *m, const uint8_t *theta)
{
    shake256_xof_ctx theta_xof_ctx = {0};
    uint64_t r1[VEC_N_SIZE_64] = {0};
    uint64_t r2[VEC_N_SIZE_64] = {0};
    uint64_t e[VEC_N_SIZE_64] = {0};
    uint64_t tmp1[VEC_N_SIZE_64];
    uint64_t tmp2[VEC_N_SIZE_64];
    uint64_t tmp3[VEC_N_SIZE_64];

    // Initialize Xof using theta
    xof_init(&theta_xof_ctx, theta, SEED_BYTES);

    // Retrieve h and s from public key
    //hqc_public_key_from_string(h, s, pk); // skip this step since they are already in cached_pk_t
    const cached_pk_t * cpk = (const cached_pk_t *)_cpk;

    // Generate re, e and r1
    vect_sample_fixed_weight2(&theta_xof_ctx, r2, PARAM_OMEGA_R);
    vect_sample_fixed_weight2(&theta_xof_ctx, e, PARAM_OMEGA_E);
    vect_sample_fixed_weight2(&theta_xof_ctx, r1, PARAM_OMEGA_R);
    // Compute u = r1 + r2.h
    //vect_mul(u, r2, h);
    //ring_mul_x2_fftform( tmp1 , tmp2 , r2 , (uint64_t*)cpk->h_fft , (uint64_t*)cpk->s_fft);
    ring_mul_x2_fftform( tmp1 , tmp2 , r2 , cpk->h_fft , cpk->s_fft , (cpk->pk).s );
    vect_add(c_pke->u, r1, tmp1, VEC_N_SIZE_64);

    // Compute v = m.G by encoding the message
    code_encode(c_pke->v, m);

    // Compute v = C.encode(m) + Truncate(s.r2 + e)
    vect_add(tmp3, e, tmp2, VEC_N_SIZE_64);
    vect_truncate(tmp3);
    vect_add(c_pke->v, c_pke->v, tmp3, VEC_N1N2_SIZE_64);

    #ifdef VERBOSE
        //printf("\n\nh: "); vect_print(h, VEC_N_SIZE_BYTES);
        //printf("\n\ns: "); vect_print(s, VEC_N_SIZE_BYTES);
        printf("\n\nr1: "); vect_print(r1, VEC_N_SIZE_BYTES);
        printf("\n\nr2: "); vect_print(r2, VEC_N_SIZE_BYTES);
        printf("\n\ne: "); vect_print(e, VEC_N_SIZE_BYTES);
        printf("\n\nnon-truncated v: "); vect_print(tmp2, VEC_N_SIZE_BYTES);

        printf("\n\nu: "); vect_print(u, VEC_N_SIZE_BYTES);
        printf("\n\nv: "); vect_print(v, VEC_N1N2_SIZE_BYTES);
    #endif

    // Zeroize sensitive data
    memset_zero(r1, sizeof r1);
    memset_zero(r2, sizeof r2);
    memset_zero(e, sizeof e);
    // memset_zero(tmp, sizeof tmp);
    memset_zero(&theta_xof_ctx, sizeof theta_xof_ctx);
    memset_zero(tmp1, sizeof tmp1);
    memset_zero(tmp2, sizeof tmp2);
    memset_zero(tmp3, sizeof tmp3);

}

uint8_t cache_hqcpke_decrypt(uint64_t *m, const uint8_t *_csk, const ciphertext_pke_t *c_pke)
{
    uint64_t tmp1[VEC_N_SIZE_64] = {0};
    uint64_t tmp2[VEC_N_SIZE_64] = {0};

    // Retrieve x, y, pk from secret key
    const cached_sk_t * csk = (const cached_sk_t *)_csk;

    // Compute v - u.y
    //vect_mul(tmp2, y, u);
    ring_mul_fftform( tmp1 , c_pke->u , csk->y_fft );
    vect_truncate(tmp1);
    // Compute v - Truncate(u.y)
    vect_add(tmp2, c_pke->v, tmp1, VEC_N1N2_SIZE_64);

    #ifdef VERBOSE
        printf("\n\nu: "); vect_print(u, VEC_N_SIZE_BYTES);
        printf("\n\nv: "); vect_print(v, VEC_N1N2_SIZE_BYTES);
        //printf("\n\ny: "); vect_print(y, VEC_N_SIZE_BYTES);
        printf("\n\nv - u.y: "); vect_print(tmp2, VEC_N_SIZE_BYTES);
    #endif

    // Compute plaintext m
    code_decode(m, tmp2);

    // Zeroize sensitive data
    memset_zero(tmp1, sizeof tmp1);
    memset_zero(tmp2, sizeof tmp2);

    return 0;
}





///////////////////////////////////////////////////////////////////////

#include "parsing.h"
#include "vector.h"


// int crypto_kem_keypair(unsigned char *pk, unsigned char *sk) {
int cache_hqckem_keypair(unsigned char* pk, unsigned char* sk , unsigned char *cpk, unsigned char* csk) {
    #ifdef VERBOSE
        printf("\n\n\n\n### KEYGEN ###");
    #endif

    uint8_t seed_kem[SEED_BYTES] = {0};
    uint8_t sigma[PARAM_SECURITY_BYTES] = {0};
    uint8_t seed_pke[SEED_BYTES] = {0};
    shake256_xof_ctx ctx_kem;

    // uint8_t ek_pke[PUBLIC_KEY_BYTES] = {0};
    // uint8_t dk_pke[SECRET_KEY_BYTES] = {0};

    // Sample seed_kem
    prng_get_bytes(seed_kem, SEED_BYTES);

    // Compute seed_pke and randomness sigma
    xof_init(&ctx_kem, seed_kem, SEED_BYTES);
    xof_get_bytes(&ctx_kem, seed_pke, SEED_BYTES);
    xof_get_bytes(&ctx_kem, sigma, PARAM_SECURITY_BYTES);

    // Compute HQC-PKE keypair
    cache_hqcpke_keygen(pk, sk, cpk, csk, seed_pke);

    // Compute HQC-KEM keypair
    sk_t * _sk = (sk_t*) sk;
    memcpy( _sk->sigma , sigma , sizeof(_sk->sigma) );
    memcpy( _sk->seed_kem , seed_kem , sizeof(_sk->seed_kem) );
    cached_sk_t * _csk = (cached_sk_t *)csk;
    memcpy( &(_csk->sk) , sk , sizeof(_csk->sk) );

    // Zeroize sensitive data
    memset_zero(seed_kem, sizeof seed_kem);
    memset_zero(sigma, sizeof sigma);
    memset_zero(seed_pke, sizeof seed_pke);

    return 0;
}



//int crypto_kem_enc(unsigned char *ct, unsigned char *ss, const unsigned char *pk) {
int cache_hqckem_enc(unsigned char* ct, unsigned char* ss, const unsigned char* cpk) {
    #ifdef VERBOSE
        printf("\n\n\n\n### ENCAPS ###");
    #endif

    uint64_t m_u64[(PARAM_SECURITY_BYTES+7)/8] = {0};
    uint8_t * m = (uint8_t *) m_u64;
    uint8_t K_theta[SHARED_SECRET_BYTES + SEED_BYTES] = {0};
    uint8_t theta[SEED_BYTES] = {0};
    uint8_t hash_ek_kem[SEED_BYTES] = {0};
    ciphertext_kem_t c_kem_t = {0};

    // Sample message m and salt
    prng_get_bytes(m, PARAM_SECURITY_BYTES);
    prng_get_bytes(c_kem_t.salt, SALT_BYTES);

    
    // Compute shared key K and ciphertext c_kem
    hash_h(hash_ek_kem, (const uint8_t *)&(((const cached_pk_t *)cpk)->pk));
    hash_g(K_theta, hash_ek_kem, m, c_kem_t.salt);
    memcpy(theta, K_theta + SEED_BYTES, SEED_BYTES);

    // Encrypting m
    //hqc_pke_encrypt(u, v, (uint64_t *)m, theta, pk);
    cache_hqcpke_encrypt(&c_kem_t.c_pke, cpk, m_u64, theta);

    // Computing shared secret
    hqc_c_kem_to_string(ct, &c_kem_t);
    memcpy(ss, K_theta, SHARED_SECRET_BYTES);

    #ifdef VERBOSE
        //printf("\n\npk: "); for(int i = 0 ; i < PUBLIC_KEY_BYTES ; ++i) printf("%02x", pk[i]);
        printf("\n\nm: "); vect_print((uint64_t *)m, VEC_K_SIZE_BYTES);
        printf("\n\ntheta: "); for(int i = 0 ; i < SHAKE256_512_BYTES ; ++i) printf("%02x", theta[i]);
        printf("\n\nciphertext: "); for(int i = 0 ; i < CIPHERTEXT_BYTES ; ++i) printf("%02x", ct[i]);
        printf("\n\nsecret 1: "); for(int i = 0 ; i < SHARED_SECRET_BYTES ; ++i) printf("%02x", ss[i]);
    #endif

    // Zeroize sensitive data
    memset_zero(m_u64, sizeof m_u64);
    memset_zero(K_theta, sizeof K_theta);
    memset_zero(theta, sizeof theta);

    return 0;
}


//int crypto_kem_dec(unsigned char *ss, const unsigned char *ct, const unsigned char *sk) {
int cache_hqckem_dec(unsigned char* ss, const unsigned char* ct, const unsigned char* csk) {
    #ifdef VERBOSE
        printf("\n\n\n\n### DECAPS ###");
    #endif

    //uint8_t ek_pke[CACHE_PUBLICKEYBYTES] = {0};
    uint8_t sigma[PARAM_SECURITY_BYTES] = {0};
    uint64_t m_prime_u64[(PARAM_SECURITY_BYTES+7)/8] = {0};
    uint8_t * m_prime = (uint8_t*) m_prime_u64;
    uint8_t hash_ek_kem[SEED_BYTES] = {0};
    uint8_t K_theta_prime[SHARED_SECRET_BYTES + SEED_BYTES] = {0};
    uint8_t K_bar[SHARED_SECRET_BYTES] = {0};
    uint8_t theta_prime[SEED_BYTES] = {0};
    ciphertext_kem_t c_kem_t = {0};
    ciphertext_kem_t c_kem_prime_t = {0};
    uint8_t result;

    // Parse decapsulation key dk_kem
    const cached_sk_t * _csk = (const cached_sk_t*) csk;
    cached_pk_t cpk;
    memcpy( &(cpk.pk), &((_csk->sk).pk), sizeof(cpk.pk) );
    memcpy( cpk.s_fft, _csk->s_fft, sizeof( cpk.s_fft) );
    memcpy( cpk.h_fft, _csk->h_fft, sizeof( cpk.h_fft) );
    memcpy( sigma , (_csk->sk).sigma , PARAM_SECURITY_BYTES );

    // Parse ciphertext c_kem
    hqc_c_kem_from_string(&c_kem_t.c_pke, c_kem_t.salt, ct);

    // Compute message m_prime
    result = cache_hqcpke_decrypt(m_prime_u64, csk, &c_kem_t.c_pke);

    // Compute shared key K_prime and ciphertext c_kem_prime
    hash_h(hash_ek_kem, (const uint8_t *)&cpk.pk);
    hash_g(K_theta_prime, hash_ek_kem, m_prime, c_kem_t.salt);
    memcpy(ss, K_theta_prime, SHARED_SECRET_BYTES);
    memcpy(theta_prime, K_theta_prime + SHARED_SECRET_BYTES, SEED_BYTES);

    cache_hqcpke_encrypt(&c_kem_prime_t.c_pke, (const uint8_t*)&cpk, m_prime_u64, theta_prime);
    memcpy(c_kem_prime_t.salt, c_kem_t.salt, SALT_BYTES);

    // Compute rejection key K_bar
    hash_j(K_bar, hash_ek_kem, sigma, &c_kem_t);
    result |= vect_compare((uint8_t *)c_kem_t.c_pke.u, (uint8_t *)c_kem_prime_t.c_pke.u, VEC_N_SIZE_BYTES);
    result |= vect_compare((uint8_t *)c_kem_t.c_pke.v, (uint8_t *)c_kem_prime_t.c_pke.v, VEC_N1N2_SIZE_BYTES);
    result |= vect_compare(c_kem_t.salt, c_kem_prime_t.salt, SALT_BYTES);
    result -= 1;
    for (size_t i = 0; i < SHARED_SECRET_BYTES; ++i) {
        ss[i] = (ss[i] & result) ^ (K_bar[i] & ~result);
    }



    #ifdef VERBOSE
        //printf("\n\npk: "); for(int i = 0 ; i < PUBLIC_KEY_BYTES ; ++i) printf("%02x", pk[i]);
        //printf("\n\nsk: "); for(int i = 0 ; i < SECRET_KEY_BYTES ; ++i) printf("%02x", sk[i]);
        printf("\n\nciphertext: "); for(int i = 0 ; i < CIPHERTEXT_BYTES ; ++i) printf("%02x", ct[i]);
        printf("\n\nm: "); vect_print((uint64_t *)m, VEC_K_SIZE_BYTES);
        printf("\n\ntheta: "); for(int i = 0 ; i < SHAKE256_512_BYTES ; ++i) printf("%02x", theta[i]);
        printf("\n\n\n# Checking Ciphertext- Begin #");
        printf("\n\nu2: "); vect_print(u2, VEC_N_SIZE_BYTES);
        printf("\n\nv2: "); vect_print(v2, VEC_N1N2_SIZE_BYTES);
        printf("\n\n# Checking Ciphertext - End #\n");
    #endif

     // Zeroize sensitive data
    memset_zero(sigma, sizeof sigma);
    memset_zero(m_prime, sizeof m_prime);
    memset_zero(K_theta_prime, sizeof K_theta_prime);
    memset_zero(K_bar, sizeof K_bar);
    memset_zero(theta_prime, sizeof theta_prime);

    return 0;
}




