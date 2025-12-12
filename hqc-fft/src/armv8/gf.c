/**
 * @file gf.c
 * @brief Galois field implementation with multiplication using the pclmulqdq instruction
 */

#include "gf.h"
#include "parameters.h"
#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>


/**
 * @brief Generates exp and log lookup tables of GF(2^m).
 *
 * @note   this function is not used in the code; it was used to generate
 *         the lookup table for GF(2^8).
 *
 * The logarithm of 0 is defined as 2^PARAM_M by convention. <br>
 * The last two elements of the exp table are needed by the gf_mul function from gf_lutmul.c
 * (for example if both elements to multiply are zero).
 * @param[out] exp Array of size 2^PARAM_M + 2 receiving the powers of the primitive element
 * @param[out] log Array of size 2^PARAM_M receiving the logarithms of the elements of GF(2^m)
 * @param[in] m Parameter of Galois field GF(2^m)
 */
void gf_generate(uint16_t *exp, uint16_t *log, const int16_t m) {
    uint16_t elt = 1;
    uint16_t alpha = 2;  // primitive element of GF(2^PARAM_M)
    uint16_t gf_poly = PARAM_GF_POLY;

    for (size_t i = 0; i < (1U << m) - 1; ++i) {
        exp[i] = elt;
        log[elt] = i;

        elt *= alpha;
        if (elt >= 1 << m)
            elt ^= gf_poly;
    }

    exp[(1 << m) - 1] = 1;
    exp[1 << m] = 2;
    exp[(1 << m) + 1] = 4;
    log[0] = 0;  // by convention
}


#include "gf_neon.h"


/**
 * Multiplies two elements of GF(2^GF_M).
 * @returns the product a*b
 * @param[in] a Element of GF(2^GF_M)
 * @param[in] b Element of GF(2^GF_M)
 */
uint16_t gf_mul(uint16_t a, uint16_t b) {
    uint8x16_t mask_f = vdupq_n_u8( 0xf );
    uint8x16_t tab_rd0 = vld1q_u8(__gf256_bit_8_11_reduce);
    uint8x16_t tab_rd1 = vld1q_u8(__gf256_bit_12_15_reduce);
    uint8x16_t a_u8 = vdupq_n_u8(a&0xff);
    uint8x16_t b_u8 = vdupq_n_u8(b&0xff);
    uint8x16_t rr = _gf256v_mul_neon( a_u8 , b_u8 , mask_f , tab_rd0 , tab_rd1 );
    return vgetq_lane_u8(rr,0);
}



/**
 * Squares an element of GF(2^8).
 * @returns a^2
 * @param[in] a Element of GF(2^8)
 */
uint16_t gf_square(uint16_t a) {
    uint8x16_t mask_f = vdupq_n_u8( 0xf );
    uint8x16_t tab_rd0 = vld1q_u8(__gf256_bit_8_11_reduce);
    uint8x16_t tab_rd1 = vld1q_u8(__gf256_bit_12_15_reduce);
    uint8x16_t a_u8 = vdupq_n_u8(a&0xff);
    uint8x16_t rr = _gf256v_mul_neon( a_u8 , a_u8 , mask_f , tab_rd0 , tab_rd1 );
    return vgetq_lane_u8(rr,0);
}

/**
 * Computes the inverse of an element of GF(2^8),
 * using the addition chain 1 2 3 4 7 11 15 30 60 120 127 254
 * @returns the inverse of a
 * @param[in] a Element of GF(2^8)
 */
uint16_t gf_inverse(uint16_t a) {
    uint16_t inv = a;
    uint16_t tmp1, tmp2;

    inv = gf_square(a);       /* a^2 */
    tmp1 = gf_mul(inv, a);    /* a^3 */
    inv = gf_square(inv);     /* a^4 */
    tmp2 = gf_mul(inv, tmp1); /* a^7 */
    tmp1 = gf_mul(inv, tmp2); /* a^11 */
    inv = gf_mul(tmp1, inv);  /* a^15 */
    inv = gf_square(inv);     /* a^30 */
    inv = gf_square(inv);     /* a^60 */
    inv = gf_square(inv);     /* a^120 */
    inv = gf_mul(inv, tmp2);  /* a^127 */
    inv = gf_square(inv);     /* a^254 */
    return inv;
}

