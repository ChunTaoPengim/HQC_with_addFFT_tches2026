/**
 * @file reed_muller.c
 * @brief Constant time implementation of Reed-Muller code RM(1,7)
 */

#include "reed_muller.h"
#include "parameters.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <arm_neon.h>
#include "data_structures.h"

// number of repeated code words
#define MULTIPLICITY                   CEIL_DIVIDE(PARAM_N2, 128)

// copy bit 0 into all bits of a 64 bit value
#define BIT0MASK(x) (int64_t)(-((x) & 1))
static const uint16_t values[16] = {
    1 << 0, 1 << 1, 1 << 2, 1 << 3,
    1 << 4, 1 << 5, 1 << 6, 1 << 7,
    1 << 8, 1 << 9, 1 << 10, 1 << 11,
    1 << 12, 1 << 13, 1 << 14, 1 << 15
};

static inline uint16x8_t bitmask_from_16(uint16_t x, uint16_t n)
{
    uint16x8_t full_length = vdupq_n_u16(x);

    uint16x8_t bitmask = vld1q_u16(values + n);

    uint16x8_t x_mask = vandq_u16(full_length, bitmask);

    uint16x8_t result = vceqq_u16(x_mask, vdupq_n_u16(0));
    result = vaddq_u16(result, vdupq_n_u16(1));

    return result;
}

void encode(codeword *word, int32_t message);
static inline void expand_and_sum(expandedCodeword *dst, codeword src[]);
static inline void hadamard(expandedCodeword *src, expandedCodeword *dst);
static inline int32_t find_peaks(expandedCodeword *transform);



/**
 * @brief Encode a single byte into a single codeword using RM(1,7)
 *
 * Encoding matrix of this code:
 * bit pattern (note that bits are numbered big endian)
 * 0   aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
 * 1   cccccccc cccccccc cccccccc cccccccc
 * 2   f0f0f0f0 f0f0f0f0 f0f0f0f0 f0f0f0f0
 * 3   ff00ff00 ff00ff00 ff00ff00 ff00ff00
 * 4   ffff0000 ffff0000 ffff0000 ffff0000
 * 5   00000000 ffffffff 00000000 ffffffff
 * 6   00000000 00000000 ffffffff ffffffff
 * 7   ffffffff ffffffff ffffffff ffffffff
 *
 * @param[out] word An RM(1,7) codeword
 * @param[in] message A message to encode
 */
inline void encode(codeword *word, int32_t message) {
    int32_t first_word;
    first_word = BIT0MASK(message >> 7);
    first_word ^= BIT0MASK(message >> 0) & 0xaaaaaaaa;
    first_word ^= BIT0MASK(message >> 1) & 0xcccccccc;
    first_word ^= BIT0MASK(message >> 2) & 0xf0f0f0f0;
    first_word ^= BIT0MASK(message >> 3) & 0xff00ff00;
    first_word ^= BIT0MASK(message >> 4) & 0xffff0000;
    word->u32[0] = first_word;
    first_word ^= BIT0MASK(message >> 5);
    word->u32[1] = first_word;
    first_word ^= BIT0MASK(message >> 6);
    word->u32[3] = first_word;
    first_word ^= BIT0MASK(message >> 5);
    word->u32[2] = first_word;
    return;
}



/**
 * @brief Add multiple codewords into expanded codeword
 *
 * Note: this does not write the codewords as -1 or +1 as the green machine does
 * instead, just 0 and 1 is used.
 * The resulting hadamard transform has:
 * all values are halved
 * the first entry is 64 too high
 *
 * @param[out] dst Structure that contain the expanded codeword
 * @param[in] src Structure that contain the codeword
 */
static inline void expand_and_sum(expandedCodeword *dst, codeword src[]) {
    
#if defined(NEW_CODE)
    for(size_t part = 0; part < 8; part++) {

        uint16x8_t bit_array1 = bitmask_from_16(src->u16[part], 0);
        uint16x8_t bit_array2 = bitmask_from_16(src->u16[part], 8);
        for (size_t copy = 1; copy < MULTIPLICITY; copy++) {
            // bit_array = _mm256_add_epi16(bit_array, bitmask_from_16(src[copy].u16[part]));
            bit_array1 = vaddq_u16(bit_array1, bitmask_from_16(src[copy].u16[part], 0));
            bit_array2 = vaddq_u16(bit_array2, bitmask_from_16(src[copy].u16[part], 8));
        }
        vst1q_u16((uint16_t *) dst -> i16 + part * 16, bit_array1);
        vst1q_u16((uint16_t *) dst -> i16 + part * 16 + 8, bit_array2);
    }
#else
    // start converting the first copy
    for (size_t part = 0; part < 8; part++) {
        for (size_t i = 0; i < 16; ++i) {
            dst->i16[(part << 4) + i] = src->u16[part] >> i & 1;
        }
    }
    // sum the rest of the copies
    for (size_t copy = 1; copy < MULTIPLICITY; copy++) {
        for (size_t part = 0 ; part < 8 ; part++) {
            for (size_t i = 0; i < 16; ++i) {
                dst->i16[(part << 4) + i] += src[copy].u16[part] >> i & 1;
            }
        }
    }
#endif
}



/**
 * @brief Hadamard transform
 *
 * Perform hadamard transform of src and store result in dst
 * src is overwritten: it is also used as intermediate buffer
 *
 * @param[out] src Structure that contain the expanded codeword
 * @param[out] dst Structure that contain the expanded codeword
 */
static inline void hadamard(expandedCodeword *src, expandedCodeword *dst) {
    // the passes move data:
    // src -> dst -> src -> dst -> src -> dst -> src -> dst
    // using p1 and p2 alternately
    expandedCodeword *p1 = src;
    expandedCodeword *p2 = dst;
    for (size_t pass = 0; pass < 7; pass++) {
        for (int32_t i = 0; i < 64; i = i + 8) {
                int16x8_t a1 = vld1q_s16(p1->i16 + 2*i);
                int16x8_t a2 = vld1q_s16(p1->i16 + 2*i + 8);
                
                int16x8_t c1 = vpaddq_s16(a1, a2);
                vst1q_s16(p2->i16 + i, c1);
                int16x8x2_t a_deinterleave = vuzpq_s16(a1, a2);
                c1 = vsubq_s16(a_deinterleave.val[0], a_deinterleave.val[1]);
                vst1q_s16(p2->i16 + i + 64, c1);
        }
        // swap p1, p2 for next round
        expandedCodeword *p3 = p1;
        p1 = p2;
        p2 = p3;
    }
}
/**
 * @brief Finding the location of the highest value
 *
 * This is the final step of the green machine: find the location of the highest value,
 * and add 128 if the peak is positive
 * Notes on decoding
 * The standard "Green machine" decoder works as follows:
 * if the received codeword is W, compute (2 * W - 1) * H7
 * The entries of the resulting vector are always even and vary from
 * -128 (= the complement is a code word, add bit 7 to decode)
 * via 0 (this is a different codeword)
 * to 128 (this is the code word).
 *
 * Our decoding differs in two ways:
 * - We take W instead of 2 * W - 1 (so the entries are 0,1 instead of -1,1)
 * - We take the sum of the repititions (so the entries are 0..MULTIPLICITY)
 * This implies that we have to subtract 64M (M=MULTIPLICITY)
 * from the first entry to make sure the first codewords is handled properly
 * and that the entries vary from -64M to 64M.
 * -64M or 64M stands for a perfect codeword.
 *
 * @param[in] transform Structure that contain the expanded codeword
 */
static inline int32_t find_peaks(expandedCodeword *transform)
{
    
    int16x8_t peak_value = vld1q_s16(transform -> i16);
    int16x8_t peak_abs_value = vabsq_s16(peak_value);
    int16x8_t pos = {0, 1, 2, 3, 4, 5, 6, 7};
    int16x8_t pos_constant = {0, 1, 2, 3, 4, 5, 6, 7};
    int16x8_t pos_add = {8, 8, 8, 8, 8, 8, 8, 8};
    int16x8_t peak_value_new, absolute;
    uint16x8_t mask;
    
    for(int32_t i = 8; i < 128; i = i + 8)
    {
        pos_constant = vaddq_s16(pos_constant, pos_add);
        peak_value_new = vld1q_s16(transform->i16 + i);
        absolute = vabsq_s16(peak_value_new);
        mask = vcgtq_s16(absolute, peak_abs_value);

        
        peak_abs_value = vbslq_s16(mask, absolute, peak_abs_value);
        peak_value = vbslq_s16(mask, peak_value_new, peak_value);
        pos = vbslq_s16(mask, pos_constant, pos);
        
    }
    int max = 0;
    for(int i = 0; i < 8; i++)
    {
        int mask = (max ^ i) & (-(int)(peak_abs_value[i] > peak_abs_value[max]));
        max = max ^ mask;
    }
    
    //set bit 7
    
    int32_t calculator = (int32_t)pos[max] | (int32_t)(128 * ((int32_t)peak_value[max] > 0));
    return calculator;

}



/**
 * @brief Encodes the received word
 *
 * The message consists of N1 bytes each byte is encoded into PARAM_N2 bits,
 * or MULTIPLICITY repeats of 128 bits
 *
 * @param[out] cdw Array of size VEC_N1N2_SIZE_64 receiving the encoded message
 * @param[in] msg Array of size VEC_N1_SIZE_64 storing the message
 */
void reed_muller_encode(uint64_t *cdw, const uint64_t *msg) {
    uint8_t *message_array = (uint8_t *) msg;
    codeword *codeArray = (codeword *) cdw;
    for (size_t i = 0; i < VEC_N1_SIZE_BYTES; i++) {
        // fill entries i * MULTIPLICITY to (i+1) * MULTIPLICITY
        int32_t pos = i * MULTIPLICITY;
        // encode first word
        encode(&codeArray[pos], message_array[i]);
        // copy to other identical codewords
        for (size_t copy = 1; copy < MULTIPLICITY; copy++) {
            memcpy(&codeArray[pos + copy], &codeArray[pos], sizeof(codeword));
        }
        
    }
    return;
}



/**
 * @brief Decodes the received word
 *
 * Decoding uses fast hadamard transform, for a more complete picture on Reed-Muller decoding, see MacWilliams, Florence Jessie, and Neil James Alexander Sloane.
 * The theory of error-correcting codes codes @cite macwilliams1977theory
 *
 * @param[out] msg Array of size VEC_N1_SIZE_64 receiving the decoded message
 * @param[in] cdw Array of size VEC_N1N2_SIZE_64 storing the received word
 */
void reed_muller_decode(uint64_t *msg, const uint64_t *cdw) {
    uint8_t *message_array = (uint8_t *) msg;
    codeword *codeArray = (codeword *) cdw;
    expandedCodeword expanded;
    for (size_t i = 0; i < VEC_N1_SIZE_BYTES; i++) {
        // collect the codewords
        
        expand_and_sum(&expanded, &codeArray[i * MULTIPLICITY]);
        
        // apply hadamard transform
        expandedCodeword transform;
        hadamard(&expanded, &transform);
        
        // fix the first entry to get the half Hadamard transform
        transform.i16[0] -= 64 * MULTIPLICITY;
        // finish the decoding
        
        message_array[i] = find_peaks(&transform);
        
    }
    
}
