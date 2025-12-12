#include "stdint.h"
#include "string.h"

#include "ringmul.h"

#include "btfy.h"
#include "gf256.h"
#include "bitpoly_to_gf256x2.h"
#include "bc_8.h"
#include "arm_neon.h"


const uint8x16_t idx = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
const uint16x8_t mask = {0, 0, 0, 0xFFFF, 0, 0, 0, 0};
const uint16x8_t final_mask1 = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x0000};
const uint16x8_t mask2 = {0, 0, 0, 0, 0, 0, 0xFFFF, 0xFFFF};
const uint16x8_t final_mask = {0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x0000};
const uint16x8_t mask3 = {0, 0, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF};
const uint16x8_t final_mask2 = {0x0000, 0xFFFF, 0xFFFF, 0xffff, 0xffff, 0xffff, 0x0000, 0x0000};

static inline void gf_16_vmull(uint8_t* c0, uint8_t * c1, const uint8_t* a_8, const uint8_t* b_8)
{
    const poly8x8_t a = vreinterpret_p8_u8(vld1_u8(a_8));

    const poly8x8_t b = vreinterpret_p8_u8(vld1_u8(b_8));

    poly8x8_t a_1 = vext_p8(a, a, 1);
    poly8x8_t a_2 = vext_p8(a, a, 2);
    poly8x8_t a_3 = vext_p8(a, a, 3);
    poly8x8_t b_1 = vext_p8(b, b, 1);
    poly8x8_t b_2 = vext_p8(b, b, 2);
    poly8x8_t b_3 = vext_p8(b, b, 3);
    poly8x8_t b_4 = vext_p8(b, b, 4);
    poly16x8_t d = vmull_p8(a, b); // d = A*B, with index = [0, 2, 4, 6, 8, 10, 12, 14]

    poly16x8_t L = vaddq_p16(vmull_p8(a, b_1), vmull_p8(a_1, b));//  L = E+F, with index = [7, 1, 3, 5, 7, 9, 11, 13]
    poly16x8_t add_term = vandq_u16(vdupq_n_u16(vgetq_lane_p16(L, 7)), vreinterpretq_p16_u16(mask));
    L = vaddq_p16(L, add_term);
    L = vandq_u16(L, vreinterpretq_p16_u16(final_mask1));


    poly16x8_t M = vaddq_p16(vmull_p8(a, b_2), vmull_p8(a_2, b)); // M = G+H, with index = [6, 8, 2, 4, 6, 8, 10, 12]
    poly16x8_t M_shifted = vextq_p16(M, M, 7); // 
    poly16x8_t add_vec = vandq_u16(M, vreinterpretq_p16_u16(mask2));
    add_vec = vextq_p16(add_vec, add_vec, 3);
    M = veorq_u16(M_shifted, add_vec);
    M = vandq_u16(M, vreinterpretq_p16_u16(final_mask));

    

    poly16x8_t N = vaddq_p16(vmull_p8(a, b_3), vmull_p8(a_3, b)); // N = I+J, with index = [5, 7, 9, 3, 5, 7, 9, 11] 
    poly16x8_t N_shifted = vextq_p16(N, N, 7);
    add_vec = vandq_u16(N, vreinterpretq_p16_u16(mask3));
    add_vec = vextq_p16(add_vec, add_vec, 3);
    N = veorq_u16(N_shifted, add_vec);
    N = vandq_u16(N, vreinterpretq_p16_u16(final_mask2));
    



    poly16x8_t K = vmull_p8(a, b_4); // K = L, with index = [4, 6, 8, 10, 4, 6, 8, 10]
    K = vcombine_p16(vadd_p16(vget_low_p16(K), vget_high_p16(K)), vdup_n_p16(0));
    K = vextq_p16(K, K, 6);

    d = veorq_u16(d, M);
    d = veorq_u16(d, K); // representing [d0, d2, d4, d6, d8, d10, d12, d14]
    
    N = veorq_u16(N, L); // representing [d1, d3, d5, d7, d9, d11, d13, x
    

    uint8x16_t d_u8 = vreinterpretq_u8_u16(d);
    uint8x16_t N_u8 = vreinterpretq_u8_u16(N);

    uint8x16_t even_bytes = vuzp1q_u8(d_u8, N_u8);
    uint8x16_t odd_bytes = vuzp2q_u8(d_u8, N_u8);
    even_bytes = vqtbl1q_u8(even_bytes, idx);
    odd_bytes = vqtbl1q_u8(odd_bytes, idx);

    vst1q_u8(c0, even_bytes);
    vst1q_u8(c1, odd_bytes);
}

static inline void gf_16_vmull_vec(uint8_t* c0, uint8_t * c1, const uint8x8_t a_8 , const uint8x8_t b_8)
{
    const poly8x8_t a = vreinterpret_p8_u8(a_8);

    const poly8x8_t b = vreinterpret_p8_u8(b_8);

    poly8x8_t a_1 = vext_p8(a, a, 1);
    poly8x8_t a_2 = vext_p8(a, a, 2);
    poly8x8_t a_3 = vext_p8(a, a, 3);
    poly8x8_t b_1 = vext_p8(b, b, 1);
    poly8x8_t b_2 = vext_p8(b, b, 2);
    poly8x8_t b_3 = vext_p8(b, b, 3);
    poly8x8_t b_4 = vext_p8(b, b, 4);
    poly16x8_t d = vmull_p8(a, b); // d = A*B, with index = [0, 2, 4, 6, 8, 10, 12, 14]

    poly16x8_t L = vaddq_p16(vmull_p8(a, b_1), vmull_p8(a_1, b));//  L = E+F, with index = [7, 1, 3, 5, 7, 9, 11, 13]
    poly16x8_t add_term = vandq_u16(vdupq_n_u16(vgetq_lane_p16(L, 7)), vreinterpretq_p16_u16(mask));
    L = vaddq_p16(L, add_term);
    L = vandq_u16(L, vreinterpretq_p16_u16(final_mask1));


    poly16x8_t M = vaddq_p16(vmull_p8(a, b_2), vmull_p8(a_2, b)); // M = G+H, with index = [6, 8, 2, 4, 6, 8, 10, 12]
    poly16x8_t M_shifted = vextq_p16(M, M, 7); // 
    poly16x8_t add_vec = vandq_u16(M, vreinterpretq_p16_u16(mask2));
    add_vec = vextq_p16(add_vec, add_vec, 3);
    M = veorq_u16(M_shifted, add_vec);
    M = vandq_u16(M, vreinterpretq_p16_u16(final_mask));

    

    poly16x8_t N = vaddq_p16(vmull_p8(a, b_3), vmull_p8(a_3, b)); // N = I+J, with index = [5, 7, 9, 3, 5, 7, 9, 11] 
    poly16x8_t N_shifted = vextq_p16(N, N, 7);
    add_vec = vandq_u16(N, vreinterpretq_p16_u16(mask3));
    add_vec = vextq_p16(add_vec, add_vec, 3);
    N = veorq_u16(N_shifted, add_vec);
    N = vandq_u16(N, vreinterpretq_p16_u16(final_mask2));
    



    poly16x8_t K = vmull_p8(a, b_4); // K = L, with index = [4, 6, 8, 10, 4, 6, 8, 10]
    K = vcombine_p16(vadd_p16(vget_low_p16(K), vget_high_p16(K)), vdup_n_p16(0));
    K = vextq_p16(K, K, 6);

    d = veorq_u16(d, M);
    d = veorq_u16(d, K); // representing [d0, d2, d4, d6, d8, d10, d12, d14]
    
    N = veorq_u16(N, L); // representing [d1, d3, d5, d7, d9, d11, d13, x
    

    uint8x16_t d_u8 = vreinterpretq_u8_u16(d);
    uint8x16_t N_u8 = vreinterpretq_u8_u16(N);

    uint8x16_t even_bytes = vuzp1q_u8(d_u8, N_u8);
    uint8x16_t odd_bytes = vuzp2q_u8(d_u8, N_u8);
    even_bytes = vqtbl1q_u8(even_bytes, idx);
    odd_bytes = vqtbl1q_u8(odd_bytes, idx);

    vst1q_u8(c0, even_bytes);
    vst1q_u8(c1, odd_bytes);
}
static inline void karat_mult_square(uint8_t* c0, uint8_t * c1, const uint8_t* a_8, const uint8_t* b_8, int length)
{
    if(length == 2)
    {
        gf_16_vmull(c0, c1, a_8, b_8);
        gf_16_vmull(c0 + 16, c1+16, a_8 + 8, b_8 + 8);

        uint8x16_t a1 = vld1q_u8(a_8);
        uint8x16_t b1 = vld1q_u8(b_8);
        uint8_t temp_c0[16];
        uint8_t temp_c1[16];
        gf_16_vmull_vec(temp_c0, temp_c1, veor_u8(vget_low_u8(a1), vget_high_u8(a1)), veor_u8(vget_low_u8(b1), vget_high_u8(b1)));
        uint8x16_t temp_c0_v = vld1q_u8(temp_c0);
        uint8x16_t temp_c1_v = vld1q_u8(temp_c1);
        a1 = vld1q_u8(c0);
        b1 = vld1q_u8(c1);
        uint8x16_t a2 = vld1q_u8(c0 + 16);
        uint8x16_t b2 = vld1q_u8(c1 + 16);
        temp_c0_v = veorq_u8(temp_c0_v, veorq_u8(a1, a2));
        temp_c1_v = veorq_u8(temp_c1_v, veorq_u8(b1, b2));
        a1 = vld1q_u8(c0+8);
        b1 = vld1q_u8(c1+8);
        temp_c0_v = veorq_u8(temp_c0_v, a1);
        temp_c1_v = veorq_u8(temp_c1_v, b1);
        vst1q_u8(c0+8, temp_c0_v);
        vst1q_u8(c1+8, temp_c1_v);
    }
    else
    {
        int half = length >> 1;
        karat_mult_square(c0, c1, a_8, b_8, half);
        karat_mult_square(c0 + (length << 3), c1 + (length << 3), a_8 + (half << 3), b_8 + (half << 3), half);
        uint8_t temp_c0[(length) << 3];
        uint8_t temp_c1[(length) << 3];
        uint8_t tempa[half << 3];
        uint8_t tempb[half << 3];
        uint8x16_t a1;
        uint8x16_t a2;
        uint8x16_t b1;
        uint8x16_t b2;
        for(int i = 0; i < half; i = i + 2)
        {
            a1 = vld1q_u8(a_8 + (i << 3));
            b1 = vld1q_u8(b_8 + (i << 3));
            a2 = vld1q_u8(a_8 + ((i + half) << 3));
            b2 = vld1q_u8(b_8 + ((i + half) << 3));
            vst1q_u8(tempa + (i << 3), veorq_u8(a1, a2));
            vst1q_u8(tempb + (i << 3), veorq_u8(b1, b2));
        }
        karat_mult_square(temp_c0, temp_c1, tempa, tempb, half);
        for(int i = 0; i < length; i += 2)
        {
            a1 = vld1q_u8(c0 + (i << 3));
            a2 = vld1q_u8(c0 + ((i+length) << 3));
            b1 = vld1q_u8(temp_c0 + (i << 3));
            b1 = veorq_u8(b1, veorq_u8(a1, a2));
            vst1q_u8(temp_c0 + (i << 3), b1);

            a1 = vld1q_u8(c1 + (i << 3));
            a2 = vld1q_u8(c1 + ((i+length) << 3));
            b1 = vld1q_u8(temp_c1 + (i << 3));
            b1 = veorq_u8(b1, veorq_u8(a1, a2));
            vst1q_u8(temp_c1 + (i << 3), b1);
        }
        for(int i = half; i < half+length; i = i +2)
        {
            a1 = vld1q_u8(c0 + (i << 3));
            a2 = vld1q_u8(temp_c0 + ((i-half) << 3));
            vst1q_u8(c0 + (i << 3), veorq_u8(a1, a2));

            a1 = vld1q_u8(c1 + (i << 3));
            a2 = vld1q_u8(temp_c1 + ((i-half) << 3));
            vst1q_u8(c1 + (i << 3), veorq_u8(a1, a2));
        }
    }
}
void ringmul_mul_384(  uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b )
{
    const uint8_t *a0, *b0, *a1, *b1, *a2, *b2;
    uint8_t aa01[128], bb01[128], aa02[128], bb02[128];
    // , aa12[128], bb12[128];
    uint8_t D0_low[256], D1_low[256], D2_low[256], D3_low[256], D4_low[256];
    uint8_t ro256_low[3 * 256];

    uint8_t D0_high[256], D1_high[256], D2_high[256], D3_high[256], D4_high[256];
    uint8_t ro256_high[3 * 256];

    a0 = a;
    a1 = a + 128;
    a2 = a + 256;

    b0 = b;
    b1 = b + 128;
    b2 = b + 256;

    uint8x16_t temp1;
    uint8x16_t temp2;
    uint8x16_t temp3;
    uint8x16_t temp4;
    uint8x16_t temp5;
    uint8x16_t temp6;

    for(int32_t i = 0; i < 128; i = i + 16) {

        temp1 = vld1q_u8(a0 + i);
        temp2 = vld1q_u8(a1 + i);
        vst1q_u8(aa01 + i, veorq_u8(temp1, temp2));
        // aa01[i] = a0[i] ^ a1[i];

        temp3 = vld1q_u8(b0 + i);
        temp4 = vld1q_u8(b1 + i);
        vst1q_u8(bb01 + i, veorq_u8(temp3, temp4));
        // bb01[i] = b0[i] ^ b1[i];

        temp5 = vld1q_u8(a2 + i);
        // vst1q_u8(aa12 + i, veorq_u8(temp5, temp2));
        // aa12[i] = a2[i] ^ a1[i];
        temp6 = vld1q_u8(b2 + i);
        // vst1q_u8(bb12 + i, veorq_u8(temp4, temp6));
        // bb12[i] = b2[i] ^ b1[i];

        vst1q_u8(aa02 + i, veorq_u8(temp1, temp5));
        // aa02[i] = a0[i] ^ a2[i];
        vst1q_u8(bb02 + i, veorq_u8(temp3, temp6));
        // bb02[i] = b0[i] ^ b2[i];
    }

    karat_mult_square(D0_low, D0_high, a0, b0, 16);
    karat_mult_square(D1_low, D1_high, a1, b1, 16);
    karat_mult_square(D2_low, D2_high, a2, b2, 16);

    karat_mult_square(D3_low, D3_high, aa01, bb01, 16);
    karat_mult_square(D4_low, D4_high, aa02, bb02, 16);
    // karat_mult_square(D5, aa12, bb12, 16);

    for(int32_t i = 0; i < 128; i = i + 16) {
        int32_t j = i + 128;
        temp1 = vld1q_u8(D0_low + i);
        temp2 = vld1q_u8(D0_low + j);
        temp3 = vld1q_u8(D1_low + i);
        uint8x16_t middle0 = veorq_u8(temp1, veorq_u8(temp2, temp3));
        // uint64_t middle0 = D0[i] ^ D1[i] ^ D0[j];
        // ro256[i] = D0[i];
        vst1q_u8(ro256_low + i, temp1);
        temp4 = vld1q_u8(D3_low + i);
        vst1q_u8(ro256_low + j, veorq_u8(temp4, middle0));
        // ro256[j]  = D3[i] ^ middle0;
        temp1 = vld1q_u8(D1_low + j);
        temp2 = vld1q_u8(D2_low + i);
        temp3 = vld1q_u8(D3_low + j);
        temp4 = vld1q_u8(D4_low + i);
        temp1 = veorq_u8(temp1, veorq_u8(temp2, temp3));
        temp4 = veorq_u8(temp4, middle0);
        vst1q_u8(ro256_low + j + 128, veorq_u8(temp4, temp1));

        temp1 = vld1q_u8(D0_high + i);
        temp2 = vld1q_u8(D0_high + j);
        temp3 = vld1q_u8(D1_high + i);
        middle0 = veorq_u8(temp1, veorq_u8(temp2, temp3));
        vst1q_u8(ro256_high + i, temp1);
        temp4 = vld1q_u8(D3_high + i);
        vst1q_u8(ro256_high + j, veorq_u8(temp4, middle0));
        temp1 = vld1q_u8(D1_high + j);
        temp2 = vld1q_u8(D2_high + i);
        temp3 = vld1q_u8(D3_high + j);
        temp4 = vld1q_u8(D4_high + i);
        temp1 = veorq_u8(temp1, veorq_u8(temp2, temp3));
        temp4 = veorq_u8(temp4, middle0);
        vst1q_u8(ro256_high + j + 128, veorq_u8(temp4, temp1));

    }

    memcpy(c0, ro256_low, 384 * sizeof(uint8_t));
    memcpy(c1, ro256_high, 384 * sizeof(uint8_t));
}
void ringmul_mul_1024( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b )
{
    const uint8_t *a0, *b0, *a1, *b1;
    
    uint8_t D0_low[1024], D1_low[1024], D2_low[1024];

    uint8_t D0_high[1024], D1_high[1024], D2_high[1024];

    a0 = a;
    a1 = a + 512;

    b0 = b;
    b1 = b + 512;

    karat_mult_square(D0_low, D0_high, a0, b0, 64);
    karat_mult_square(D1_low, D1_high, a1, b0, 64);
    karat_mult_square(D2_low, D2_high, a0, b1, 64);

    uint8x16_t temp1;
    uint8x16_t temp2;
    uint8x16_t temp3;
    for(int i = 0; i < 512; i+=16)
    {
        temp1 = vld1q_u8(D0_low + i + 512);
        temp2 = vld1q_u8(D1_low + i);
        temp3 = vld1q_u8(D2_low + i);
        temp1 = veorq_u8(temp1, veorq_u8(temp2, temp3));
        vst1q_u8(D0_low + i + 512, temp1);

        temp1 = vld1q_u8(D0_high + i + 512);
        temp2 = vld1q_u8(D1_high + i);
        temp3 = vld1q_u8(D2_high + i);
        temp1 = veorq_u8(temp1, veorq_u8(temp2, temp3));
        vst1q_u8(D0_high + i + 512, temp1);
    }
    memcpy(c0 , D0_low, 1024 * sizeof(uint8_t));
    memcpy(c1 , D0_high, 1024 * sizeof(uint8_t));
}