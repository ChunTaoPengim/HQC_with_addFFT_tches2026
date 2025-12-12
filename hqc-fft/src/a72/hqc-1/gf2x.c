/**
 * \file gf2x.c
 * \brief AVX2 implementation of multiplication of two polynomials
 */

 #include "gf2x.h"
 #include "parameters.h"
 #include <stdint.h>
 #include <string.h>
 #include <stdio.h>
#include<arm_neon.h>
 
 
 #define VEC_N_ARRAY_SIZE_VEC CEIL_DIVIDE(PARAM_N_MULT, 64) /*!< The number of needed vectors to store PARAM_N bits*/
 #define WORD 64
 #define LAST64 (PARAM_N >> 6)
 
 #define T_3W 2048
 #define T_3W_64 (T_3W >> 6) // 32
 #define T2_3W_64 (2 * T_3W_64) // 64
 #define T2REC_3W_64 (6 * T_3W_64) // 198
 #define T_3W_256 (T_3W >> 8)
 #define T2_3W_256 (2 * T_3W_256)
 #define T2REC_3W_256 (6 * T_3W_256) 
 
 #define T_TM3R_3W (PARAM_N_MULT / 3) //6016
 #define T_TM3R (PARAM_N_MULT + 384) //18432
 #define T_TM3R_3W_256 ((T_TM3R_3W + 128) / (4 * WORD)) // 24
 #define T_TM3R_3W_64 (T_TM3R_3W_256 << 2) // 96
 
 
 uint64_t a1_times_a2[VEC_N_256_SIZE_64 << 1];
 uint64_t o256[VEC_N_ARRAY_SIZE_VEC << 2];
 
 uint64_t bloc64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight
 uint64_t bit64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight
 
 

#define N_PAD_U64  (17920/64)
#define N_MOD_64   (5)


 /**
  * @brief Compute o(x) = a(x) mod \f$ X^n - 1\f$
  *
  * This function computes the modular reduction of the polynomial a(x)
  *
  * @param[out] o Pointer to the result
  * @param[in] a Pointer to the polynomial a(x)
  */
void reduce(uint64_t *o, const uint64_t *a) {

    for(int i=0;i<VEC_N_SIZE_64;i++) {
        o[i] = a[i] ^ ((a[VEC_N_SIZE_64-1+i]>>N_MOD_64) | (a[VEC_N_SIZE_64+i]<<(64-N_MOD_64)));
    }
    o[VEC_N_SIZE_64-1] &= BITMASK(PARAM_N, 64);
}
  
static inline poly128_t vmull64_a72(const uint64_t a_64, const uint64_t b_64)
{
    const poly8x8_t a = vreinterpret_p8_u64(vdup_n_u64(a_64));

    const poly8x8_t b = vreinterpret_p8_u64(vdup_n_u64(b_64));

    poly8x8_t a_1 = vext_p8(a, a, 1);
    poly8x8_t a_2 = vext_p8(a, a, 2);
    poly8x8_t a_3 = vext_p8(a, a, 3);
    poly8x8_t b_1 = vext_p8(b, b, 1);
    poly8x8_t b_2 = vext_p8(b, b, 2);
    poly8x8_t b_3 = vext_p8(b, b, 3);
    poly8x8_t b_4 = vext_p8(b, b, 4);
    poly16x8_t d = vmull_p8(a, b);
    uint64_t k_48 = 0x0000FFFFFFFFFFFF;
    poly16x8_t L = vaddq_p16(vmull_p8(a, b_1), vmull_p8(a_1, b));
    poly16x4_t result_high = vget_high_p16(L);
    poly16x4_t result_low = vget_low_p16(L);
    result_low = vadd_p16(result_high, result_low);
    result_high = vreinterpret_p16_u16(vand_u16(vreinterpret_u16_p16(result_high), vreinterpret_u16_u64(vdup_n_u64(k_48))));
    result_low = vadd_p16(result_high, result_low);
    poly8x16_t temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 15);


    poly16x8_t result = vaddq_p16( d, vreinterpretq_p16_p8(temp_result));


    poly16x8_t M = vaddq_p16(vmull_p8(a, b_2), vmull_p8(a_2, b));
    uint64_t k_32 = 0x00000000FFFFFFFF;
    result_high = vget_high_p16(M);
    result_low = vget_low_p16(M);
    result_low = vadd_p16(result_high, result_low);
    result_high = vreinterpret_p16_u16(vand_u16(vreinterpret_u16_p16(result_high), vreinterpret_u16_u64(vdup_n_u64(k_32))));
    result_low = vadd_p16(result_high, result_low);
    temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 14);
    result = vaddq_p16(result, vreinterpretq_p16_p8(temp_result));

    poly16x8_t N = vaddq_p16(vmull_p8(a, b_3), vmull_p8(a_3, b));
    uint64_t k_16 = 0x000000000000FFFF;
    result_high = vget_high_p16(N);
    result_low = vget_low_p16(N);
    result_low = vadd_p16(result_high, result_low);
    result_high = vreinterpret_p16_u16(vand_u16(vreinterpret_u16_p16(result_high), vreinterpret_u16_u64(vdup_n_u64(k_16))));
    result_low = vadd_p16(result_high, result_low);
    temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 13);
    result = vaddq_p16(result, vreinterpretq_p16_p8(temp_result));

    poly16x8_t K = vmull_p8(a, b_4);
    result_high = vget_high_p16(K);
    result_low = vget_low_p16(K);
    result_low = vadd_p16(result_high, result_low);
    result_high = vdup_n_p16(0);
    temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 12);
    result = vaddq_p16(result, vreinterpretq_p16_p8(temp_result));
    return vreinterpretq_p128_p16(result);
}
 
 /**
  * @brief Compute C(x) = A(x)*B(x)
  * A(x) and B(x) are stored in 128-bit registers
  * This function computes A(x)*B(x) using Karatsuba
  *
  * @param[out] C Pointer to the result
  * @param[in] A Pointer to the polynomial A(x)
  * @param[in] B Pointer to the polynomial B(x)
  */
void schoolbook(uint64_t *c, const uint64_t * a, const uint64_t *b, int length)
{
    poly128_t temp;
    poly64x2_t result;
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < length; j++)
        {
            temp = vmull64_a72((poly64_t)a[i], (poly64_t)b[j]);
            result = vreinterpretq_p64_p128(temp);
            c[i+j] ^= (uint64_t)vgetq_lane_p64(result, 0);
            c[i+j+1] ^= (uint64_t)vgetq_lane_p64(result, 1);
        }
    }
    return;
}


void karat_mult_square( uint64_t *c, const uint64_t * a, const uint64_t *b, int length)
  {   
  
      if(length == 2)
      {
          poly128_t temp_f1 = vmull64_a72(a[1], b[1]);
          poly128_t temp_f2 = vmull64_a72(a[0], b[0]);
          poly128_t temp_f3 = vmull64_a72((a[0]^a[1]) , (b[0]^b[1]));
          temp_f3 ^= (temp_f1 ^ temp_f2);
          c[3] = vgetq_lane_p64(vreinterpretq_p64_p128(temp_f1), 1);
          c[2] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f1), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f3), 1) ) ;
          c[1] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f3), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f2), 1) ) ;
          c[0] = vgetq_lane_p64(vreinterpretq_p64_p128(temp_f2), 0);
      }
      else if(length == 4)
      {
          poly128_t temp_f1 = vmull64_a72(a[3], b[3]);
          poly128_t temp_f2 = vmull64_a72(a[2], b[2]);
          poly128_t temp_f3 = vmull64_a72((a[3]^a[2]) , (b[3]^b[2]));
          temp_f3 ^= (temp_f1 ^ temp_f2);
  
          poly128_t temp_f4 = vmull64_a72(a[1], b[1]);
          poly128_t temp_f5 = vmull64_a72(a[0], b[0]);
          poly128_t temp_f6 = vmull64_a72((a[1]^a[0]) , (b[0]^b[1]));
          temp_f6 ^= (temp_f4 ^ temp_f5);
  
          poly128_t temp_f7 = vmull64_a72((a[1]^a[3]), (b[1] ^ b[3])) ;
          poly128_t temp_f8 = vmull64_a72((a[0]^a[2]), (b[0] ^ b[2])) ;
          poly128_t temp_f9 = vmull64_a72((a[0]^a[1]^a[2]^a[3]), (b[0]^b[1]^b[2]^b[3]));
          temp_f9 ^= (temp_f7 ^ temp_f8 );
  
          temp_f7 ^= (temp_f1 ^ temp_f4);
          temp_f8 ^= (temp_f2 ^ temp_f5);
          temp_f9 ^= (temp_f3 ^ temp_f6);
          temp_f2 ^= temp_f7;
          temp_f4 ^= temp_f8;
          c[7] = vgetq_lane_p64(vreinterpretq_p64_p128(temp_f1), 1);
          c[6] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f1), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f3), 1) ) ;
          c[5] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f3), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f2), 1) ) ;
          c[4] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f2), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f9), 1) ) ;
          c[3] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f9), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f4), 1) ) ;
          c[2] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f4), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f6), 1) ) ;
          c[1] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f6), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f5), 1) ) ;
          c[0] = vgetq_lane_p64(vreinterpretq_p64_p128(temp_f5), 0);
      }
      else
      {
          int half = length >> 1; 
          karat_mult_square(c, a, b, half);
          karat_mult_square(c + length, a + half, b+ half, half);
          uint64_t temp[length];
          uint64_t tempa[half];
          uint64_t tempb[half];
          for(int i = 0; i < half; i++)
          {
              tempa[i] = (a[i] ^ a[i+half]);
              tempb[i] = (b[i] ^ b[i+half]);
          }
          karat_mult_square(temp, tempa, tempb, half);
          for(int i = 0; i < length; i++)
          {
              temp[i] ^= (c[i]^c[i+length]);
          }
          for(int i = half; i < half+length; i++)
          {
              c[i] ^= temp[i-half];
          }
      }
  }
 
 
 /**
  * @brief Compute C(x) = A(x)*B(x)
  *
  * This function computes A(x)*B(x) using Karatsuba 3 part split
  * A(x) and B(x) are stored in 256-bit registers
  * @param[out] C Pointer to the result
  * @param[in] A Pointer to the polynomial A(x)
  * @param[in] B Pointer to the polynomial B(x)
  */
void karat_mult3(uint64_t *Out, uint64_t *A, uint64_t *B) {
     uint64_t *a0, *b0, *a1, *b1, *a2, *b2;
     uint64_t aa01[T_3W_64], bb01[T_3W_64], aa02[T_3W_64], bb02[T_3W_64], aa12[T_3W_64], bb12[T_3W_64];
     uint64_t D0[T2_3W_64], D1[T2_3W_64], D2[T2_3W_64], D3[T2_3W_64], D4[T2_3W_64], D5[T2_3W_64];
     uint64_t ro256[3 * T2_3W_64];
 
     a0 = A;
     a1 = A + T_3W_64;
     a2 = A + (T_3W_64 << 1);
 
     b0 = B;
     b1 = B + T_3W_64;
     b2 = B + (T_3W_64 << 1);
 
     for(int32_t i = 0; i < T_3W_64; i++) {
         aa01[i] = a0[i] ^ a1[i];
         bb01[i] = b0[i] ^ b1[i];
 
         aa12[i] = a2[i] ^ a1[i];
         bb12[i] = b2[i] ^ b1[i];
 
         aa02[i] = a0[i] ^ a2[i];
         bb02[i] = b0[i] ^ b2[i];
     }
 
     karat_mult_square(D0, a0, b0, T_3W_64);
     karat_mult_square(D1, a1, b1, T_3W_64);
     karat_mult_square(D2, a2, b2, T_3W_64);
 
     karat_mult_square(D3, aa01, bb01, T_3W_64);
     karat_mult_square(D4, aa02, bb02, T_3W_64);
     karat_mult_square(D5, aa12, bb12, T_3W_64);
 
     for(int32_t i = 0; i < T_3W_64; i++) {
         int32_t j = i + T_3W_64;
         uint64_t middle0 = D0[i] ^ D1[i] ^ D0[j];
         ro256[i] = D0[i];
         ro256[j]  = D3[i] ^ middle0;
         ro256[j + T_3W_64] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle0;
         middle0 = D1[j] ^ D2[i] ^ D2[j];
         ro256[j + (T_3W_64 << 1)] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle0;
         ro256[i + (T_3W_64 << 2)] = D5[j] ^ middle0;
         ro256[j + (T_3W_64 << 2)] = D2[j];
     }
 
     for(int32_t i = 0; i < T2REC_3W_64; i++) {
         Out[i] = ro256[i];
     }
 }
 
 
 
 /**
  * @brief Compute B(x) = A(x)/(x+1) 
  *
  * This function computes A(x)/(x+1) using a Quercia like algorithm
  * @param[out] out Pointer to the result
  * @param[in] in Pointer to the polynomial A(x)
  * @param[in] size used to define the number of coeeficients of A
  */
 static inline void divide_by_x_plus_one_64(uint64_t *out, uint64_t *in, int32_t size){
     
     out[0] = in[0];
     for(int32_t i=1; i < 2 * (size << 2); i++) {
         out[i]= out[i - 1] ^ in[i];
     }
 }
 
 
 
 /**
  * @brief Compute C(x) = A(x)*B(x)
  *
  * This function computes A(x)*B(x) using Toom-Cook 3 part split
  * A(x) and B(x) are stored in 256-bit registers
  * @param[out] C Pointer to the result
  * @param[in] A Pointer to the polynomial A(x)
  * @param[in] B Pointer to the polynomial B(x)
  */
void toom_3_mult_arm(uint64_t * out, const uint64_t * A, const uint64_t * B)
{
    static uint64_t U0[T_TM3R_3W_64], V0[T_TM3R_3W_64], U1[T_TM3R_3W_64], V1[T_TM3R_3W_64], U2[T_TM3R_3W_64], V2[T_TM3R_3W_64];
    static uint64_t W0[2 * (T_TM3R_3W_64)], W1[2 * (T_TM3R_3W_64)], W2[2 * (T_TM3R_3W_64)], W3[2 * (T_TM3R_3W_64)], W4[2 * (T_TM3R_3W_64)];
    static uint64_t tmp[4 * (T_TM3R_3W_64)];
    // static uint64_t ro256[6 * (T_TM3R_3W_64)];
    const uint64_t zero = 0x0ul;

    int32_t T2 = T_TM3R_3W_64 << 1;
    for(int32_t i = 0; i < T_TM3R_3W_64 - 2; i++) {
        int32_t i4 = i;
        int32_t i42 = i4 - 2;
        U0[i]= A[i4];
        V0[i]= B[i4];
        U1[i]= A[i42 + T_TM3R_3W_64];
        V1[i]= B[i42 + T_TM3R_3W_64];
        U2[i]= A[i4 + T2 - 4];
        V2[i]= B[i4 + T2 - 4];
    }
    
    U0[T_TM3R_3W_64-2]= 0x0ul;
    U0[T_TM3R_3W_64-1]= 0x0ul;
    V0[T_TM3R_3W_64-2]= 0x0ul;
    V0[T_TM3R_3W_64-1]= 0x0ul;
    
    U1[T_TM3R_3W_64-2]= 0x0ul;
    U1[T_TM3R_3W_64-1]= 0x0ul;
    V1[T_TM3R_3W_64-2]= 0x0ul;
    V1[T_TM3R_3W_64-1]= 0x0ul;

    U2[T_TM3R_3W_64-2]= 0x0ul;
    U2[T_TM3R_3W_64-1]= 0x0ul;
    V2[T_TM3R_3W_64-2]= 0x0ul;
    V2[T_TM3R_3W_64-1]= 0x0ul;
    // EVALUATION PHASE : x= X^64
    // P(X): P0=(0); P1=(1); P2=(x); P3=(1+x); P4=(\infty)
    // Evaluation: 5*2 add, 2*2 shift; 5 mul (n)
    //W3 = U2 + U1 + U0 ; W2 = V2 + V1 + V0
    for(int32_t i = 0; i < T_TM3R_3W_64; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    
    //W1 = W2 * W3
    karat_mult3(W1, W2, W3);
    //W0 =(U1 + U2*x)*x ; W4 =(V1 + V2*x)*x (SIZE = T_TM3R_3W_64 !)
    W0[0] = 0x0ul;
    W0[1] = U1[0];
    W4[0] = 0x0ul;
    W4[1] = V1[0];
    for(int i = 0; i < T_TM3R_3W_64 -3 ; i++)
    {
        W0[i+2] = (U1[i+1] ^ U2[i]);
        W4[i+2] = (V1[i+1] ^ V2[i]);
    }
    W0[T_TM3R_3W_64 - 1] = U2[T_TM3R_3W_64-3];
    W4[T_TM3R_3W_64 - 1] = V2[T_TM3R_3W_64-3];
    //W3 = W3 + W0; W2 = W2 + W4
    for(int32_t i = 0; i < T_TM3R_3W_64; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    //W0 = W0 + U0; W4 = W4 + V0
    for(int32_t i = 0; i < T_TM3R_3W_64; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    

    //W3 = W3 * W2; W2 = W0 * W4
    karat_mult3(tmp, W3, W2);
    for(int32_t i = 0; i < 2 * (T_TM3R_3W_64); i++) {
        W3[i] = tmp[i];
    }
    
    karat_mult3(W2, W0, W4);

    //W4 = U2 * V2; W0 = U0 * V0
    karat_mult3(W4, U2, V2);
    karat_mult3(W0, U0, V0);

    // interpolation phase
     //W3 = W3 + W2
    for(int32_t i = 0; i < 2 * (T_TM3R_3W_64); i++) {
        W3[i] ^= W2[i];
    }
 
     //W1 = W1 + W0
    for(int32_t i = 0; i < 2 * (T_TM3R_3W_64); i++) {
        W1[i] ^= W0[i];
    }
 
    //W2 =(W2 + W0)/x -> x = X^64	
    
    for(int32_t i = 0; i < (T_TM3R_3W_64 << 1) - 1; i++) {
        W2[i] = W2[i+1] ^ W0[i+1];  
    }
 
    //W2 =(W2 + W3 + W4*(x^3+1))/(x+1)	
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1];
    tmp[2] = W2[2] ^ W3[2] ^ W4[2];

    for(int32_t i = 3; i < (T_TM3R_3W_64 << 1) - 1; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ W4[i-3];
    }
    
    divide_by_x_plus_one_64(W2, tmp, T_TM3R_3W_256);
    W2[2 * (T_TM3R_3W_64) - 1] = zero;
     
    //W3 =(W3 + W1)/(x*(x+1))
 
    for(int32_t i = 0; i < (T_TM3R_3W_64 << 1) - 1; i++) {
        tmp[i] = W3[i+1] ^ W1[i+1];
    }
     
    divide_by_x_plus_one_64(W3, tmp, T_TM3R_3W_256);
    W3[2 * (T_TM3R_3W_64) - 1] = zero;
 
    //W1 = W1 + W4 + W2
    for(int32_t i = 0; i < 2 * (T_TM3R_3W_64); i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
 
    //W2 = W2 + W3
    for(int32_t i = 0; i < 2 * (T_TM3R_3W_64); i++) {
        W2[i] ^= W3[i];
    }
 
    // Recomposition
    //W  = W0+ W1*x+ W2*x^2+ W3*x^3 + W4*x^4 
    //Attention : W0, W1, W4 of size 2*T_TM3R_3W_256, W2 and W3 of size 2*(T_TM3R_3W_256)
    for(int32_t i = 0; i < (T_TM3R_3W_64 << 1) - 4; i++) {
        out[i] = W0[i];
        out[i + 2 * T_TM3R_3W_64 - 4] = W2[i];
        out[i + 4 * T_TM3R_3W_64 - 8] = W4[i];
    }
    
    out[(T_TM3R_3W_64 << 1) - 4] ^= W0[(T_TM3R_3W_64 << 1) - 4];
    out[(T_TM3R_3W_64 << 1) - 3] ^= W0[(T_TM3R_3W_64 << 1) - 3];
    out[(T_TM3R_3W_64 << 1) - 2] ^= W0[(T_TM3R_3W_64 << 1) - 2];
    out[(T_TM3R_3W_64 << 1) - 1] ^= W0[(T_TM3R_3W_64 << 1) - 1];

    out[(T_TM3R_3W_64 << 2) - 8] ^= W2[(T_TM3R_3W_64 << 1) - 4];
    out[(T_TM3R_3W_64 << 2) - 7] ^= W2[(T_TM3R_3W_64 << 1) - 3];
    out[(T_TM3R_3W_64 << 2) - 6] ^= W2[(T_TM3R_3W_64 << 1) - 2];
    out[(T_TM3R_3W_64 << 2) - 5] ^= W2[(T_TM3R_3W_64 << 1) - 1];
    
    for(int i = 0; i < (T_TM3R_3W_64 << 1) ; i++)
    {
        out[T_TM3R_3W_64 - 2 + i] ^= W1[i];
        out[3*T_TM3R_3W_64 - 6 + i] ^= W3[i];
    }

}
 
 
 /**
  * @brief Multiply two polynomials modulo \f$ X^n - 1\f$.
  *
  * This functions multiplies a dense polynomial <b>a1</b> (of Hamming weight equal to <b>weight</b>)
  * and a dense polynomial <b>a2</b>. The multiplication is done modulo \f$ X^n - 1\f$.
  *
  * @param[out] o Pointer to the result
  * @param[in] a1 Pointer to a polynomial
  * @param[in] a2 Pointer to a polynomial
  */
 void vect_mul(uint64_t *o, const uint64_t *a1, const uint64_t *a2) {

    uint64_t temp_a1[282] = {0};
    uint64_t temp_a2[282] = {0};
    memcpy(temp_a1, a1, VEC_N_SIZE_64 * sizeof(uint64_t));
    memcpy(temp_a2, a2, VEC_N_SIZE_64 * sizeof(uint64_t));

    toom_3_mult_arm(a1_times_a2, temp_a1, temp_a2);
    reduce(o, a1_times_a2);
 
     // clear all
     #ifdef __STDC_LIB_EXT1__
         memset_s(a1_times_a2, 0, (VEC_N_256_SIZE_64 << 1) * sizeof(uint64_t));
     #else
         memset(a1_times_a2, 0, (VEC_N_256_SIZE_64 << 1) * sizeof(uint64_t));
     #endif
 }