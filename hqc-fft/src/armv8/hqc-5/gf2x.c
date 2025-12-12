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
 
 
#define VEC_N_ARRAY_SIZE_VEC CEIL_DIVIDE(PARAM_N_MULT, 256) /*!< The number of needed vectors to store PARAM_N bits*/
#define WORD 64
#define LAST64 (PARAM_N >> 6)

#define T_5W 4096
#define T_5W_256 (T_5W >> 8)
#define T2_5W_256 (2 * T_5W_256)
#define T_5W_64 (T_5W >> 6) // 64
#define T2_5W_64 (2 * T_5W_64) // 128
#define t5 (5 * T_5W / WORD)

#define T_TM3R_3W (PARAM_N_MULT / 3) // 19968
#define T_TM3R (PARAM_N_MULT + 384) 
#define tTM3R ((T_TM3R) / WORD) // 
#define T_TM3R_3W_256 ((T_TM3R_3W + 128) / (4 * WORD))
#define T_TM3R_3W_64 (T_TM3R_3W_256 << 2) // 

 uint64_t a1_times_a2[VEC_N_256_SIZE_64 << 1];
 uint64_t o256[VEC_N_ARRAY_SIZE_VEC << 2];
 
 uint64_t bloc64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight
 uint64_t bit64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight
 
//  static inline void reduce(__m256i *o, const __m256i *a);
//  static inline void karat_mult_1(__m128i *C, __m128i *A, __m128i *B);
//  static inline void karat_mult_2(__m256i *C, __m256i *A, __m256i *B);
//  static inline void karat_mult_4(__m256i *C, __m256i *A, __m256i *B);
//  static inline void karat_mult_8(__m256i *C, __m256i *A, __m256i *B);
//  static inline void karat_mult3(__m256i *C, __m256i *A, __m256i *B);
//  static inline void divide_by_x_plus_one_256(__m256i *out, __m256i *in, int32_t size);
//  static inline void toom_3_mult(__m256i *C, const __m256i *A, const __m256i *B);
 
 
 /**
  * @brief Compute o(x) = a(x) mod \f$ X^n - 1\f$
  *
  * This function computes the modular reduction of the polynomial a(x)
  *
  * @param[out] o Pointer to the result
  * @param[in] a Pointer to the polynomial a(x)
  */
void reduce(uint64_t *o, const uint64_t *a) {

    uint64_t r, carry;
    static const int32_t dec64 = PARAM_N & 0x3f;
    static int32_t d0;
    int32_t i, i2;

    d0 = WORD - dec64;
    for (i = LAST64 ; i < (PARAM_N >> 5) - 4; i += 1) {
       r =  a[i] >> dec64;
       carry = a[i+1] << d0;  
   
       r ^= carry;
       i2 = (i - LAST64);
       o256[i2] = a[i2] ^ r;
   }
   i = i - LAST64;

   for (; i < LAST64 + 1 ; i++) {
       r = a[i + LAST64] >> dec64;
       carry = a[i + LAST64 + 1] << d0;
       r ^= carry;
       o256[i] = a[i] ^ r;
   }

   o256[LAST64] &= BITMASK(PARAM_N, 64);
   memcpy(o, o256, VEC_N_SIZE_BYTES);
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
            temp = vmull_p64((poly64_t)a[i], (poly64_t)b[j]);
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
          poly128_t temp_f1 = vmull_p64(a[1], b[1]);
          poly128_t temp_f2 = vmull_p64(a[0], b[0]);
          poly128_t temp_f3 = vmull_p64((a[0]^a[1]) , (b[0]^b[1]));
          temp_f3 ^= (temp_f1 ^ temp_f2);
          c[3] = vgetq_lane_p64(vreinterpretq_p64_p128(temp_f1), 1);
          c[2] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f1), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f3), 1) ) ;
          c[1] = (vgetq_lane_p64(vreinterpretq_p64_p128(temp_f3), 0)^ vgetq_lane_p64(vreinterpretq_p64_p128(temp_f2), 1) ) ;
          c[0] = vgetq_lane_p64(vreinterpretq_p64_p128(temp_f2), 0);
      }
      else if(length == 4)
      {
          poly128_t temp_f1 = vmull_p64(a[3], b[3]);
          poly128_t temp_f2 = vmull_p64(a[2], b[2]);
          poly128_t temp_f3 = vmull_p64((a[3]^a[2]) , (b[3]^b[2]));
          temp_f3 ^= (temp_f1 ^ temp_f2);
  
          poly128_t temp_f4 = vmull_p64(a[1], b[1]);
          poly128_t temp_f5 = vmull_p64(a[0], b[0]);
          poly128_t temp_f6 = vmull_p64((a[1]^a[0]) , (b[0]^b[1]));
          temp_f6 ^= (temp_f4 ^ temp_f5);
  
          poly128_t temp_f7 = vmull_p64((a[1]^a[3]), (b[1] ^ b[3])) ;
          poly128_t temp_f8 = vmull_p64((a[0]^a[2]), (b[0] ^ b[2])) ;
          poly128_t temp_f9 = vmull_p64((a[0]^a[1]^a[2]^a[3]), (b[0]^b[1]^b[2]^b[3]));
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
void karat_mult_5(uint64_t *Out, uint64_t *A, uint64_t *B) {
     uint64_t *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;
     static uint64_t aa01[T_5W_64], bb01[T_5W_64], aa02[T_5W_64], bb02[T_5W_64], aa03[T_5W_64], bb03[T_5W_64], aa04[T_5W_64], bb04[T_5W_64], aa12[T_5W_64], bb12[T_5W_64], aa13[T_5W_64], bb13[T_5W_64], aa14[T_5W_64], bb14[T_5W_64], aa23[T_5W_64], bb23[T_5W_64], aa24[T_5W_64], bb24[T_5W_64], aa34[T_5W_64], bb34[T_5W_64];

     static uint64_t D0[T2_5W_64], D1[T2_5W_64], D2[T2_5W_64], D3[T2_5W_64], D4[T2_5W_64], D01[T2_5W_64], D02[T2_5W_64], D03[T2_5W_64], D04[T2_5W_64], D12[T2_5W_64], D13[T2_5W_64], D14[T2_5W_64], D23[T2_5W_64], D24[T2_5W_64], D34[T2_5W_64];

     uint64_t ro256[t5 << 1];
 
     a0 = A;
     a1 = a0 + T_5W_64;
     a2 = a1 + T_5W_64;
     a3 = a2 + T_5W_64;
     a4 = a3 + T_5W_64;
     b0 = B;
     b1 = b0 + T_5W_64;
     b2 = b1 + T_5W_64;
     b3 = b2 + T_5W_64;
     b4 = b3 + T_5W_64;
 
     for (int32_t i = 0 ; i < T_5W_64 ; i++) {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];

        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];

        aa03[i] = a0[i] ^ a3[i];
        bb03[i] = b0[i] ^ b3[i];

        aa04[i] = a0[i] ^ a4[i];
        bb04[i] = b0[i] ^ b4[i];

        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];

        aa13[i] = a3[i] ^ a1[i];
        bb13[i] = b3[i] ^ b1[i];

        aa14[i] = a4[i] ^ a1[i];
        bb14[i] = b4[i] ^ b1[i];

        aa23[i] = a2[i] ^ a3[i];
        bb23[i] = b2[i] ^ b3[i];

        aa24[i] = a2[i] ^ a4[i];
        bb24[i] = b2[i] ^ b4[i];

        aa34[i] = a3[i] ^ a4[i];
        bb34[i] = b3[i] ^ b4[i];
    }
    
    karat_mult_square(D0, a0, b0, T_5W_64);
    karat_mult_square(D1, a1, b1, T_5W_64);
    karat_mult_square(D2, a2, b2, T_5W_64);
    karat_mult_square(D3, a3, b3, T_5W_64);
    karat_mult_square(D4, a4, b4, T_5W_64);

    karat_mult_square(D01, aa01, bb01, T_5W_64);
    karat_mult_square(D02, aa02, bb02, T_5W_64);
    karat_mult_square(D03, aa03, bb03, T_5W_64);
    karat_mult_square(D04, aa04, bb04, T_5W_64);

    karat_mult_square(D12, aa12, bb12, T_5W_64);
    karat_mult_square(D13, aa13, bb13, T_5W_64);
    karat_mult_square(D14, aa14, bb14, T_5W_64);

    karat_mult_square(D23, aa23, bb23, T_5W_64);
    karat_mult_square(D24, aa24, bb24, T_5W_64);

    karat_mult_square(D34, aa34, bb34, T_5W_64);
     
 
    for (int32_t i = 0 ; i < T_5W_64 ; i++) {
        ro256[i] = D0[i];
        ro256[i + T_5W_64] = D0[i + T_5W_64] ^ D01[i] ^ D0[i] ^ D1[i];
        ro256[i + 2 * T_5W_64] = D1[i] ^ D02[i] ^ D0[i] ^ D2[i] ^ D01[i + T_5W_64] ^ D0[i + T_5W_64] ^ D1[i + T_5W_64];
        ro256[i + 3 * T_5W_64] = D1[i + T_5W_64] ^ D03[i] ^ D0[i] ^ D3[i] ^ D12[i] ^ D1[i] ^ D2[i] ^ D02[i + T_5W_64] ^ D0[i + T_5W_64] ^ D2[i + T_5W_64];
        ro256[i + 4 * T_5W_64] = D2[i] ^ D04[i] ^ D0[i] ^ D4[i] ^ D13[i] ^ D1[i] ^ D3[i] ^ D03[i + T_5W_64] ^ D0[i + T_5W_64] ^ D3[i + T_5W_64] ^ D12[i + T_5W_64] ^ D1[i + T_5W_64] ^ D2[i + T_5W_64];
        ro256[i + 5 * T_5W_64] = D2[i + T_5W_64] ^ D14[i] ^ D1[i] ^ D4[i] ^ D23[i] ^ D2[i] ^ D3[i] ^ D04[i + T_5W_64] ^ D0[i + T_5W_64] ^ D4[i + T_5W_64] ^ D13[i + T_5W_64] ^ D1[i + T_5W_64] ^ D3[i + T_5W_64];
        ro256[i + 6 * T_5W_64] = D3[i] ^ D24[i] ^ D2[i] ^ D4[i] ^ D14[i + T_5W_64] ^ D1[i + T_5W_64] ^ D4[i + T_5W_64] ^ D23[i + T_5W_64] ^ D2[i + T_5W_64] ^ D3[i + T_5W_64];
        ro256[i + 7 * T_5W_64] = D3[i + T_5W_64] ^ D34[i] ^ D3[i] ^ D4[i] ^ D24[i + T_5W_64] ^ D2[i + T_5W_64] ^ D4[i + T_5W_64];
        ro256[i + 8 * T_5W_64] = D4[i] ^ D34[i + T_5W_64] ^ D3[i + T_5W_64] ^ D4[i + T_5W_64];
        ro256[i + 9 * T_5W_64] = D4[i + T_5W_64];
    }

    for(int32_t i = 0 ; i < T_5W_64 * 10 ; i++) {
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
static inline void divide_by_x_plus_one_256(uint64_t *in, uint64_t *out, int32_t size) {
    out[0] = in[0];	
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];
    for(int32_t i = 4 ; i < 2 * (size + 8) ; i++) {
        out[i] = out[i - 4] ^ in[i];
    }
}



/**
* @brief Compute C(x) = A(x)*B(x) using TOOM3Mult with recursive call 
*
* This function computes A(x)*B(x) using recursive TOOM-COOK3 Multiplication
* @param[out] Out Pointer to the result
* @param[in] A Pointer to the polynomial A(x)
* @param[in] B Pointer to the polynomial B(x)
*/
void toom_3_mult_arm( uint64_t *out, const uint64_t *A, const uint64_t *B) {
    static uint64_t U0[T_TM3R_3W_64 + 8], V0[T_TM3R_3W_64 + 8], U1[T_TM3R_3W_64 + 8], V1[T_TM3R_3W_64 + 8], U2[T_TM3R_3W_64 + 8], V2[T_TM3R_3W_64 + 8];
    static uint64_t W0[2 * (T_TM3R_3W_64 + 8)], W1[2 * (T_TM3R_3W_64 + 8)], W2[2 * (T_TM3R_3W_64 + 8)], W3[2 * (T_TM3R_3W_64 + 8)], W4[2 * (T_TM3R_3W_64 + 8)];
    static uint64_t tmp[2 * (T_TM3R_3W_64 + 8) + 12];
    static const uint64_t zero = 0x0ul;
    int32_t T2 = T_TM3R_3W_64 << 1;

    for (int32_t i = 0 ; i < T_TM3R_3W_64 ; i++) {
        U0[i]= A[i];
        V0[i]= B[i];
        U1[i]= A[i + T_TM3R_3W_64];
        V1[i]= B[i + T_TM3R_3W_64];
        U2[i]= A[i + T2];
        V2[i]= B[i + T2];
    }

    for (int32_t i = T_TM3R_3W_64 ; i < T_TM3R_3W_64 + 8 ; i++)	{
        U0[i] = zero;
        V0[i] = zero;
        U1[i] = zero;
        V1[i] = zero;
        U2[i] = zero;
        V2[i] = zero;
    }

    // EVALUATION PHASE : x= X^256
    // P(X): P0=(0); P1=(1); P2=(x); P3=(1+x); P4=(\infty)
    // Evaluation: 5*2 add, 2*2 shift; 5 mul (n)
    //W3 = U2 + U1 + U0 ; W2 = V2 + V1 + V0

    for (int32_t i = 0 ; i < T_TM3R_3W_64 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    for (int32_t i = T_TM3R_3W_64 ; i < T_TM3R_3W_64 + 8 ; i++) {
        W2[i] = zero;
        W3[i] = zero;
    }

    //W1 = W2 * W3
    karat_mult_5(W1, W2, W3);

    //W0 =(U1 + U2*x)*x ; W4 =(V1 + V2*x)*x (SIZE = T_TM3_3W_64 + 8 !)
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = zero;
    W4[1] = zero;
    W0[2] = zero;
    W4[2] = zero;
    W0[3] = zero;
    W4[3] = zero;

    W0[4] = U1[0];
    W4[4] = V1[0];
    W0[5] = U1[1];
    W4[5] = V1[1];
    W0[6] = U1[2];
    W4[6] = V1[2];
    W0[7] = U1[3];
    W4[7] = V1[3];

    for (int32_t i = 4 ; i < T_TM3R_3W_64 + 4 ; i++) {
        W0[i + 4] = U1[i] ^ U2[i - 4];
        W4[i + 4] = V1[i] ^ V2[i - 4];
    }

    W0[T_TM3R_3W_64 + 4] = U2[T_TM3R_3W_64 - 4];
    W0[T_TM3R_3W_64 + 5] = U2[T_TM3R_3W_64 - 3];
    W0[T_TM3R_3W_64 + 6] = U2[T_TM3R_3W_64 - 2];
    W0[T_TM3R_3W_64 + 7] = U2[T_TM3R_3W_64 - 1];

    W4[T_TM3R_3W_64 + 4] = V2[T_TM3R_3W_64 - 4];
    W4[T_TM3R_3W_64 + 5] = V2[T_TM3R_3W_64 - 3];
    W4[T_TM3R_3W_64 + 6] = V2[T_TM3R_3W_64 - 2];
    W4[T_TM3R_3W_64 + 7] = V2[T_TM3R_3W_64 - 1];

    //W3 = W3 + W0      ; W2 = W2 + W4
    for (int32_t i = 0 ; i < T_TM3R_3W_64 + 8 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }

    //W0 = W0 + U0      ; W4 = W4 + V0
    for (int32_t i = 0 ; i < T_TM3R_3W_64 + 8 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    //W3 = W3 * W2      ; W2 = W0 * W4
    karat_mult_5(tmp, W3, W2);
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64 + 8) ; i++) {
        W3[i] = tmp[i];
    }

    karat_mult_5(W2, W0, W4);

    //W4 = U2 * V2      ; W0 = U0 * V0
    karat_mult_5(W4, U2, V2);
    karat_mult_5(W0, U0, V0);

    //INTERPOLATION PHASE
    //9 add, 1 shift, 1 Smul, 2 Sdiv (2n)
    //W3 = W3 + W2
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64 + 8) ; i++) {
        W3[i] ^= W2[i];
    }

    //W1 = W1 + W0
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64) ; i++) {
        W1[i] ^= W0[i];
    }

    //W2 =(W2 + W0)/x
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64 + 8) - 4 ; i++) {
        int32_t i1 = i + 4;
        W2[i] = W2[i1] ^ W0[i1];
    }

    W2[2 * (T_TM3R_3W_64 + 8) - 4] = zero;
    W2[2 * (T_TM3R_3W_64 + 8) - 3] = zero;
    W2[2 * (T_TM3R_3W_64 + 8) - 2] = zero;
    W2[2 * (T_TM3R_3W_64 + 8) - 1] = zero;

    //W2 =(W2 + W3 + W4*(x^3+1))/(x+1)
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64 + 8) ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }

    tmp[2 * (T_TM3R_3W_64 + 8)] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 1] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 2] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 3] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 4] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 5] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 6] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 7] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 8] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 9] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 10] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) + 11] = zero;

    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64) ; i++) {
        tmp[i + 12] ^= W4[i];
    }

    divide_by_x_plus_one_256(tmp, W2, T_TM3R_3W_64);

    //W3 =(W3 + W1)/(x*(x+1))
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64 + 8) - 4 ; i++) {
        int32_t i1 = i + 4;
        tmp[i] = W3[i1] ^ W1[i1];
    }

    tmp[2 * (T_TM3R_3W_64 + 8) - 4] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) - 3] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) - 2] = zero;
    tmp[2 * (T_TM3R_3W_64 + 8) - 1] = zero;
    divide_by_x_plus_one_256(tmp, W3, T_TM3R_3W_64);

    //W1 = W1 + W4 + W2
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64 + 8) ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    //W2 = W2 + W3
    for (int32_t i = 0 ; i < 2 * (T_TM3R_3W_64 + 8) ; i++) {
        W2[i] ^= W3[i];
    }

    //Recomposition
    //W  = W0+ W1*x+ W2*x^2+ W3*x^3 + W4*x^4
    
    for(int32_t i = 0; i < 2 * (T_TM3R_3W_64 + 8) - 16 ; i++) {
        out[i] = W0[i];
        out[i + 2 * T_TM3R_3W_64] = W2[i];
        out[i + 4 * T_TM3R_3W_64] = W4[i];
    }
    for(int i = 2 * (T_TM3R_3W_64 + 8) - 16; i < 2 * (T_TM3R_3W_64 + 8); i++)
    {
        out[i] ^= W0[i];
        out[i + 2 * T_TM3R_3W_64] ^= W2[i];
    }
    for(int i = 0; i < 2 * (T_TM3R_3W_64 + 8); i++)
    {
        out[i + T_TM3R_3W_64] ^= W1[i];
        out[i + 3 * T_TM3R_3W_64] ^= W3[i];
    }
}

void vect_mul(uint64_t *o, const uint64_t *a1, const uint64_t *a2) {
    
    uint64_t temp_a1[936] = {0};
    uint64_t temp_a2[936] = {0};
    // copy a1 and a2 to temp arrays
    memcpy(temp_a1, a1, VEC_N_SIZE_BYTES);
    memcpy(temp_a2, a2, VEC_N_SIZE_BYTES);

    toom_3_mult_arm(a1_times_a2, temp_a1, temp_a2);
    reduce(o, a1_times_a2);

    // clear all
    #ifdef __STDC_LIB_EXT1__
        memset_s(a1_times_a2, 0, (VEC_N_SIZE_64 << 1) * sizeof(uint64_t));
    #else
        memset(a1_times_a2, 0, (VEC_N_SIZE_64 << 1) * sizeof(uint64_t));
    #endif
}
