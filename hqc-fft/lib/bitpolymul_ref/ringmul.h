
#ifndef _RINGMUL_H_
#define _RINGMUL_H_


#include <stdint.h>


#ifdef  __cplusplus
extern  "C" {
#endif

/// @brief transform an input polynomial in F216[x]/s12 into its evaluated values in F256x2
///        s12 = x^4096 + x^256 + x^16 + x
/// @param fft_a0 [out] evaluated values of the input polynomial in low  byte of F256x2.  size: 32768 bits = 4096 byte
/// @param fft_a1 [out] evaluated values of the input polynomial in high byte of F256x2.  size: 32768 bits = 4096 byte
/// @param poly_a [in] F2 polynomial of 17920 bits = (2048+192) bytes
void ringmul_s12_input_2240( uint8_t * fft_a0 , uint8_t * fft_a1 , const uint8_t * poly_a );

/// @brief transform an input polynomial in F216[x]/s13 into its evaluated values in F256x2
///        s13 = x^8192 + x^4096 + x^512 + x^256 + x^32 + x^16 + x^2 + x
/// @param fft_a0 [out] evaluated values of the input polynomial in low  byte of F256x2.  size: 65536 bits = 8192 byte
/// @param fft_a1 [out] evaluated values of the input polynomial in high byte of F256x2.  size: 65536 bits = 8192 byte
/// @param poly_a [in] F2 polynomial of 35864 bits = (4096+512) bytes
void ringmul_s13_input_4608( uint8_t * fft_a0 , uint8_t * fft_a1 , const uint8_t * poly_a );

///
/// @brief c in F216[x]/s12 = a_fft * b_fft
///
/// @param c0 [out] size : 32768 bits = 4096 byte
/// @param c1 [out] size : 32768 bits = 4096 byte
/// @param a0_fft [in] size : 3278 bits = 4096 byte
/// @param a1_fft [in] size : 3278 bits = 4096 byte
/// @param b0_fft [in] size : 32768 bits = 4096 byte
/// @param b1_fft [in] size : 32768 bits = 4096 byte
void ringmul_s12_mul( uint8_t * c0 , uint8_t * c1 , const uint8_t * a0_fft , const uint8_t * a1_fft , const uint8_t * b0_fft , const uint8_t * b1_fft );

///
/// @brief c in F216[x]/s13 = a_fft * b_fft
///
/// @param c0 [out] size : 65536 bits = 8192 byte
/// @param c1 [out] size : 65536 bits = 8192 byte
/// @param a0_fft [in] size : 65536 bits = 8192 byte
/// @param a1_fft [in] size : 65536 bits = 8192 byte
/// @param b0_fft [in] size : 65536 bits = 8192 byte
/// @param b1_fft [in] size : 65536 bits = 8192 byte
void ringmul_s13_mul( uint8_t * c0 , uint8_t * c1 , const uint8_t * a0_fft , const uint8_t * a1_fft , const uint8_t * b0_fft , const uint8_t * b1_fft );


/// @brief multiply two polynomials in F216[x]/ x^384
/// @param c0 [out] low  byte of F216. size : 384 bytes
/// @param c1 [out] high byte of F216. size : 384 bytes
/// @param a [in] size : 384 bytes
/// @param b [in] size : 384 bytes
void ringmul_mul_384( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b );

/// @brief multiply two polynomials in F216[x]/ x^1024
/// @param c0 [out] low  byte of F216. size : 1024 bytes
/// @param c1 [out] high byte of F216. size : 1024 bytes
/// @param a [in] size : 1024 bytes
/// @param b [in] size : 1024 bytes
void ringmul_mul_1024( uint8_t * c0 , uint8_t * c1 , const uint8_t * a , const uint8_t * b );

#ifdef  __cplusplus
}
#endif


#endif

