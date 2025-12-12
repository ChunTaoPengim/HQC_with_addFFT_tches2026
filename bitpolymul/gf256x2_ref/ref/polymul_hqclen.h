
#ifndef _POLYMUL_HQCLEN_H_
#define _POLYMUL_HQCLEN_H_


#include <stdint.h>

#include "polymul.h"

#ifdef  __cplusplus
extern  "C" {
#endif

#define POLYMUL_17920_FFTSIZE_BYTE  (8192 + 384)

///
/// @brief input transform for multiplying two bit-polynomials of size < 17920 bits
///
/// @param a_fft [out] size : 65536 + 3072 bits = 1024+48 u64  = 8192+384 bytes
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 256+24 u64 = 2048+192 bytes
void polymul_17920_input( uint64_t * a_fft , const uint64_t * a );

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 560 u64 = 4480=(4096+384) bytes
/// @param a_fft [in] a transformed bit-polynomial. size : 1072 u64 = 8192+384 bytes
/// @param b_fft [in] a transformed bit-polynomial. size : 1072 u64 = 8192+384 bytes
void polymul_17920_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft );

// target : 17669 bits
#define POLYMUL_280U64_FFTSIZE_U64  (1024)

///
/// @brief input transform for multiplying two bit-polynomials of size < (16384+1536) bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 65536 bits = 1024 u64  = 8192 bytes
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 256+24 u64 = 2048+192 bytes
void polymul_280U64_input( uint64_t * a_fft , const uint64_t * a );

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 512+48 u64 = 4480=(4096+384) bytes
/// @param a [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param b [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
void polymul_280U64_mul( uint64_t * c , const uint64_t * a , const uint64_t * b , const uint64_t * a_fft , const uint64_t * b_fft );




#define POLYMUL_36864_FFTSIZE_BYTE  (16384 + 1024)

///
/// @brief input transform for multiplying two bit-polynomials of size < 36860 bits
///
/// @param a_fft [out] size : 131072 + 8192 bits = 2048+128 u64 = 16384+1024 bytes
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 576 u64 = 4096+512 bytes
void polymul_36864_input( uint64_t * a_fft , const uint64_t * a );

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 65536 + 8192 bits = 1152 u64 = 8192+1024 bytes
/// @param a_fft [in] a transformed bit-polynomial. size : 2176 u64 = 16384+1024 bytes
/// @param b_fft [in] a transformed bit-polynomial. size : 2176 u64 = 16384+1024 bytes
void polymul_36864_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft );


// target : 35851 bits
#define POLYMUL_576U64_FFTSIZE_U64  (2048)

///
/// @brief input transform for multiplying two bit-polynomials of size < 32768+4096 bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 2048 u64  = 16384 bytes
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 512+64 u64 = 4608=(4096+512) bytes
void polymul_576U64_input( uint64_t * a_fft , const uint64_t * a );

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 1024+128 u64 = 8976=(8192+1024) bytes
/// @param a [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param b [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
void polymul_576U64_mul( uint64_t * c , const uint64_t * a , const uint64_t * b , const uint64_t * a_fft , const uint64_t * b_fft );



#ifdef  __cplusplus
}
#endif


#endif

