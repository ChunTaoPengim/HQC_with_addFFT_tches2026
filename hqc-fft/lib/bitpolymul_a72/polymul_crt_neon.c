#include "stdint.h"
#include "string.h"

#include "polymul.h"


#include "ringmul.h"

#include "gf256.h"

#include "combine.h"

///
/// @brief input transform for multiplying two bit-polynomials of size < 17920 bits
///
/// @param a_fft [out] size : 65536 + 3072 bits = 1024+48 u64  = 8192+384 bytes
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 256+24 u64 = 2048+192 bytes
void polymul_17920_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s12_input_2240( (uint8_t*)a_fft , ((uint8_t*)a_fft)+4096 ,  (uint8_t*)a );
    memcpy( ((uint8_t*)a_fft)+4096*2 , ((uint8_t*)a) , 384 );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 560 u64 = 4480=(4096+384) bytes
/// @param a_fft [in] a transformed bit-polynomial. size : 1072 u64 = 8192+384 bytes
/// @param b_fft [in] a transformed bit-polynomial. size : 1072 u64 = 8192+384 bytes
void polymul_17920_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft )
{
    polymul_280U64_mul( c , a_fft+(POLYMUL_280U64_FFTSIZE_U64) , b_fft+(POLYMUL_280U64_FFTSIZE_U64) , a_fft , b_fft );
}

/////////////////////

///
/// @brief input transform for multiplying two bit-polynomials of size < (16384+1536) bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 65536 bits = 1024 u64  = 8192 bytes
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 256+24 u64 = 2048+192 bytes
void polymul_280U64_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s12_input_2240( (uint8_t*)a_fft , ((uint8_t*)a_fft)+(POLYMUL_280U64_FFTSIZE_U64*4) ,  (uint8_t*)a );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 512+48 u64 = 4480=(4096+384) bytes
/// @param a [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param b [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
void polymul_280U64_mul( uint64_t * _c , const uint64_t * _a , const uint64_t * _b , const uint64_t * _a_fft , const uint64_t * _b_fft )
{
    uint8_t * c = (uint8_t*)_c;
    const uint8_t * a_fft = (const uint8_t*)_a_fft;
    const uint8_t * b_fft = (const uint8_t*)_b_fft;

    uint8_t pc_m384_l[384+32];
    uint8_t pc_m384_h[384+32];
    ringmul_mul_384( pc_m384_l , pc_m384_h , (const uint8_t*)_a , (const uint8_t*)_b );
    pc_m384_l[384] = 0;
    pc_m384_h[384] = 0;
    uint8_t pc_s12_l[4096+384+32];
    uint8_t pc_s12_h[4096+384+32];
    ringmul_s12_mul( pc_s12_l , pc_s12_h , a_fft , a_fft+4096 , b_fft , b_fft+4096 );
    memset( pc_s12_l+4096 , 0 , 384+32 );
    memset( pc_s12_h+4096 , 0 , 384+32 );

    ringmul_combine_4479(c, pc_m384_l, pc_m384_h, pc_s12_l, pc_s12_h);
}

/////////////////////

///
/// @brief input transform for multiplying two bit-polynomials of size < 32768+4096 bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 2048 u64  = 16384 bytes
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 512+64 u64 = 4608=(4096+512) bytes
void polymul_36864_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s13_input_4608( (uint8_t*)a_fft , ((uint8_t*)a_fft)+8192 ,  (uint8_t*)a );
    memcpy( ((uint8_t*)a_fft)+8192*2 , ((uint8_t*)a) , 1024 );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 65536 + 8192 bits = 1152 u64 = 8192+1024 bytes
/// @param a_fft [in] a transformed bit-polynomial. size : 2176 u64 = 16384+1024 bytes
/// @param b_fft [in] a transformed bit-polynomial. size : 2176 u64 = 16384+1024 bytes
void polymul_36864_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft )
{
    polymul_576U64_mul( c , a_fft+(POLYMUL_576U64_FFTSIZE_U64) , b_fft+(POLYMUL_576U64_FFTSIZE_U64) , a_fft , b_fft );
}

/////////////////////

///
/// @brief input transform for multiplying two bit-polynomials of size < 32768+4096 bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 2048 u64  = 16384 bytes
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 512+64 u64 = 4608=(4096+512) bytes
void polymul_576U64_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s13_input_4608( (uint8_t*)a_fft , ((uint8_t*)a_fft)+(POLYMUL_576U64_FFTSIZE_U64*4) ,  (uint8_t*)a );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 1024+128 u64 = 8976=(8192+1024) bytes
/// @param a [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param b [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
void polymul_576U64_mul( uint64_t * _c , const uint64_t * _a , const uint64_t * _b , const uint64_t * _a_fft , const uint64_t * _b_fft )
{
    uint8_t * c = (uint8_t*)_c;
    const uint8_t * a_fft = (const uint8_t*)_a_fft;
    const uint8_t * b_fft = (const uint8_t*)_b_fft;

    uint8_t pc_m1024_l[1024+32];
    uint8_t pc_m1024_h[1024+32];
    ringmul_mul_1024( pc_m1024_l , pc_m1024_h , (const uint8_t*)_a , (const uint8_t*)_b );
    pc_m1024_l[1024] = 0;
    pc_m1024_h[1024] = 0;
    uint8_t pc_s13_l[8192+1024+32];
    uint8_t pc_s13_h[8192+1024+32];
    ringmul_s13_mul( pc_s13_l , pc_s13_h , a_fft , a_fft+8192 , b_fft , b_fft+8192 );
    memset( pc_s13_l+8192 , 0 , 1024+32 );
    memset( pc_s13_h+8192 , 0 , 1024+32 );

    ringmul_combine_9215(c, pc_m1024_l, pc_m1024_h, pc_s13_l, pc_s13_h);
}

