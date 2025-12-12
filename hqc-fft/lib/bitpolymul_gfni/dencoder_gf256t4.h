
#ifndef _DENCODE_GF256T4_H_
#define _DENCODE_GF256T4_H_


#include <stdint.h>


#ifdef  __cplusplus
extern  "C" {
#endif


/// @brief encode bit-vector input of size (n*src_bits) bits into n gf256t4 elements
/// @param v0 0-th byte of gf256t4 vector 
/// @param v1 1-st byte of gf256t4 vector
/// @param v2 2-nd byte of gf256t4 vector
/// @param v3 3-rd byte of gf256t4 vector
/// @param n number of output gf256t4 elements
/// @param bitvec input of a bit vector of size n*src_bits bits.
/// @param src_bits number of meaningful bits of gf256t4 elements, optimizing for src_bits=16.
void encode_gf256t4( uint8_t * v0 , uint8_t * v1, uint8_t * v2, uint8_t * v3, unsigned n,
    const uint32_t * bitvec , unsigned src_bits );

/// @brief reverse operation of encode_gf256t4()
/// @param bitvec output of a bit vector of size n*32 bits
/// @param v0 0-th byte of gf256t4 vector
/// @param v1 1-st byte of gf256t4 vector
/// @param v2 2-nd byte of gf256t4 vector
/// @param v3 3-rd byte of gf256t4 vector
/// @param n number of input gf256t4 elements, equal to (n*32)-bit output.
void decode_gf256t4( uint32_t * bitvec 
    , const uint8_t * v0, const uint8_t * v1, const uint8_t * v2, const uint8_t * v3, unsigned n );




#ifdef  __cplusplus
}
#endif


#endif
