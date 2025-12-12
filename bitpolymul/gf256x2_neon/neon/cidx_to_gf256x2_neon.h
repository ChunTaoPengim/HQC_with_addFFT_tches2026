#ifndef _CIDX_TO_GF256X2_NEON_H_
#define _CIDX_TO_GF256X2_NEON_H_


#include <stdint.h>
#include <arm_neon.h>

// cidx_to_gf256x2 = [  0x1, 0xbc, 0x5c, 0xc, 0xae, 0x5a, 0xe, 0x84, 0x1f6, 0xbc5c, 0x5c18, 0xc7e, 0xae46, 0x5a68, 0xe02, 0x8456,]

static const uint8_t _tbl_cidx_to_gf256[16*4] __attribute__((aligned(32))) = {
    0x00, 0x01, 0xbc, 0xbd, 0x5c, 0x5d, 0xe0, 0xe1, 0x0c, 0x0d, 0xb0, 0xb1, 0x50, 0x51, 0xec, 0xed,
    0x00, 0xae, 0x5a, 0xf4, 0x0e, 0xa0, 0x54, 0xfa, 0x84, 0x2a, 0xde, 0x70, 0x8a, 0x24, 0xd0, 0x7e,
    0x00, 0xf6, 0x5c, 0xaa, 0x18, 0xee, 0x44, 0xb2, 0x7e, 0x88, 0x22, 0xd4, 0x66, 0x90, 0x3a, 0xcc,
    0x00, 0x46, 0x68, 0x2e, 0x02, 0x44, 0x6a, 0x2c, 0x56, 0x10, 0x3e, 0x78, 0x54, 0x12, 0x3c, 0x7a,
};


static inline
uint8x16_t cidx_to_gf256( uint8x16_t cidx )
{
    uint8x16_t mask_f = vdupq_n_u8( 0x0f );
    uint8x16_t tbl_l = vld1q_u8( _tbl_cidx_to_gf256 );
    uint8x16_t tbl_h = vld1q_u8( _tbl_cidx_to_gf256 + 16 );
    return vqtbl1q_u8( tbl_l , cidx&mask_f ) ^ vqtbl1q_u8( tbl_h , vshrq_n_u8(cidx, 4) );
}

static inline
uint8x16_t cidx_to_gf256x2_l( uint8x16_t cidx_l , uint8x16_t cidx_h )
{
    uint8x16_t mask_f = vdupq_n_u8( 0x0f );
    uint8x16_t tbl_0l_0 = vld1q_u8( _tbl_cidx_to_gf256 );
    uint8x16_t tbl_0h_0 = vld1q_u8( _tbl_cidx_to_gf256 + 16 );
    uint8x16_t tbl_1l_0 = vld1q_u8( _tbl_cidx_to_gf256 + 32);
    uint8x16_t tbl_1h_0 = vld1q_u8( _tbl_cidx_to_gf256 + 48 );

    return vqtbl1q_u8( tbl_0l_0 , cidx_l&mask_f ) ^ vqtbl1q_u8( tbl_0h_0 , vshrq_n_u8(cidx_l, 4) )
     ^ vqtbl1q_u8( tbl_1l_0 , cidx_h&mask_f ) ^ vqtbl1q_u8( tbl_1h_0 , vshrq_n_u8(cidx_h, 4) );
}

static inline
uint8x16_t cidx_to_gf256x2_h( uint8x16_t cidx_h ) { return cidx_to_gf256(cidx_h); }



#endif
