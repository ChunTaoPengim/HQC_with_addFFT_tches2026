
#include "bitpoly_to_gf256x2.h"


#include <arm_neon.h>

// static const uint16_t _bitpoly_to_gf256x2[8] __attribute__((aligned(32))) = { 0x1, 0x37c, 0x52c, 0xecc4, 0x112a, 0xb016, 0xd9a, 0xf806 };

static const uint8_t _tbl_bitpoly_to_gf256x2[16*4] __attribute__((aligned(32))) = {
    0x00, 0x01, 0x7c, 0x7d, 0x2c, 0x2d, 0x50, 0x51, 0xc4, 0xc5, 0xb8, 0xb9, 0xe8, 0xe9, 0x94, 0x95, 
    0x00, 0x00, 0x03, 0x03, 0x05, 0x05, 0x06, 0x06, 0xec, 0xec, 0xef, 0xef, 0xe9, 0xe9, 0xea, 0xea,
    0x00, 0x2a, 0x16, 0x3c, 0x9a, 0xb0, 0x8c, 0xa6, 0x06, 0x2c, 0x10, 0x3a, 0x9c, 0xb6, 0x8a, 0xa0,
    0x00, 0x11, 0xb0, 0xa1, 0x0d, 0x1c, 0xbd, 0xac, 0xf8, 0xe9, 0x48, 0x59, 0xf5, 0xe4, 0x45, 0x54,
};


void bitpoly_to_gf256x2_n( uint8_t * v0 , uint8_t * v1 , const uint8_t * bitpoly , unsigned n )
{
    uint8x16_t tabl_0 = vld1q_u8( _tbl_bitpoly_to_gf256x2 );
    uint8x16_t tabl_1 = vld1q_u8( _tbl_bitpoly_to_gf256x2 + 16 );
    uint8x16_t tabh_0 = vld1q_u8( _tbl_bitpoly_to_gf256x2 + 32);
    uint8x16_t tabh_1 = vld1q_u8( _tbl_bitpoly_to_gf256x2 + 48 );
    uint8x16_t mask_f = vdupq_n_u8( 0xf );

    unsigned rem = n&15;
    if (rem) { // do not expect this
        for(unsigned i=0;i<rem;i++){
            uint16_t r = bitpoly_to_gf256x2( bitpoly[i] );
            v0[i] = r&0xff;
            v1[i] = r>>8;
        }
        n -= rem;
        v0 += rem;
        v1 += rem;
        bitpoly += rem;
    }
    while(n) {
        uint8x16_t b = vld1q_u8( bitpoly );
        uint8x16_t bl = b&mask_f;
        uint8x16_t bh = vshrq_n_u8( b, 4 );
        uint8x16_t vv0 = vqtbl1q_u8( tabl_0 , bl )^vqtbl1q_u8( tabh_0 , bh );
        uint8x16_t vv1 = vqtbl1q_u8( tabl_1 , bl )^vqtbl1q_u8( tabh_1 , bh );
        vst1q_u8( v0 , vv0 );
        vst1q_u8( v1 , vv1 );
        n -= 16;
        v0 += 16;
        v1 += 16;
        bitpoly += 16;
    }
}

//static const uint16_t _gf256x2_to_bitpoly[16] __attribute__((aligned(32))) = { 0x1, 0x50f3, 0xce53, 0x6ee4, 0x6881, 0x4151, 0x409a, 0x8e55, 0xf586, 0x3c79, 0x1464, 0x3961, 0x8ad0, 0xd586, 0x5140, 0xa957 };

static const uint8_t _tbl_gf256x2_to_bitpoly[16*8] __attribute__((aligned(32))) = {
    0x00, 0x01, 0xf3, 0xf2, 0x53, 0x52, 0xa0, 0xa1, 0xe4, 0xe5, 0x17, 0x16, 0xb7, 0xb6, 0x44, 0x45,
    0x00, 0x00, 0x50, 0x50, 0xce, 0xce, 0x9e, 0x9e, 0x6e, 0x6e, 0x3e, 0x3e, 0xa0, 0xa0, 0xf0, 0xf0,
    0x00, 0x81, 0x51, 0xd0, 0x9a, 0x1b, 0xcb, 0x4a, 0x55, 0xd4, 0x04, 0x85, 0xcf, 0x4e, 0x9e, 0x1f,
    0x00, 0x68, 0x41, 0x29, 0x40, 0x28, 0x01, 0x69, 0x8e, 0xe6, 0xcf, 0xa7, 0xce, 0xa6, 0x8f, 0xe7,
    0x00, 0x86, 0x79, 0xff, 0x64, 0xe2, 0x1d, 0x9b, 0x61, 0xe7, 0x18, 0x9e, 0x05, 0x83, 0x7c, 0xfa,
    0x00, 0xf5, 0x3c, 0xc9, 0x14, 0xe1, 0x28, 0xdd, 0x39, 0xcc, 0x05, 0xf0, 0x2d, 0xd8, 0x11, 0xe4,
    0x00, 0xd0, 0x86, 0x56, 0x40, 0x90, 0xc6, 0x16, 0x57, 0x87, 0xd1, 0x01, 0x17, 0xc7, 0x91, 0x41,
    0x00, 0x8a, 0xd5, 0x5f, 0x51, 0xdb, 0x84, 0x0e, 0xa9, 0x23, 0x7c, 0xf6, 0xf8, 0x72, 0x2d, 0xa7,
};


void gf256x2_to_bitpoly_n( uint8_t * bp0 , uint8_t * bp1 , const uint8_t * v0 , const uint8_t * v1 , unsigned n )
{
    uint8x16_t mask_f = vdupq_n_u8( 0xf );
    uint8x16_t tab0l_0 = vld1q_u8( _tbl_gf256x2_to_bitpoly );
    uint8x16_t tab0l_1 = vld1q_u8( _tbl_gf256x2_to_bitpoly + 16 );
    uint8x16_t tab0h_0 = vld1q_u8( _tbl_gf256x2_to_bitpoly + 32 );
    uint8x16_t tab0h_1 = vld1q_u8( _tbl_gf256x2_to_bitpoly + 48 );
    uint8x16_t tab1l_0 = vld1q_u8( _tbl_gf256x2_to_bitpoly + 64 );
    uint8x16_t tab1l_1 = vld1q_u8( _tbl_gf256x2_to_bitpoly + 80 );
    uint8x16_t tab1h_0 = vld1q_u8( _tbl_gf256x2_to_bitpoly + 96 );
    uint8x16_t tab1h_1 = vld1q_u8( _tbl_gf256x2_to_bitpoly + 112 );


    unsigned rem = n&15;
    if (rem) { // do not expect this
        for(unsigned i=0;i<rem;i++){
            uint16_t r = gf256x2_to_bitpoly( v0[i] , v1[i] );
            bp0[i] = r&0xff;
            bp1[i] = r>>8;
        }
        n -= rem;
        v0 += rem;
        v1 += rem;
        bp0 += rem;
        bp1 += rem;
    }
    while(n) {
        uint8x16_t vv0 = vld1q_u8( v0 );
        uint8x16_t vv1 = vld1q_u8( v1 );
        uint8x16_t v0l = vv0&mask_f;
        uint8x16_t v0h = vshrq_n_u8( vv0, 4 );
        uint8x16_t v1l = vv1&mask_f;
        uint8x16_t v1h = vshrq_n_u8( vv1, 4 );

        uint8x16_t bb0 = vqtbl1q_u8( tab0l_0 , v0l )^vqtbl1q_u8( tab0h_0 , v0h )^vqtbl1q_u8( tab1l_0 , v1l )^vqtbl1q_u8( tab1h_0 , v1h );
        uint8x16_t bb1 = vqtbl1q_u8( tab0l_1 , v0l )^vqtbl1q_u8( tab0h_1 , v0h )^vqtbl1q_u8( tab1l_1 , v1l )^vqtbl1q_u8( tab1h_1 , v1h );
        vst1q_u8( bp0 , bb0 );
        vst1q_u8( bp1 , bb1 );
        n -= 16;
        v0 += 16;
        v1 += 16;
        bp0 += 16;
        bp1 += 16;
    }
}



