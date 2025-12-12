
#include "bc_1.h"



void bc_1_256( void *_poly , unsigned n_256bit )
{
  uint32_t *poly = (uint32_t*)_poly;
  for(int i=n_256bit;i>0;i--) {
    uint32_t p0 = poly[0];
    uint32_t p1 = poly[1];
    uint32_t p2 = poly[2];
    uint32_t p3 = poly[3];
    uint32_t p4 = poly[4];
    uint32_t p5 = poly[5];
    uint32_t p6 = poly[6];
    uint32_t p7 = poly[7];

//layer 13: [256] s4^8:(128,8) suggest unit:1
    p4 ^= p7>>24;
    p3 ^= (p7<<8)^(p6>>24);
    p2 ^= (p6<<8)^(p5>>24);
    p1 ^= (p5<<8)^(p4>>24);
    p0 ^= (p4<<8);
//layer 14: [128] s4^4:(64,4) suggest unit:1
    p2 ^= p3>>28;
    p1 ^= (p3<<4)^(p2>>28);
    p0 ^= (p2<<4);

    p6 ^= p7>>28;
    p5 ^= (p7<<4)^(p6>>28);
    p4 ^= (p6<<4);
//layer 15: [64] s4^2:(32,2) suggest unit:1
    p1 ^= p1>>30;
    p3 ^= p3>>30;
    p5 ^= p5>>30;
    p7 ^= p7>>30;
    p0 ^= p1<<2;
    p2 ^= p3<<2;
    p4 ^= p5<<2;
    p6 ^= p7<<2;

//layer 16: [32] s4^1:(16,1) suggest unit:1
    p0 ^= ((p0&0x80000000)>>15);
    p1 ^= ((p1&0x80000000)>>15);
    p2 ^= ((p2&0x80000000)>>15);
    p3 ^= ((p3&0x80000000)>>15);
    p4 ^= ((p4&0x80000000)>>15);
    p5 ^= ((p5&0x80000000)>>15);
    p6 ^= ((p6&0x80000000)>>15);
    p7 ^= ((p7&0x80000000)>>15);
    p0 ^= ((p0&0x7fff0000)>>15);
    p1 ^= ((p1&0x7fff0000)>>15);
    p2 ^= ((p2&0x7fff0000)>>15);
    p3 ^= ((p3&0x7fff0000)>>15);
    p4 ^= ((p4&0x7fff0000)>>15);
    p5 ^= ((p5&0x7fff0000)>>15);
    p6 ^= ((p6&0x7fff0000)>>15);
    p7 ^= ((p7&0x7fff0000)>>15);

//layer 17: [256] s2^2:(128,32) suggest unit:16
    p4 ^= p7;
    p3 ^= p6;
    p2 ^= p5;
    p1 ^= p4;
//layer 18: [128] s2^1:(64,16) suggest unit:16
    p2 ^= p3>>16;
    p1 ^= (p3<<16)^(p2>>16);
    p0 ^= (p2<<16);
    p6 ^= p7>>16;
    p5 ^= (p7<<16)^(p6>>16);
    p4 ^= (p6<<16);
//layer 19: [256] s1^1:(128,64) suggest unit:64
    p5 ^= p7;
    p4 ^= p6;
    p3 ^= p5;
    p2 ^= p4;
//layer 20: [64] s1^1:(32,16) suggest unit:16
    p1 ^= p1>>16;
    p3 ^= p3>>16;
    p5 ^= p5>>16;
    p7 ^= p7>>16;
    p0 ^= p1<<16;
    p2 ^= p3<<16;
    p4 ^= p5<<16;
    p6 ^= p7<<16;
//layer 21: [16] s2^2:(8,2) suggest unit:1
    p0 ^= (p0&0xc000c000)>>6;
    p1 ^= (p1&0xc000c000)>>6;
    p2 ^= (p2&0xc000c000)>>6;
    p3 ^= (p3&0xc000c000)>>6;
    p4 ^= (p4&0xc000c000)>>6;
    p5 ^= (p5&0xc000c000)>>6;
    p6 ^= (p6&0xc000c000)>>6;
    p7 ^= (p7&0xc000c000)>>6;
    p0 ^= (p0&0x3f003f00)>>6;
    p1 ^= (p1&0x3f003f00)>>6;
    p2 ^= (p2&0x3f003f00)>>6;
    p3 ^= (p3&0x3f003f00)>>6;
    p4 ^= (p4&0x3f003f00)>>6;
    p5 ^= (p5&0x3f003f00)>>6;
    p6 ^= (p6&0x3f003f00)>>6;
    p7 ^= (p7&0x3f003f00)>>6;
//layer 22: [8] s2^1:(4,1) suggest unit:1
    p0 ^= ((p0&0x80808080)>>3);
    p1 ^= ((p1&0x80808080)>>3);
    p2 ^= ((p2&0x80808080)>>3);
    p3 ^= ((p3&0x80808080)>>3);
    p4 ^= ((p4&0x80808080)>>3);
    p5 ^= ((p5&0x80808080)>>3);
    p6 ^= ((p6&0x80808080)>>3);
    p7 ^= ((p7&0x80808080)>>3);
    p0 ^= ((p0&0x70707070)>>3);
    p1 ^= ((p1&0x70707070)>>3);
    p2 ^= ((p2&0x70707070)>>3);
    p3 ^= ((p3&0x70707070)>>3);
    p4 ^= ((p4&0x70707070)>>3);
    p5 ^= ((p5&0x70707070)>>3);
    p6 ^= ((p6&0x70707070)>>3);
    p7 ^= ((p7&0x70707070)>>3);

//layer 23: [16] s1^1:(8,4) suggest unit:4
    p0 ^= ((p0&0xf000f000)>>4);
    p1 ^= ((p1&0xf000f000)>>4);
    p2 ^= ((p2&0xf000f000)>>4);
    p3 ^= ((p3&0xf000f000)>>4);
    p4 ^= ((p4&0xf000f000)>>4);
    p5 ^= ((p5&0xf000f000)>>4);
    p6 ^= ((p6&0xf000f000)>>4);
    p7 ^= ((p7&0xf000f000)>>4);
    p0 ^= ((p0&0x0f000f00)>>4);
    p1 ^= ((p1&0x0f000f00)>>4);
    p2 ^= ((p2&0x0f000f00)>>4);
    p3 ^= ((p3&0x0f000f00)>>4);
    p4 ^= ((p4&0x0f000f00)>>4);
    p5 ^= ((p5&0x0f000f00)>>4);
    p6 ^= ((p6&0x0f000f00)>>4);
    p7 ^= ((p7&0x0f000f00)>>4);

//layer 24: [4] s1^1:(2,1) suggest unit:1
    p0 ^= ((p0&0x88888888)>>1);
    p1 ^= ((p1&0x88888888)>>1);
    p2 ^= ((p2&0x88888888)>>1);
    p3 ^= ((p3&0x88888888)>>1);
    p4 ^= ((p4&0x88888888)>>1);
    p5 ^= ((p5&0x88888888)>>1);
    p6 ^= ((p6&0x88888888)>>1);
    p7 ^= ((p7&0x88888888)>>1);
    p0 ^= ((p0&0x44444444)>>1);
    p1 ^= ((p1&0x44444444)>>1);
    p2 ^= ((p2&0x44444444)>>1);
    p3 ^= ((p3&0x44444444)>>1);
    p4 ^= ((p4&0x44444444)>>1);
    p5 ^= ((p5&0x44444444)>>1);
    p6 ^= ((p6&0x44444444)>>1);
    p7 ^= ((p7&0x44444444)>>1);

    poly[0] = p0;
    poly[1] = p1;
    poly[2] = p2;
    poly[3] = p3;
    poly[4] = p4;
    poly[5] = p5;
    poly[6] = p6;
    poly[7] = p7;
    poly += 8;
  }
}

void ibc_1_256( void *_poly , unsigned n_256bit )
{
  uint32_t *poly = (uint32_t*)_poly;
  for(int i=n_256bit;i>0;i--) {
    uint32_t p0 = poly[0];
    uint32_t p1 = poly[1];
    uint32_t p2 = poly[2];
    uint32_t p3 = poly[3];
    uint32_t p4 = poly[4];
    uint32_t p5 = poly[5];
    uint32_t p6 = poly[6];
    uint32_t p7 = poly[7];

//layer 7: [4] s1^1:(2,1) suggest unit:1
    p0 ^= ((p0&0xcccccccc)>>1);
    p1 ^= ((p1&0xcccccccc)>>1);
    p2 ^= ((p2&0xcccccccc)>>1);
    p3 ^= ((p3&0xcccccccc)>>1);
    p4 ^= ((p4&0xcccccccc)>>1);
    p5 ^= ((p5&0xcccccccc)>>1);
    p6 ^= ((p6&0xcccccccc)>>1);
    p7 ^= ((p7&0xcccccccc)>>1);

//layer 6: [16] s1^1:(8,4) suggest unit:4
    p0 ^= ((p0&0xff00ff00)>>4);
    p1 ^= ((p1&0xff00ff00)>>4);
    p2 ^= ((p2&0xff00ff00)>>4);
    p3 ^= ((p3&0xff00ff00)>>4);
    p4 ^= ((p4&0xff00ff00)>>4);
    p5 ^= ((p5&0xff00ff00)>>4);
    p6 ^= ((p6&0xff00ff00)>>4);
    p7 ^= ((p7&0xff00ff00)>>4);

//layer 5: [8] s2^1:(4,1) suggest unit:1
    p0 ^= ((p0&0xf0f0f0f0)>>3);
    p1 ^= ((p1&0xf0f0f0f0)>>3);
    p2 ^= ((p2&0xf0f0f0f0)>>3);
    p3 ^= ((p3&0xf0f0f0f0)>>3);
    p4 ^= ((p4&0xf0f0f0f0)>>3);
    p5 ^= ((p5&0xf0f0f0f0)>>3);
    p6 ^= ((p6&0xf0f0f0f0)>>3);
    p7 ^= ((p7&0xf0f0f0f0)>>3);

//layer 4: [16] s2^2:(8,2) suggest unit:1
    p0 ^= ((p0&0xff00ff00)>>6);
    p1 ^= ((p1&0xff00ff00)>>6);
    p2 ^= ((p2&0xff00ff00)>>6);
    p3 ^= ((p3&0xff00ff00)>>6);
    p4 ^= ((p4&0xff00ff00)>>6);
    p5 ^= ((p5&0xff00ff00)>>6);
    p6 ^= ((p6&0xff00ff00)>>6);
    p7 ^= ((p7&0xff00ff00)>>6);

//layer 3: [64] s1^1:(32,16) suggest unit:16
    p0 ^= p1<<16;
    p2 ^= p3<<16;
    p4 ^= p5<<16;
    p6 ^= p7<<16;

    p1 ^= p1>>16;
    p3 ^= p3>>16;
    p5 ^= p5>>16;
    p7 ^= p7>>16;

//layer 2: [256] s1^1:(128,64) suggest unit:64
    p3 ^= p5;
    p2 ^= p4;

    p5 ^= p7;
    p4 ^= p6;

//layer 1: [128] s2^1:(64,16) suggest unit:16
    p0 ^= (p2<<16);
    p1 ^= (p3<<16)^(p2>>16);
    p2 ^= p3>>16;

    p4 ^= (p6<<16);
    p5 ^= (p7<<16)^(p6>>16);
    p6 ^= p7>>16;

//layer 0: [256] s2^2:(128,32) suggest unit:16
    p1 ^= p4;
    p2 ^= p5;
    p3 ^= p6;
    p4 ^= p7;

//layer 16: [32] s4^1:(16,1) suggest unit:1
    p0 ^= ((p0&0xffff0000)>>15);
    p1 ^= ((p1&0xffff0000)>>15);
    p2 ^= ((p2&0xffff0000)>>15);
    p3 ^= ((p3&0xffff0000)>>15);
    p4 ^= ((p4&0xffff0000)>>15);
    p5 ^= ((p5&0xffff0000)>>15);
    p6 ^= ((p6&0xffff0000)>>15);
    p7 ^= ((p7&0xffff0000)>>15);

//layer 15: [64] s4^2:(32,2) suggest unit:1
    p0 ^= p1<<2;
    p2 ^= p3<<2;
    p4 ^= p5<<2;
    p6 ^= p7<<2;

    p1 ^= p1>>30;
    p3 ^= p3>>30;
    p5 ^= p5>>30;
    p7 ^= p7>>30;

//layer 14: [128] s4^4:(64,4) suggest unit:1
    p0 ^= (p2<<4);
    p1 ^= (p3<<4)^(p2>>28);
    p2 ^= p3>>28;

    p4 ^= (p6<<4);
    p5 ^= (p7<<4)^(p6>>28);
    p6 ^= p7>>28;

//layer 13: [256] s4^8:(128,8) suggest unit:1
    p0 ^= (p4<<8);
    p1 ^= (p5<<8)^(p4>>24);
    p2 ^= (p6<<8)^(p5>>24);
    p3 ^= (p7<<8)^(p6>>24);
    p4 ^= p7>>24;


    poly[0] = p0;
    poly[1] = p1;
    poly[2] = p2;
    poly[3] = p3;
    poly[4] = p4;
    poly[5] = p5;
    poly[6] = p6;
    poly[7] = p7;

    poly += 8;
  }
}






/////////////////////////////////////////


static inline
void _div_2terms_poly( uint32_t * poly , unsigned HT_deg_X32x, unsigned TT_deg_X32x ) {
    for(unsigned i=(HT_deg_X32x*2-1);i>=HT_deg_X32x;i--) poly[i-HT_deg_X32x+TT_deg_X32x] ^= poly[i];
}


static inline
void _div_X32x128_X16( uint32_t *poly ) {
#define H 128
#define L 16
  unsigned src1;
  unsigned src0 = poly[2*H-1];
  poly[H] ^= src0>>(32-L);
  for(unsigned i=H-1;i>0;i--) {
    src1 = src0;
    src0 = poly[i+H-1];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[0] ^= (src0<<L);
#undef H
#undef L
}

static inline
void _div_X32x64_X8( uint32_t *poly ) {
#define H 64
#define L 8
  unsigned src1;
  unsigned src0 = poly[2*H-1];
  poly[H] ^= src0>>(32-L);
  for(unsigned i=H-1;i>0;i--) {
    src1 = src0;
    src0 = poly[i+H-1];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[0] ^= (src0<<L);
#undef H
#undef L
}

static inline
void _div_X32x32_X4( uint32_t *poly ) {
#define H 32
#define L 4
  unsigned src1;
  unsigned src0 = poly[2*H-1];
  poly[H] ^= src0>>(32-L);
  for(unsigned i=H-1;i>0;i--) {
    src1 = src0;
    src0 = poly[i+H-1];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[0] ^= (src0<<L);
#undef H
#undef L
}

static inline
void _div_X32x16_X2( uint32_t *poly ) {
#define H 16
#define L 2
  unsigned src1;
  unsigned src0 = poly[2*H-1];
  poly[H] ^= src0>>(32-L);
  for(unsigned i=H-1;i>0;i--) {
    src1 = src0;
    src0 = poly[i+H-1];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[0] ^= (src0<<L);
#undef H
#undef L
}

static inline
void _div_X32x8_X1( uint32_t *poly ) {
#define H 8
#define L 1
  unsigned src1;
  unsigned src0 = poly[2*H-1];
  poly[H] ^= src0>>(32-L);
  for(unsigned i=H-1;i>0;i--) {
    src1 = src0;
    src0 = poly[i+H-1];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[0] ^= (src0<<L);
#undef H
#undef L
}




// s8 = x^256 - x = x^(32*8) - x
// s8^2 = x^512 - x^2 = x^(32*16) - x^2
// s8^4 = x^(32*32) - x^4
// s8^8 = x^(32*64) - x^8
// s8^16 = x^(32*128) - x^16
// s8^32 = x^(32*256) - x^32   <-- compute in 32-bit units
// s8^64 = x^(32*512) - x^(32*2)



static
void repr_s8_1( uint32_t * poly , unsigned n_byte )
{
  unsigned n_32 = n_byte>>2;
  unsigned log_n_32 = __builtin_ctz( n_32 );
  for( unsigned i=log_n_32-1;i>=8;i--) {
    unsigned s_ht = 1<<i;
    unsigned s_tt = 1<<(i-8);
    unsigned len = s_ht<<1;
    // div X^{32*s_ht} - X^{32*s_tt}
    for(unsigned j=0;j<n_32;j+=len) { _div_2terms_poly( poly+j , s_ht , s_tt ); }
  }
  // div  x^(32*128) - x^16
  if( log_n_32 > 6 ) { for(unsigned j=0;j<n_32;j+=256) _div_X32x128_X16( poly+j ); }
  // div  x^(32*64) - x^8
  if( log_n_32 > 5 ) { for(unsigned j=0;j<n_32;j+=128) _div_X32x64_X8( poly+j ); }
  // div  x^(32*32) - x^4
  if( log_n_32 > 4 ) { for(unsigned j=0;j<n_32;j+=64) _div_X32x32_X4( poly+j ); }
  // div  x^(32*16) - x^2
  if( log_n_32 > 3 ) { for(unsigned j=0;j<n_32;j+=32) _div_X32x16_X2( poly+j ); }
  // div  x^(32*8) - x^1
  if( log_n_32 > 2 ) { for(unsigned j=0;j<n_32;j+=16) _div_X32x8_X1( poly+j ); }
}





static inline
void _idiv_2terms_poly( uint32_t * poly , unsigned HT_deg_X32x, unsigned TT_deg_X32x ) {
    for(unsigned i=HT_deg_X32x;i<HT_deg_X32x*2;i++) poly[i-HT_deg_X32x+TT_deg_X32x] ^= poly[i];
}


static inline
void _idiv_X32x128_X16( uint32_t *poly ) {
#define H 128
#define L 16
  unsigned src0;
  unsigned src1 = poly[H];
  poly[0] ^= (src1<<L);
  for(unsigned i=1;i<H;i++) {
    src0 = src1;
    src1 = poly[i+H];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[H] ^= src1>>(32-L);
#undef H
#undef L
}

static inline
void _idiv_X32x64_X8( uint32_t *poly ) {
#define H 64
#define L 8
  unsigned src0;
  unsigned src1 = poly[H];
  poly[0] ^= (src1<<L);
  for(unsigned i=1;i<H;i++) {
    src0 = src1;
    src1 = poly[i+H];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[H] ^= src1>>(32-L);
#undef H
#undef L
}

static inline
void _idiv_X32x32_X4( uint32_t *poly ) {
#define H 32
#define L 4
  unsigned src0;
  unsigned src1 = poly[H];
  poly[0] ^= (src1<<L);
  for(unsigned i=1;i<H;i++) {
    src0 = src1;
    src1 = poly[i+H];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[H] ^= src1>>(32-L);
#undef H
#undef L
}

static inline
void _idiv_X32x16_X2( uint32_t *poly ) {
#define H 16
#define L 2
  unsigned src0;
  unsigned src1 = poly[H];
  poly[0] ^= (src1<<L);
  for(unsigned i=1;i<H;i++) {
    src0 = src1;
    src1 = poly[i+H];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[H] ^= src1>>(32-L);
#undef H
#undef L
}

static inline
void _idiv_X32x8_X1( uint32_t *poly ) {
#define H 8
#define L 1
  unsigned src0;
  unsigned src1 = poly[H];
  poly[0] ^= (src1<<L);
  for(unsigned i=1;i<H;i++) {
    src0 = src1;
    src1 = poly[i+H];
    poly[i] ^= (src1<<L) | (src0>>(32-L));
  }
  poly[H] ^= src1>>(32-L);
#undef H
#undef L
}


static
void irepr_s8_1( uint32_t * poly , unsigned n_byte )
{
  unsigned n_32 = n_byte>>2;
  unsigned log_n_32 = __builtin_ctz( n_32 );

  // div  x^(32*8) - x^1
  if( log_n_32 > 3 ) { for(unsigned j=0;j<n_32;j+=16) _idiv_X32x8_X1( poly+j ); }
  // div  x^(32*16) - x^2
  if( log_n_32 > 4 ) { for(unsigned j=0;j<n_32;j+=32) _idiv_X32x16_X2( poly+j ); }
  // div  x^(32*32) - x^4
  if( log_n_32 > 5 ) { for(unsigned j=0;j<n_32;j+=64) _idiv_X32x32_X4( poly+j ); }
  // div  x^(32*64) - x^8
  if( log_n_32 > 6 ) { for(unsigned j=0;j<n_32;j+=128) _idiv_X32x64_X8( poly+j ); }
  // div  x^(32*128) - x^16
  if( log_n_32 > 7 ) { for(unsigned j=0;j<n_32;j+=256) _idiv_X32x128_X16( poly+j ); }

  for( unsigned i=8;i<log_n_32;i++) {
    unsigned s_ht = 1<<i;
    unsigned s_tt = 1<<(i-8);
    unsigned len = s_ht<<1;
    // div X^{32*s_ht} - X^{32*s_tt}
    for(unsigned j=0;j<n_32;j+=len) { _idiv_2terms_poly( poly+j , s_ht , s_tt ); }
  }

}


/////////////////////////////////////////
//
// Optimized version for n_bits = 16384, 32768, 65536
//
/////////////////////////////////////////


static
void repr_s8_1_16384( uint32_t *poly )
{
  uint32_t *ptr;
//layer 0: [16384] s8^32:(8192,32) suggest unit:1
  for(int i=(8192/32)-4;i>=0;i-=4) {
    uint32_t p0 = poly[i+(8192/32)];
    uint32_t p1 = poly[i+(8192/32)+1];
    uint32_t p2 = poly[i+(8192/32)+2];
    uint32_t p3 = poly[i+(8192/32)+3];
    uint32_t q0 = poly[i+1];
    uint32_t q1 = poly[i+1+1];
    uint32_t q2 = poly[i+1+2];
    uint32_t q3 = poly[i+1+3];
    poly[i+1] = q0^p0;
    poly[i+1+1] = q1^p1;
    poly[i+1+2] = q2^p2;
    poly[i+1+3] = q3^p3;
  }
//layer 1: [8192] s8^16:(4096,16) suggest unit:1
  ptr = poly;
  for(int j=(16384/8192);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(4096/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(4096/32)+3];
      uint32_t p2 = ptr[i+(4096/32)+2];
      uint32_t p1 = ptr[i+(4096/32)+1];
      uint32_t p0 = ptr[i+(4096/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>16)^(leftover<<16);
      ptr[i+1+2] = q2^(p2>>16)^(p3<<16);
      ptr[i+1+1] = q1^(p1>>16)^(p2<<16);
      ptr[i+1] = q0^(p0>>16)^(p1<<16);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<16);
    ptr += (8192/32);
  }

//layer 2: [4096] s8^8:(2048,8) suggest unit:1
  ptr = poly;
  for(int j=(16384/4096);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(2048/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(2048/32)+3];
      uint32_t p2 = ptr[i+(2048/32)+2];
      uint32_t p1 = ptr[i+(2048/32)+1];
      uint32_t p0 = ptr[i+(2048/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>24)^(leftover<<8);
      ptr[i+1+2] = q2^(p2>>24)^(p3<<8);
      ptr[i+1+1] = q1^(p1>>24)^(p2<<8);
      ptr[i+1] = q0^(p0>>24)^(p1<<8);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<8);
    ptr += (4096/32);
  }

//layer 3: [2048] s8^4:(1024,4) suggest unit:1
  ptr = poly;
  for(int j=(16384/2048);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(1024/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(1024/32)+3];
      uint32_t p2 = ptr[i+(1024/32)+2];
      uint32_t p1 = ptr[i+(1024/32)+1];
      uint32_t p0 = ptr[i+(1024/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>28)^(leftover<<4);
      ptr[i+1+2] = q2^(p2>>28)^(p3<<4);
      ptr[i+1+1] = q1^(p1>>28)^(p2<<4);
      ptr[i+1] = q0^(p0>>28)^(p1<<4);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<4);
    ptr += (2048/32);
  }

//layer 4: [1024] s8^2:(512,2) suggest unit:1
  ptr = poly;
  for(int j=(16384/1024);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(512/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(512/32)+3];
      uint32_t p2 = ptr[i+(512/32)+2];
      uint32_t p1 = ptr[i+(512/32)+1];
      uint32_t p0 = ptr[i+(512/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>30)^(leftover<<2);
      ptr[i+1+2] = q2^(p2>>30)^(p3<<2);
      ptr[i+1+1] = q1^(p1>>30)^(p2<<2);
      ptr[i+1] = q0^(p0>>30)^(p1<<2);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<2);
    ptr += (1024/32);
  }

//layer 5: [512] s8^1:(256,1) suggest unit:1
  ptr = poly;
  for(int j=(16384/512);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(256/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(256/32)+3];
      uint32_t p2 = ptr[i+(256/32)+2];
      uint32_t p1 = ptr[i+(256/32)+1];
      uint32_t p0 = ptr[i+(256/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>31)^(leftover<<1);
      ptr[i+1+2] = q2^(p2>>31)^(p3<<1);
      ptr[i+1+1] = q1^(p1>>31)^(p2<<1);
      ptr[i+1] = q0^(p0>>31)^(p1<<1);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<1);
    ptr += (512/32);
  }
}


static
void irepr_s8_1_16384( uint32_t *poly )
{
  uint32_t *ptr;

//layer 5: [512] s8^1:(256,1) suggest unit:1
  ptr = poly;
  for(int j=(16384/512);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(256/32);i+=4) {
      uint32_t p3 = ptr[i+(256/32)+3];
      uint32_t p2 = ptr[i+(256/32)+2];
      uint32_t p1 = ptr[i+(256/32)+1];
      uint32_t p0 = ptr[i+(256/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>31)^(p3<<1);
      ptr[i+2] = q2^(p1>>31)^(p2<<1);
      ptr[i+1] = q1^(p0>>31)^(p1<<1);
      ptr[i] = q0^(leftover>>31)^(p0<<1);
      leftover= p3;
    }
    ptr[(256/32)] ^= (leftover>>31);
    ptr += (512/32);
  }

//layer 4: [1024] s8^2:(512,2) suggest unit:1
  ptr = poly;
  for(int j=(16384/1024);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(512/32);i+=4) {
      uint32_t p3 = ptr[i+(512/32)+3];
      uint32_t p2 = ptr[i+(512/32)+2];
      uint32_t p1 = ptr[i+(512/32)+1];
      uint32_t p0 = ptr[i+(512/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>30)^(p3<<2);
      ptr[i+2] = q2^(p1>>30)^(p2<<2);
      ptr[i+1] = q1^(p0>>30)^(p1<<2);
      ptr[i] = q0^(leftover>>30)^(p0<<2);
      leftover= p3;
    }
    ptr[(512/32)] ^= (leftover>>30);
    ptr += (1024/32);
  }

//layer 3: [2048] s8^4:(1024,4) suggest unit:1
  ptr = poly;
  for(int j=(16384/2048);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(1024/32);i+=4) {
      uint32_t p3 = ptr[i+(1024/32)+3];
      uint32_t p2 = ptr[i+(1024/32)+2];
      uint32_t p1 = ptr[i+(1024/32)+1];
      uint32_t p0 = ptr[i+(1024/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>28)^(p3<<4);
      ptr[i+2] = q2^(p1>>28)^(p2<<4);
      ptr[i+1] = q1^(p0>>28)^(p1<<4);
      ptr[i] = q0^(leftover>>28)^(p0<<4);
      leftover= p3;
    }
    ptr[(1024/32)] ^= (leftover>>28);
    ptr += (2048/32);
  }

//layer 2: [4096] s8^8:(2048,8) suggest unit:1
  ptr = poly;
  for(int j=(16384/4096);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(2048/32);i+=4) {
      uint32_t p3 = ptr[i+(2048/32)+3];
      uint32_t p2 = ptr[i+(2048/32)+2];
      uint32_t p1 = ptr[i+(2048/32)+1];
      uint32_t p0 = ptr[i+(2048/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>24)^(p3<<8);
      ptr[i+2] = q2^(p1>>24)^(p2<<8);
      ptr[i+1] = q1^(p0>>24)^(p1<<8);
      ptr[i] = q0^(leftover>>24)^(p0<<8);
      leftover= p3;
    }
    ptr[(2048/32)] ^= (leftover>>24);
    ptr += (4096/32);
  }

//layer 1: [8192] s8^16:(4096,16) suggest unit:1
  ptr = poly;
  for(int j=(16384/8192);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(4096/32);i+=4) {
      uint32_t p3 = ptr[i+(4096/32)+3];
      uint32_t p2 = ptr[i+(4096/32)+2];
      uint32_t p1 = ptr[i+(4096/32)+1];
      uint32_t p0 = ptr[i+(4096/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>16)^(p3<<16);
      ptr[i+2] = q2^(p1>>16)^(p2<<16);
      ptr[i+1] = q1^(p0>>16)^(p1<<16);
      ptr[i] = q0^(leftover>>16)^(p0<<16);
      leftover= p3;
    }
    ptr[(4096/32)] ^= (leftover>>16);
    ptr += (8192/32);
  }

//layer 0: [16384] s8^32:(8192,32) suggest unit:1
  for(int i=0;i<(8192/32);i+=4) {
    uint32_t p0 = poly[i+(8192/32)];
    uint32_t p1 = poly[i+(8192/32)+1];
    uint32_t p2 = poly[i+(8192/32)+2];
    uint32_t p3 = poly[i+(8192/32)+3];
    uint32_t q0 = poly[i+1];
    uint32_t q1 = poly[i+1+1];
    uint32_t q2 = poly[i+1+2];
    uint32_t q3 = poly[i+1+3];
    poly[i+1] = q0^p0;
    poly[i+1+1] = q1^p1;
    poly[i+1+2] = q2^p2;
    poly[i+1+3] = q3^p3;
  }

}

//////////////////


static
void repr_s8_1_32768( uint32_t *poly )
{
  const unsigned w = 32768;
  uint32_t *ptr;

//layer 0: [32768] s8^64:(16384,64) suggest unit:1
#define DEG_H 16384
#define DEG_L 64
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 0: [16384] s8^32:(8192,32) suggest unit:1
#define DEG_H 8192
#define DEG_L 32
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 1: [8192] s8^16:(4096,16) suggest unit:1
  ptr = poly;
  for(int j=(w/8192);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(4096/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(4096/32)+3];
      uint32_t p2 = ptr[i+(4096/32)+2];
      uint32_t p1 = ptr[i+(4096/32)+1];
      uint32_t p0 = ptr[i+(4096/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>16)^(leftover<<16);
      ptr[i+1+2] = q2^(p2>>16)^(p3<<16);
      ptr[i+1+1] = q1^(p1>>16)^(p2<<16);
      ptr[i+1] = q0^(p0>>16)^(p1<<16);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<16);
    ptr += (8192/32);
  }

//layer 2: [4096] s8^8:(2048,8) suggest unit:1
  ptr = poly;
  for(int j=(w/4096);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(2048/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(2048/32)+3];
      uint32_t p2 = ptr[i+(2048/32)+2];
      uint32_t p1 = ptr[i+(2048/32)+1];
      uint32_t p0 = ptr[i+(2048/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>24)^(leftover<<8);
      ptr[i+1+2] = q2^(p2>>24)^(p3<<8);
      ptr[i+1+1] = q1^(p1>>24)^(p2<<8);
      ptr[i+1] = q0^(p0>>24)^(p1<<8);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<8);
    ptr += (4096/32);
  }

//layer 3: [2048] s8^4:(1024,4) suggest unit:1
  ptr = poly;
  for(int j=(w/2048);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(1024/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(1024/32)+3];
      uint32_t p2 = ptr[i+(1024/32)+2];
      uint32_t p1 = ptr[i+(1024/32)+1];
      uint32_t p0 = ptr[i+(1024/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>28)^(leftover<<4);
      ptr[i+1+2] = q2^(p2>>28)^(p3<<4);
      ptr[i+1+1] = q1^(p1>>28)^(p2<<4);
      ptr[i+1] = q0^(p0>>28)^(p1<<4);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<4);
    ptr += (2048/32);
  }

//layer 4: [1024] s8^2:(512,2) suggest unit:1
  ptr = poly;
  for(int j=(w/1024);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(512/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(512/32)+3];
      uint32_t p2 = ptr[i+(512/32)+2];
      uint32_t p1 = ptr[i+(512/32)+1];
      uint32_t p0 = ptr[i+(512/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>30)^(leftover<<2);
      ptr[i+1+2] = q2^(p2>>30)^(p3<<2);
      ptr[i+1+1] = q1^(p1>>30)^(p2<<2);
      ptr[i+1] = q0^(p0>>30)^(p1<<2);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<2);
    ptr += (1024/32);
  }

//layer 5: [512] s8^1:(256,1) suggest unit:1
  ptr = poly;
  for(int j=(w/512);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(256/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(256/32)+3];
      uint32_t p2 = ptr[i+(256/32)+2];
      uint32_t p1 = ptr[i+(256/32)+1];
      uint32_t p0 = ptr[i+(256/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>31)^(leftover<<1);
      ptr[i+1+2] = q2^(p2>>31)^(p3<<1);
      ptr[i+1+1] = q1^(p1>>31)^(p2<<1);
      ptr[i+1] = q0^(p0>>31)^(p1<<1);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<1);
    ptr += (512/32);
  }
}


static
void irepr_s8_1_32768( uint32_t *poly )
{
  const unsigned w = 32768;
  uint32_t *ptr;

//layer 5: [512] s8^1:(256,1) suggest unit:1
  ptr = poly;
  for(int j=(w/512);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(256/32);i+=4) {
      uint32_t p3 = ptr[i+(256/32)+3];
      uint32_t p2 = ptr[i+(256/32)+2];
      uint32_t p1 = ptr[i+(256/32)+1];
      uint32_t p0 = ptr[i+(256/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>31)^(p3<<1);
      ptr[i+2] = q2^(p1>>31)^(p2<<1);
      ptr[i+1] = q1^(p0>>31)^(p1<<1);
      ptr[i] = q0^(leftover>>31)^(p0<<1);
      leftover= p3;
    }
    ptr[(256/32)] ^= (leftover>>31);
    ptr += (512/32);
  }

//layer 4: [1024] s8^2:(512,2) suggest unit:1
  ptr = poly;
  for(int j=(w/1024);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(512/32);i+=4) {
      uint32_t p3 = ptr[i+(512/32)+3];
      uint32_t p2 = ptr[i+(512/32)+2];
      uint32_t p1 = ptr[i+(512/32)+1];
      uint32_t p0 = ptr[i+(512/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>30)^(p3<<2);
      ptr[i+2] = q2^(p1>>30)^(p2<<2);
      ptr[i+1] = q1^(p0>>30)^(p1<<2);
      ptr[i] = q0^(leftover>>30)^(p0<<2);
      leftover= p3;
    }
    ptr[(512/32)] ^= (leftover>>30);
    ptr += (1024/32);
  }

//layer 3: [2048] s8^4:(1024,4) suggest unit:1
  ptr = poly;
  for(int j=(w/2048);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(1024/32);i+=4) {
      uint32_t p3 = ptr[i+(1024/32)+3];
      uint32_t p2 = ptr[i+(1024/32)+2];
      uint32_t p1 = ptr[i+(1024/32)+1];
      uint32_t p0 = ptr[i+(1024/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>28)^(p3<<4);
      ptr[i+2] = q2^(p1>>28)^(p2<<4);
      ptr[i+1] = q1^(p0>>28)^(p1<<4);
      ptr[i] = q0^(leftover>>28)^(p0<<4);
      leftover= p3;
    }
    ptr[(1024/32)] ^= (leftover>>28);
    ptr += (2048/32);
  }

//layer 2: [4096] s8^8:(2048,8) suggest unit:1
  ptr = poly;
  for(int j=(w/4096);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(2048/32);i+=4) {
      uint32_t p3 = ptr[i+(2048/32)+3];
      uint32_t p2 = ptr[i+(2048/32)+2];
      uint32_t p1 = ptr[i+(2048/32)+1];
      uint32_t p0 = ptr[i+(2048/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>24)^(p3<<8);
      ptr[i+2] = q2^(p1>>24)^(p2<<8);
      ptr[i+1] = q1^(p0>>24)^(p1<<8);
      ptr[i] = q0^(leftover>>24)^(p0<<8);
      leftover= p3;
    }
    ptr[(2048/32)] ^= (leftover>>24);
    ptr += (4096/32);
  }

//layer 1: [8192] s8^16:(4096,16) suggest unit:1
  ptr = poly;
  for(int j=(w/8192);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(4096/32);i+=4) {
      uint32_t p3 = ptr[i+(4096/32)+3];
      uint32_t p2 = ptr[i+(4096/32)+2];
      uint32_t p1 = ptr[i+(4096/32)+1];
      uint32_t p0 = ptr[i+(4096/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>16)^(p3<<16);
      ptr[i+2] = q2^(p1>>16)^(p2<<16);
      ptr[i+1] = q1^(p0>>16)^(p1<<16);
      ptr[i] = q0^(leftover>>16)^(p0<<16);
      leftover= p3;
    }
    ptr[(4096/32)] ^= (leftover>>16);
    ptr += (8192/32);
  }

//layer 0: [16384] s8^32:(8192,32) suggest unit:1
#define DEG_H 8192
#define DEG_L 32
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 0: [32768] s8^64:(16384,64) suggest unit:1
#define DEG_H 16384
#define DEG_L 64
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

}


//////////////




static
void repr_s8_1_65536( uint32_t *poly )
{
  const unsigned w = 65536;
  uint32_t *ptr;

//layer 0: [65536] s8^128:(32768,128) suggest unit:1
#define DEG_H 32768
#define DEG_L 128
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 0: [32768] s8^64:(16384,64) suggest unit:1
#define DEG_H 16384
#define DEG_L 64
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 0: [16384] s8^32:(8192,32) suggest unit:1
#define DEG_H 8192
#define DEG_L 32
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 1: [8192] s8^16:(4096,16) suggest unit:1
  ptr = poly;
  for(int j=(w/8192);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(4096/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(4096/32)+3];
      uint32_t p2 = ptr[i+(4096/32)+2];
      uint32_t p1 = ptr[i+(4096/32)+1];
      uint32_t p0 = ptr[i+(4096/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>16)^(leftover<<16);
      ptr[i+1+2] = q2^(p2>>16)^(p3<<16);
      ptr[i+1+1] = q1^(p1>>16)^(p2<<16);
      ptr[i+1] = q0^(p0>>16)^(p1<<16);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<16);
    ptr += (8192/32);
  }

//layer 2: [4096] s8^8:(2048,8) suggest unit:1
  ptr = poly;
  for(int j=(w/4096);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(2048/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(2048/32)+3];
      uint32_t p2 = ptr[i+(2048/32)+2];
      uint32_t p1 = ptr[i+(2048/32)+1];
      uint32_t p0 = ptr[i+(2048/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>24)^(leftover<<8);
      ptr[i+1+2] = q2^(p2>>24)^(p3<<8);
      ptr[i+1+1] = q1^(p1>>24)^(p2<<8);
      ptr[i+1] = q0^(p0>>24)^(p1<<8);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<8);
    ptr += (4096/32);
  }

//layer 3: [2048] s8^4:(1024,4) suggest unit:1
  ptr = poly;
  for(int j=(w/2048);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(1024/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(1024/32)+3];
      uint32_t p2 = ptr[i+(1024/32)+2];
      uint32_t p1 = ptr[i+(1024/32)+1];
      uint32_t p0 = ptr[i+(1024/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>28)^(leftover<<4);
      ptr[i+1+2] = q2^(p2>>28)^(p3<<4);
      ptr[i+1+1] = q1^(p1>>28)^(p2<<4);
      ptr[i+1] = q0^(p0>>28)^(p1<<4);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<4);
    ptr += (2048/32);
  }

//layer 4: [1024] s8^2:(512,2) suggest unit:1
  ptr = poly;
  for(int j=(w/1024);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(512/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(512/32)+3];
      uint32_t p2 = ptr[i+(512/32)+2];
      uint32_t p1 = ptr[i+(512/32)+1];
      uint32_t p0 = ptr[i+(512/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>30)^(leftover<<2);
      ptr[i+1+2] = q2^(p2>>30)^(p3<<2);
      ptr[i+1+1] = q1^(p1>>30)^(p2<<2);
      ptr[i+1] = q0^(p0>>30)^(p1<<2);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<2);
    ptr += (1024/32);
  }

//layer 5: [512] s8^1:(256,1) suggest unit:1
  ptr = poly;
  for(int j=(w/512);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=(256/32)-4;i>=0;i-=4) {
      uint32_t p3 = ptr[i+(256/32)+3];
      uint32_t p2 = ptr[i+(256/32)+2];
      uint32_t p1 = ptr[i+(256/32)+1];
      uint32_t p0 = ptr[i+(256/32)];
      uint32_t q3 = ptr[i+1+3];
      uint32_t q2 = ptr[i+1+2];
      uint32_t q1 = ptr[i+1+1];
      uint32_t q0 = ptr[i+1];

      ptr[i+1+3] = q3^(p3>>31)^(leftover<<1);
      ptr[i+1+2] = q2^(p2>>31)^(p3<<1);
      ptr[i+1+1] = q1^(p1>>31)^(p2<<1);
      ptr[i+1] = q0^(p0>>31)^(p1<<1);
      leftover= p0;
    }
    ptr[0] ^= (leftover<<1);
    ptr += (512/32);
  }
}



static
void irepr_s8_1_65536( uint32_t *poly )
{
  const unsigned w = 65536;
  uint32_t *ptr;

//layer 5: [512] s8^1:(256,1) suggest unit:1
  ptr = poly;
  for(int j=(w/512);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(256/32);i+=4) {
      uint32_t p3 = ptr[i+(256/32)+3];
      uint32_t p2 = ptr[i+(256/32)+2];
      uint32_t p1 = ptr[i+(256/32)+1];
      uint32_t p0 = ptr[i+(256/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>31)^(p3<<1);
      ptr[i+2] = q2^(p1>>31)^(p2<<1);
      ptr[i+1] = q1^(p0>>31)^(p1<<1);
      ptr[i] = q0^(leftover>>31)^(p0<<1);
      leftover= p3;
    }
    ptr[(256/32)] ^= (leftover>>31);
    ptr += (512/32);
  }

//layer 4: [1024] s8^2:(512,2) suggest unit:1
  ptr = poly;
  for(int j=(w/1024);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(512/32);i+=4) {
      uint32_t p3 = ptr[i+(512/32)+3];
      uint32_t p2 = ptr[i+(512/32)+2];
      uint32_t p1 = ptr[i+(512/32)+1];
      uint32_t p0 = ptr[i+(512/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>30)^(p3<<2);
      ptr[i+2] = q2^(p1>>30)^(p2<<2);
      ptr[i+1] = q1^(p0>>30)^(p1<<2);
      ptr[i] = q0^(leftover>>30)^(p0<<2);
      leftover= p3;
    }
    ptr[(512/32)] ^= (leftover>>30);
    ptr += (1024/32);
  }

//layer 3: [2048] s8^4:(1024,4) suggest unit:1
  ptr = poly;
  for(int j=(w/2048);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(1024/32);i+=4) {
      uint32_t p3 = ptr[i+(1024/32)+3];
      uint32_t p2 = ptr[i+(1024/32)+2];
      uint32_t p1 = ptr[i+(1024/32)+1];
      uint32_t p0 = ptr[i+(1024/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>28)^(p3<<4);
      ptr[i+2] = q2^(p1>>28)^(p2<<4);
      ptr[i+1] = q1^(p0>>28)^(p1<<4);
      ptr[i] = q0^(leftover>>28)^(p0<<4);
      leftover= p3;
    }
    ptr[(1024/32)] ^= (leftover>>28);
    ptr += (2048/32);
  }

//layer 2: [4096] s8^8:(2048,8) suggest unit:1
  ptr = poly;
  for(int j=(w/4096);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(2048/32);i+=4) {
      uint32_t p3 = ptr[i+(2048/32)+3];
      uint32_t p2 = ptr[i+(2048/32)+2];
      uint32_t p1 = ptr[i+(2048/32)+1];
      uint32_t p0 = ptr[i+(2048/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>24)^(p3<<8);
      ptr[i+2] = q2^(p1>>24)^(p2<<8);
      ptr[i+1] = q1^(p0>>24)^(p1<<8);
      ptr[i] = q0^(leftover>>24)^(p0<<8);
      leftover= p3;
    }
    ptr[(2048/32)] ^= (leftover>>24);
    ptr += (4096/32);
  }

//layer 1: [8192] s8^16:(4096,16) suggest unit:1
  ptr = poly;
  for(int j=(w/8192);j>0;j--) {
    uint32_t leftover = 0;
    for(int i=0;i<(4096/32);i+=4) {
      uint32_t p3 = ptr[i+(4096/32)+3];
      uint32_t p2 = ptr[i+(4096/32)+2];
      uint32_t p1 = ptr[i+(4096/32)+1];
      uint32_t p0 = ptr[i+(4096/32)];
      uint32_t q3 = ptr[i+3];
      uint32_t q2 = ptr[i+2];
      uint32_t q1 = ptr[i+1];
      uint32_t q0 = ptr[i];

      ptr[i+3] = q3^(p2>>16)^(p3<<16);
      ptr[i+2] = q2^(p1>>16)^(p2<<16);
      ptr[i+1] = q1^(p0>>16)^(p1<<16);
      ptr[i] = q0^(leftover>>16)^(p0<<16);
      leftover= p3;
    }
    ptr[(4096/32)] ^= (leftover>>16);
    ptr += (8192/32);
  }

//layer 0: [16384] s8^32:(8192,32) suggest unit:1
#define DEG_H 8192
#define DEG_L 32
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 0: [32768] s8^64:(16384,64) suggest unit:1
#define DEG_H 16384
#define DEG_L 64
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 0: [65536] s8^128:(32768,128) suggest unit:1
#define DEG_H 32768
#define DEG_L 128
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

}






////////////////////////////////////////////////////////////////////////

#include "bc_256.h"


static void bc_1_16384( uint32_t *poly )
{
  repr_s8_1_16384( poly );

  bc_256( poly , 16384/256 );

  bc_1_256( poly , 16384/256 );
}

static void ibc_1_16384( uint32_t *poly )
{
  ibc_1_256( poly , 16384/256 );

  ibc_256( poly , 16384/256 );

  irepr_s8_1_16384( poly );
}



static void bc_1_32768( uint32_t *poly )
{
  repr_s8_1_32768( poly );

  bc_256( poly , 32768/256 );

  bc_1_256( poly , 32768/256 );
}

static void ibc_1_32768( uint32_t *poly )
{
  ibc_1_256( poly , 32768/256 );

  ibc_256( poly , 32768/256 );

  irepr_s8_1_32768( poly );
}



static void bc_1_65536( uint32_t *poly )
{
  repr_s8_1_65536( poly );

  bc_256( poly , 65536/256 );

  bc_1_256( poly , 65536/256 );
}

static void ibc_1_65536( uint32_t *poly )
{
  ibc_1_256( poly , 65536/256 );

  ibc_256( poly , 65536/256);

  irepr_s8_1_65536( poly );
}

//////////////////////////////////////////////



void bc_1( void * poly , unsigned n_byte )
{
  if(2048==n_byte) {
    bc_1_16384( poly );
  } else if(4096==n_byte) {
    bc_1_32768( poly );
  } else if(8192==n_byte) {
    bc_1_65536( poly );
  } else {
    repr_s8_1( poly , n_byte );
    bc_1_256( poly , n_byte>>5 );
    bc_256( poly , n_byte>>5 );
  }
}



void ibc_1( void * poly , unsigned n_byte )
{
  if(2048==n_byte) {
    ibc_1_16384( poly );
  } else if(4096==n_byte) {
    ibc_1_32768( poly );
  } else if(8192==n_byte) {
    ibc_1_65536( poly );
  } else {
    ibc_256( poly , n_byte>>5 );
    ibc_1_256( poly , n_byte>>5 );
    irepr_s8_1( poly , n_byte );
  }
}



