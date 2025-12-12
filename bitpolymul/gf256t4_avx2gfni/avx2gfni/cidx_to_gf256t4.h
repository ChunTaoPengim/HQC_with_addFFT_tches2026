#ifndef _CANTOR_TO_GF256T4_H_
#define _CANTOR_TO_GF256T4_H_

#include <stdint.h>

#ifdef  __cplusplus
extern  "C" {
#endif

static const uint32_t _cidx_to_gf256t4[32] __attribute__((aligned(32))) = {
  0x1,       0xbc,       0x5c,       0xc,       0xae,       0x5a,       0xe,       0x84,
  0x1f6,     0xbc5c,     0x5c18,     0xc7e,     0xae46,     0x5a68,     0xe02,     0x8456,
  0x1f6fa,   0xbc5cc4,   0x5c181e,   0xc7e5a,   0xae46a2,   0x5a686a,   0xe02fc,   0x8456e2,
  0x1f6daec, 0xbc5caed6, 0x5c186b7c, 0xc7ec17a, 0xae46be6c, 0x5a68df6a, 0xe02275c, 0x8456c9c2
};

static inline
uint32_t cidx_to_gf256t4( uint32_t cidx )
{
  uint32_t r = 0;
  while( cidx ) {
    r ^= _cidx_to_gf256t4[ __builtin_ctz(cidx) ];
    cidx &= (cidx-1);
  }
  return r;
}

#ifdef  __cplusplus
}
#endif

#endif  // #ifndef _CANTOR_TO_GF256T4_H_

