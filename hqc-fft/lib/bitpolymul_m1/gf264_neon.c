#include "gf264.h"
#include "gf264_neon.h"

uint64_t gf264_mul( uint64_t a , uint64_t b ) { return _gf264_mulx1_neon( a, b ); }

void gf264v_mul( uint64_t * c, const uint64_t *a , const uint64_t *b , unsigned len )
{
    uint64x2_t mask_0x1b = vdupq_n_u64(0x1b);
    while( len > 1 ) {
        uint64x2_t aa = vld1q_u64( a );
        a += 2;
        uint64x2_t bb = vld1q_u64( b );
        b += 2;
        uint64x2_t cc = _gf264_mul_neon( aa , bb , mask_0x1b);
        len -= 2;
        vst1q_u64( c , cc );
        c += 2;
    }
    if( len ) c[0] = _gf264_mulx1_neon( a[0] , b[0] );
}
