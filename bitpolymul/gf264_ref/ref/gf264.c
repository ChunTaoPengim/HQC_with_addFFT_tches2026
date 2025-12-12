
#include "gf264.h"

uint64_t gf264_mul( uint64_t a , uint64_t b ) { return _gf264_mul( a, b ); }

void gf264v_mul( uint64_t * c, const uint64_t *a , const uint64_t *b , unsigned len ) { for(unsigned i=0;i<len;i++) c[i] = _gf264_mul( a[i] , b[i] ); }
