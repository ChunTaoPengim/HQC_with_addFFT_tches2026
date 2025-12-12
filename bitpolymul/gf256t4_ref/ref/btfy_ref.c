
#include <stdint.h>
#include <string.h>

#include "gf256.h"
#include "cidx_to_gf256t4.h"
#include "btfy.h"


//////////////////////////////////


static inline
void btfy_unit_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned unit ,  uint32_t a )
{
	unsigned unit_2= unit/2;
    uint8_t a0 = a&0xff;
    uint8_t a1 = (a>>8)&0xff;
    uint8_t a2 = (a>>16)&0xff;
    uint8_t a3 = (a>>24);
	for(unsigned i=0;i<unit_2;i++) {
        uint32_t tmp = gf256t4_mul( v0[unit_2+i] , v1[unit_2+i] , v2[unit_2+i] , v3[unit_2+i] , a0 , a1 , a2 , a3 );
		v0[i] ^= (tmp&0xff);
        v1[i] ^= (tmp>>8)&0xff;
        v2[i] ^= (tmp>>16)&0xff;
        v3[i] ^= (tmp>>24);
		v0[unit_2+i] ^= v0[i];
		v1[unit_2+i] ^= v1[i];
		v2[unit_2+i] ^= v2[i];
		v3[unit_2+i] ^= v3[i];
	}
}

void btfy_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
{
	if( 0 == n_stage ) return;

	for(int stage=n_stage-1; stage>=0; stage--) {
		unsigned unit = (1<<(stage+1));
		unsigned num = (1<<(((int)n_stage)-1-stage));

		for(unsigned j=0;j<num;j++) {
            uint32_t idx = idx_offset+j*unit;
			uint32_t idx_gf = cidx_to_gf256t4(idx>>stage);
            btfy_unit_gf256t4( v0 + j*unit , v1 + j*unit , v2 + j*unit , v3 + j*unit , unit , idx_gf  );
		}
	}
}

/////////////////////////////////////////////////////////

static inline
void ibtfy_unit_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned unit ,  uint32_t a )
{
	unsigned unit_2= unit/2;
    uint8_t a0 = a&0xff;
    uint8_t a1 = (a>>8)&0xff;
    uint8_t a2 = (a>>16)&0xff;
    uint8_t a3 = (a>>24);
	for(unsigned i=0;i<unit_2;i++) {
		v0[unit_2+i] ^= v0[i];
		v1[unit_2+i] ^= v1[i];
		v2[unit_2+i] ^= v2[i];
		v3[unit_2+i] ^= v3[i];
        uint32_t tmp = gf256t4_mul( v0[unit_2+i] , v1[unit_2+i] , v2[unit_2+i] , v3[unit_2+i] , a0 , a1 , a2 , a3 );
		v0[i] ^= (tmp&0xff);
        v1[i] ^= (tmp>>8)&0xff;
        v2[i] ^= (tmp>>16)&0xff;
        v3[i] ^= (tmp>>24);
    }
}

void ibtfy_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
{
	if( 0 == n_stage ) return;

	for(int stage=0; stage<(int)n_stage; stage++) {
		unsigned unit = (1<<(stage+1));
		unsigned num = (1<<(((int)n_stage)-1-stage));

		for(unsigned j=0;j<num;j++) {
            uint32_t idx = idx_offset+j*unit;
			uint32_t idx_gf = cidx_to_gf256t4(idx>>stage);
            ibtfy_unit_gf256t4( v0 + j*unit , v1 + j*unit , v2 + j*unit , v3 + j*unit , unit , idx_gf  );
		}
	}
}
