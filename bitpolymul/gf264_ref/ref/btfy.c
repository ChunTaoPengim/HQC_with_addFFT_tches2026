

#include <stdint.h>

#include <string.h>

#include "gf264.h"

#include "cantor_to_gf264.h"

#include "btfy.h"




//////////////////////////////////


static inline
void butterfly_64( uint64_t * poly , unsigned unit ,  uint64_t a )
{
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[i] ^= gf264_mul( poly[unit_2+i] , a );
		poly[unit_2+i] ^= poly[i];
	}
}


static inline
void i_butterfly_64( uint64_t * poly , unsigned unit , uint64_t a )
{
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		poly[i] ^= gf264_mul( poly[unit_2+i] , a );
	}
}

/////////////////////////////////////////////////////////

static inline unsigned min(unsigned a,unsigned b) { return (a<b)?a:b; }

void btfy_64( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
	if( 0 == log_n ) return;

	for(int si=(int)log_n-1; si>=0; si--) {
		unsigned unit = (1<<(si+1));
		unsigned num = (1<<(log_n-(si+1)));  // n_terms / unit

		uint64_t extra_a = cantor_to_gf264(scalar_a>>si);

		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);
		for(unsigned j=0;j<num;j+=step_size) {
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64_t extra_a_j = extra_a ^ cantor_to_gf264(j<<1);
			for(unsigned k=0;k<step_size;k++) {
				uint64_t a = cantor_to_gf264_2x[k]^extra_a_j;
				butterfly_64( poly + (j+k)*unit , unit , a );
			}
		}
	}
}

void ibtfy_64( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
	if( 0 == log_n ) return;

	for(unsigned si=0; si<log_n; si++) {
		unsigned unit = (1<<(si+1));
		unsigned num = (1<<(log_n-(si+1)));  // n_terms / unit

		uint64_t extra_a = cantor_to_gf264(scalar_a>>si);

		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);
		for(unsigned j=0;j<num;j+=step_size) {
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64_t extra_a_j = extra_a ^ cantor_to_gf264(j<<1);
			for(unsigned k=0;k<step_size;k++) {
				uint64_t a = cantor_to_gf264_2x[k]^extra_a_j;
				i_butterfly_64( poly + (j+k)*unit , unit , a );
			}
		}
	}
}
