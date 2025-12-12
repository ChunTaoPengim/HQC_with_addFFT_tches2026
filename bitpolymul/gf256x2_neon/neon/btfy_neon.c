
#include <stdint.h>
#include <string.h>

#include "gf256.h"
#include "cidx_to_gf256x2.h"
#include "btfy.h"


////////////// ref code for processing smaller cases ////////////////////


static inline
void btfy_unit_gf256x2( uint8_t * v0 , uint8_t * v1 , unsigned unit ,  uint16_t a )
{
	unsigned unit_2= unit/2;
    uint8_t a0 = a&0xff;
    uint8_t a1 = a>>8;
	for(unsigned i=0;i<unit_2;i++) {
        uint16_t vh_i_x_a = gf256x2_mul( v0[unit_2+i] , v1[unit_2+i] , a0 , a1 );
		v0[i] ^= (vh_i_x_a&0xff);
        v1[i] ^= (vh_i_x_a>>8);
		v0[unit_2+i] ^= v0[i];
		v1[unit_2+i] ^= v1[i];
    }
}

static void btfy_gf256x2_ref( uint8_t * v0 , uint8_t * v1 , unsigned n_stage , uint16_t idx_offset )
{
	if( 0 == n_stage ) return;

	for(int stage=n_stage-1; stage>=0; stage--) {
		unsigned unit = (1<<(stage+1));
		unsigned num = (1<<(((int)n_stage)-1-stage));

		for(unsigned j=0;j<num;j++) {
            unsigned idx = idx_offset+j*unit;
			uint16_t idx_gf = cidx_to_gf256x2(idx>>stage);
            btfy_unit_gf256x2( v0 + j*unit , v1 + j*unit , unit , idx_gf  );
		}
	}
}

/////////////////////////////////////////////////////////

static inline
void ibtfy_unit_gf256x2( uint8_t * v0 , uint8_t * v1 , unsigned unit ,  uint16_t a )
{
	unsigned unit_2= unit/2;
    uint8_t a0 = a&0xff;
    uint8_t a1 = a>>8;
	for(unsigned i=0;i<unit_2;i++) {
		v0[unit_2+i] ^= v0[i];
		v1[unit_2+i] ^= v1[i];
        uint16_t vh_i_x_a = gf256x2_mul( v0[unit_2+i] , v1[unit_2+i] , a0 , a1 );
		v0[i] ^= (vh_i_x_a&0xff);
        v1[i] ^= (vh_i_x_a>>8);
    }
}

static void ibtfy_gf256x2_ref( uint8_t * v0 , uint8_t * v1  , unsigned n_stage , uint16_t idx_offset )
{
	if( 0 == n_stage ) return;

	for(int stage=0; stage<(int)n_stage; stage++) {
		unsigned unit = (1<<(stage+1));
		unsigned num = (1<<(((int)n_stage)-1-stage));

		for(unsigned j=0;j<num;j++) {
            unsigned idx = idx_offset+j*unit;
			uint16_t idx_gf = cidx_to_gf256x2(idx>>stage);
            ibtfy_unit_gf256x2( v0 + j*unit , v1 + j*unit , unit , idx_gf  );
		}
	}
}


/////////////////// neon code ///////////////

//#define _SAME_OUTPUT_WITH_REF_

#include "arm_neon.h"
#include "cidx_to_gf256x2_neon.h"
#include "gf256v_neon.h"

static inline
void btfy_unit_gf256x2_neon( uint8_t * v0 , uint8_t * v1 , unsigned unit ,  uint16_t c_idx )
{
	unsigned unit_2= unit/2;
	uint8x16_t cidx0 = vdupq_n_u8( c_idx&0xff );
	uint8x16_t cidx1 = vdupq_n_u8( c_idx>>8 );
	uint8x16_t c0 = cidx_to_gf256x2_l( cidx0 , cidx1 );
	uint8x16_t c1 = cidx_to_gf256x2_h( cidx1 );

	for(unsigned i=0;i<unit_2;i+=16) {
		uint8x16_t a0 = vld1q_u8( (v0+i)  );
		uint8x16_t a1 = vld1q_u8( (v1+i)  );
		uint8x16_t b0 = vld1q_u8( (v0+unit_2+i)  );
		uint8x16_t b1 = vld1q_u8( (v1+unit_2+i)  );

		uint8x16x2_t rr = _gf256x2v_mul( b0, b1, c0 , c1 );
		a0 ^= rr.val[0];
		a1 ^= rr.val[1];
		b0 ^= a0;
		b1 ^= a1;

		vst1q_u8((v0+i), a0);
		vst1q_u8((v1+i), a1);
		vst1q_u8((v0+unit_2+i), b0);
		vst1q_u8((v1+unit_2+i), b1);
    }
}

static inline
void ibtfy_unit_gf256x2_neon( uint8_t * v0 , uint8_t * v1 , unsigned unit ,  uint16_t c_idx )
{
	unsigned unit_2= unit/2;
	uint8x16_t cidx0 = vdupq_n_u8( c_idx&0xff );
	uint8x16_t cidx1 = vdupq_n_u8( c_idx>>8 );
	uint8x16_t c0 = cidx_to_gf256x2_l( cidx0 , cidx1 );
	uint8x16_t c1 = cidx_to_gf256x2_h( cidx1 );

	for(unsigned i=0;i<unit_2;i+=16) {
		uint8x16_t a0 = vld1q_u8( (v0+i)  );
		uint8x16_t a1 = vld1q_u8( (v1+i)  );
		uint8x16_t b0 = vld1q_u8( (v0+unit_2+i)  );
		uint8x16_t b1 = vld1q_u8( (v1+unit_2+i)  );

		b0 ^= a0;
		b1 ^= a1;
		uint8x16x2_t rr = _gf256x2v_mul( b0, b1, c0 , c1 );
		a0 ^= rr.val[0];
		a1 ^= rr.val[1];

		vst1q_u8((v0+i), a0);
		vst1q_u8((v1+i), a1);
		vst1q_u8((v0+unit_2+i), b0);
		vst1q_u8((v1+unit_2+i), b1);
    }
}

static inline
void btfy_unit_gf256_neon_x2( uint8_t * v0 , uint8_t * v1 , unsigned unit ,  uint16_t c_idx )
{
	unsigned unit_2= unit/2;
	uint8x16_t cidx0 = vdupq_n_u8( c_idx );
	uint8x16_t c0 = cidx_to_gf256( cidx0 );

	for(unsigned i=0;i<unit_2;i+=16) {
		uint8x16_t a0 = vld1q_u8( (v0+i)  );
		uint8x16_t b0 = vld1q_u8( (v0+unit_2+i)  );
		a0 ^= _gf256v_mul(b0, c0);
		b0 ^= a0;
		vst1q_u8((v0+i), a0);
		vst1q_u8((v0+unit_2+i), b0);
    }
	for(unsigned i=0;i<unit_2;i+=16) {
		uint8x16_t a0 = vld1q_u8( (v1+i)  );
		uint8x16_t b0 = vld1q_u8( (v1+unit_2+i)  );
		a0 ^= _gf256v_mul(b0, c0);
		b0 ^= a0;
		vst1q_u8((v1+i), a0);
		vst1q_u8((v1+unit_2+i), b0);
    }
}

static inline
void ibtfy_unit_gf256_neon_x2( uint8_t * v0 , uint8_t * v1 , unsigned unit ,  uint16_t c_idx )
{
	unsigned unit_2= unit/2;
	uint8x16_t cidx0 = vdupq_n_u8( c_idx );
	uint8x16_t c0 = cidx_to_gf256( cidx0 );

	for(unsigned i=0;i<unit_2;i+=16) {
		uint8x16_t a0 = vld1q_u8( (v0+i)  );
		uint8x16_t b0 = vld1q_u8( (v0+unit_2+i)  );
		b0 ^= a0;
		a0 ^= _gf256v_mul(b0, c0);
		vst1q_u8((v0+i), a0);
		vst1q_u8((v0+unit_2+i), b0);
    }
	for(unsigned i=0;i<unit_2;i+=16) {
		uint8x16_t a0 = vld1q_u8( (v1+i)  );
		uint8x16_t b0 = vld1q_u8( (v1+unit_2+i)  );
		b0 ^= a0;
		a0 ^= _gf256v_mul(b0, c0);
		vst1q_u8((v1+i), a0);
		vst1q_u8((v1+unit_2+i), b0);
    }
}

static const uint64_t s0consts[2] __attribute__((aligned(16))) = { 0xec50b00ce05cbc00, 0x42fe1ea24ef212ae};
static const uint64_t s1consts[2] __attribute__((aligned(16))) = { 0xe0e05c5cbcbc0000, 0xecec5050b0b00c0c};
static const uint64_t s2consts[2] __attribute__((aligned(16))) = { 0xbcbcbcbc00000000, 0xe0e0e0e05c5c5c5c};
static const uint64_t s3consts[2] __attribute__((aligned(16))) = { 0x0, 0xbcbcbcbcbcbcbcbc};

static inline
void btfy_s43210_gf256x2_neon( uint8_t * v0 , uint8_t * v1 , uint16_t c_idx )
{
	uint8x16_t cidx0, cidx1, c0, c1;
	uint16_t stage_idx;

	uint8x16x2_t rr;

	// input
	uint8x16_t a0 = vld1q_u8( (v0)  );
	uint8x16_t a1 = vld1q_u8( (v1)  );
	uint8x16_t b0 = vld1q_u8( (v0+16)  );
	uint8x16_t b1 = vld1q_u8( (v1+16)  );

	// stage 4
	stage_idx = c_idx >> 4;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 );
	c1 = cidx_to_gf256x2_h( cidx1 );
	rr = _gf256x2v_mul( b0, b1, c0 , c1 );
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];
	b0 ^= a0;
	b1 ^= a1;

	// stage 3
	stage_idx = c_idx >> 3;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s3consts);
	c1 = cidx_to_gf256x2_h( cidx1 );
	cidx0 = vtrn2q_u64(a0,b0);
	cidx1 = vtrn2q_u64(a1,b1);
	a0 = vtrn1q_u64(a0,b0);
	a1 = vtrn1q_u64(a1,b1);
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];
	b0 = cidx0^a0;
	b1 = cidx1^a1;

	// stage 2
	stage_idx = c_idx >> 2;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s2consts);
	c1 = cidx_to_gf256x2_h( cidx1 );
	cidx0 = vtrn2q_u32(a0,b0);
	cidx1 = vtrn2q_u32(a1,b1);
	a0 = vtrn1q_u32(a0,b0);
	a1 = vtrn1q_u32(a1,b1);
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];
	b0 = cidx0^a0;
	b1 = cidx1^a1;

	// stage 1
	stage_idx = c_idx >> 1;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s1consts);
	c1 = cidx_to_gf256x2_h( cidx1 );
	cidx0 = vtrn2q_u16(a0,b0);
	cidx1 = vtrn2q_u16(a1,b1);
	a0 = vtrn1q_u16(a0,b0);
	a1 = vtrn1q_u16(a1,b1);
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];
	b0 = cidx0^a0;
	b1 = cidx1^a1;

	// stage 0
	stage_idx = c_idx;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s0consts);
	c1 = cidx_to_gf256x2_h( cidx1 );
	cidx0 = vtrn2q_u8(a0,b0);
	cidx1 = vtrn2q_u8(a1,b1);
	a0 = vtrn1q_u8(a0,b0);
	a1 = vtrn1q_u8(a1,b1);
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];
	b0 = cidx0^a0;
	b1 = cidx1^a1;

	// output
	vst1q_u8((v0), a0);
	vst1q_u8((v1), a1);
	vst1q_u8((v0+16), b0);
	vst1q_u8((v1+16), b1);

#if defined(_SAME_OUTPUT_WITH_REF_)
	// make compatible data layout with ref code
	uint8_t t0[32];
	memcpy( t0 , v0 , 32);
	for(int i=0;i<16;i++) {
		v0[2*i] = t0[i];
		v0[2*i+1] = t0[16+i];
	}
	memcpy( t0 , v1 , 32);
	for(int i=0;i<16;i++) {
		v1[2*i] = t0[i];
		v1[2*i+1] = t0[16+i];
	}
#endif
}

static inline
void ibtfy_s01234_gf256x2_neon( uint8_t * v0 , uint8_t * v1 , uint16_t c_idx )
{
	uint8x16_t cidx0, cidx1, c0, c1;
	uint16_t stage_idx;

	uint8x16x2_t rr;

#if defined(_SAME_OUTPUT_WITH_REF_)
	// make compatible data layout with ref code
	uint8_t t0[32];
	memcpy( t0 , v0 , 32);
	for(int i=0;i<16;i++) {
		v0[i] = t0[2*i];
		v0[16+i] = t0[2*i+1];
	}
	memcpy( t0 , v1 , 32);
	for(int i=0;i<16;i++) {
		v1[i] = t0[2*i];
		v1[16+i] = t0[2*i+1];
	}
#endif

	// input
	uint8x16_t a0 = vld1q_u8( (v0)  );
	uint8x16_t a1 = vld1q_u8( (v1)  );
	uint8x16_t b0 = vld1q_u8( (v0+16)  );
	uint8x16_t b1 = vld1q_u8( (v1+16)  );

	// stage 0
	stage_idx = c_idx;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s0consts);
	c1 = cidx_to_gf256x2_h( cidx1 );

	cidx0 = b0^a0;
	cidx1 = b1^a1;
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];

	b0 = vtrn2q_u8(a0,cidx0);
	b1 = vtrn2q_u8(a1,cidx1);
	a0 = vtrn1q_u8(a0,cidx0);
	a1 = vtrn1q_u8(a1,cidx1);

	// stage 1
	stage_idx = c_idx >> 1;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s1consts);
	c1 = cidx_to_gf256x2_h( cidx1 );

	cidx0 = b0^a0;
	cidx1 = b1^a1;
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];

	b0 = vtrn2q_u16(a0,cidx0);
	b1 = vtrn2q_u16(a1,cidx1);
	a0 = vtrn1q_u16(a0,cidx0);
	a1 = vtrn1q_u16(a1,cidx1);

	// stage 2
	stage_idx = c_idx >> 2;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s2consts);
	c1 = cidx_to_gf256x2_h( cidx1 );

	cidx0 = b0^a0;
	cidx1 = b1^a1;
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];

	b0 = vtrn2q_u32(a0,cidx0);
	b1 = vtrn2q_u32(a1,cidx1);
	a0 = vtrn1q_u32(a0,cidx0);
	a1 = vtrn1q_u32(a1,cidx1);

	// stage 3
	stage_idx = c_idx >> 3;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 ) ^ vld1q_u8((uint8_t*)s3consts);
	c1 = cidx_to_gf256x2_h( cidx1 );

	cidx0 = b0^a0;
	cidx1 = b1^a1;
	rr = _gf256x2v_mul(cidx0,cidx1, c0, c1);
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];

	b0 = vtrn2q_u64(a0,cidx0);
	b1 = vtrn2q_u64(a1,cidx1);
	a0 = vtrn1q_u64(a0,cidx0);
	a1 = vtrn1q_u64(a1,cidx1);

	// stage 4
	stage_idx = c_idx >> 4;
	cidx0 = vdupq_n_u8( stage_idx&0xff );
	cidx1 = vdupq_n_u8( stage_idx>>8 );
	c0 = cidx_to_gf256x2_l( cidx0 , cidx1 );
	c1 = cidx_to_gf256x2_h( cidx1 );
	b0 ^= a0;
	b1 ^= a1;
	rr = _gf256x2v_mul( b0, b1, c0 , c1 );
	a0 ^= rr.val[0];
	a1 ^= rr.val[1];

	// output
	vst1q_u8((v0), a0);
	vst1q_u8((v1), a1);
	vst1q_u8((v0+16), b0);
	vst1q_u8((v1+16), b1);

}

/////////////////////////////////////

void btfy_gf256x2( uint8_t * v0 , uint8_t * v1 , unsigned n_stage , uint16_t idx_offset )
{
	if( 4 >= n_stage ) { btfy_gf256x2_ref(v0,v1,n_stage,idx_offset); return; }
	unsigned len = 1<<n_stage;
	for(int stage=n_stage-1; stage>4; stage--) {
		unsigned unit = (1<<(stage+1));
		unsigned num = (1<<(((int)n_stage)-1-stage));

		for(unsigned j=0;j<num;j++) {
            unsigned idx = (idx_offset+j*unit)>>stage;
			if (idx < 256) { btfy_unit_gf256_neon_x2( v0 + j*unit , v1 + j*unit , unit , idx ); } 
			else {           btfy_unit_gf256x2_neon ( v0 + j*unit , v1 + j*unit , unit , idx ); }
		}
	}
	// stage = 4
	for( unsigned i=0;i<len;i+=32) {
		/// XXX: optimizing multiplying by gf256 elements
		btfy_s43210_gf256x2_neon(v0+i,v1+i,idx_offset+i); // s4,s3,s2,s1,s0
		//btfy_gf256x2_ref( v0+i , v1+i , 5 , idx_offset+i );
	}
}



void ibtfy_gf256x2( uint8_t * v0 , uint8_t * v1  , unsigned n_stage , uint16_t idx_offset )
{
	if( 4 >= n_stage ) { ibtfy_gf256x2_ref(v0,v1,n_stage,idx_offset); return; }

	unsigned len = 1<<n_stage;
	// stage = 0 - 4
	for( unsigned i=0;i<len;i+=32) {
		/// XXX: optimizing multiplying by gf256 elements
		ibtfy_s01234_gf256x2_neon(v0+i,v1+i,idx_offset+i); // s0,s1,s2,s3,s4
		//ibtfy_gf256x2_ref( v0+i , v1+i , 5 , idx_offset+i );
	}
	// stage = 5
	for(int stage=5; stage<(int)n_stage; stage++) {
		unsigned unit = (1<<(stage+1));
		unsigned num = (1<<(((int)n_stage)-1-stage));

		for(unsigned j=0;j<num;j++) {
			unsigned idx = (idx_offset+j*unit)>>stage;
			if (idx < 256) { ibtfy_unit_gf256_neon_x2( v0 + j*unit , v1 + j*unit , unit , idx ); }
			else {           ibtfy_unit_gf256x2_neon ( v0 + j*unit , v1 + j*unit , unit , idx ); }
		}
	}
}

