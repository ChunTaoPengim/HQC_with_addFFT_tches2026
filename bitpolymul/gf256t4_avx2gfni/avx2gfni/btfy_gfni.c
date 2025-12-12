
#include <stdint.h>
#include <string.h>
#include <immintrin.h>


#include "gf256.h"
#include "btfy.h"
#include "cidx_to_gf256x2_gfni.h"

////////////// ref code for processing smaller cases ////////////////////

#include "cidx_to_gf256t4.h"

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

static
void btfy_gf256t4_ref( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
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

static
void ibtfy_gf256t4_ref( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
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

//////////////////////////////////

#include "gf256v_gfni.h"

//#define _SAME_OUTPUT_WITH_REF_

#define _s0_ymm _mm256_set_epi64x(0x18a444f814a848f4ULL,0xb60aea56ba06e65aULL,0x42fe1ea24ef212aeULL,0xec50b00ce05cbc00ULL)
#define _s1_ymm _mm256_set_epi64x(0x4242fefe1e1ea2a2ULL,0x4e4ef2f21212aeaeULL,0xecec5050b0b00c0cULL,0xe0e05c5cbcbc0000ULL)
#define _s2_ymm _mm256_set_epi64x(0xecececec50505050ULL,0xb0b0b0b00c0c0c0cULL,0xe0e0e0e05c5c5c5cULL,0xbcbcbcbc00000000ULL)
#define _s3_ymm _mm256_set_epi64x(0xe0e0e0e0e0e0e0e0ULL,0x5c5c5c5c5c5c5c5cULL,0xbcbcbcbcbcbcbcbcULL,0x0ULL)
#define _s4_ymm _mm256_set_epi64x(0xbcbcbcbcbcbcbcbcULL,0xbcbcbcbcbcbcbcbcULL,0x0ULL,0x0ULL)

static inline
void btfy_s543210_64ele( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s5_idx , uint32_t s4_idx , uint32_t s3_idx , uint32_t s2_idx , uint32_t s1_idx , uint32_t s0_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s5_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 4
	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s4_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 3
	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s3_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 2
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 1
	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 0
	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);

#if defined(_SAME_OUTPUT_WITH_REF_)
	// make compatible data layout with ref code
	uint8_t t0[64] __attribute__((aligned(32)));
	memcpy( t0 , v0 , 64);
	for(int i=0;i<32;i++) {
		v0[2*i] = t0[i];
		v0[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v1 , 64);
	for(int i=0;i<32;i++) {
		v1[2*i] = t0[i];
		v1[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v2 , 64);
	for(int i=0;i<32;i++) {
		v2[2*i] = t0[i];
		v2[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v3 , 64);
	for(int i=0;i<32;i++) {
		v3[2*i] = t0[i];
		v3[2*i+1] = t0[32+i];
	}
#endif
}

static inline
void btfy_s543210_64ele_5( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s5_idx , uint32_t s4_idx , uint32_t s3_idx , uint32_t s2_idx , uint32_t s1_idx , uint32_t s0_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );

	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 4
	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s4_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 3
	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s3_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 2
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 1
	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 0
	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);

#if defined(_SAME_OUTPUT_WITH_REF_)
	// make compatible data layout with ref code
	uint8_t t0[64] __attribute__((aligned(32)));
	memcpy( t0 , v0 , 64);
	for(int i=0;i<32;i++) {
		v0[2*i] = t0[i];
		v0[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v1 , 64);
	for(int i=0;i<32;i++) {
		v1[2*i] = t0[i];
		v1[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v2 , 64);
	for(int i=0;i<32;i++) {
		v2[2*i] = t0[i];
		v2[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v3 , 64);
	for(int i=0;i<32;i++) {
		v3[2*i] = t0[i];
		v3[2*i+1] = t0[32+i];
	}
#endif
}

static inline
void btfy_s543210_64ele_4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s5_idx , uint32_t s4_idx , uint32_t s3_idx , uint32_t s2_idx , uint32_t s1_idx , uint32_t s0_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );

	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 4
	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );

	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 3
	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s3_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 2
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 1
	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 0
	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);

#if defined(_SAME_OUTPUT_WITH_REF_)
	// make compatible data layout with ref code
	uint8_t t0[64] __attribute__((aligned(32)));
	memcpy( t0 , v0 , 64);
	for(int i=0;i<32;i++) {
		v0[2*i] = t0[i];
		v0[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v1 , 64);
	for(int i=0;i<32;i++) {
		v1[2*i] = t0[i];
		v1[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v2 , 64);
	for(int i=0;i<32;i++) {
		v2[2*i] = t0[i];
		v2[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v3 , 64);
	for(int i=0;i<32;i++) {
		v3[2*i] = t0[i];
		v3[2*i+1] = t0[32+i];
	}
#endif
}

static inline
void btfy_s543210_64ele_3( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s5_idx , uint32_t s4_idx , uint32_t s3_idx , uint32_t s2_idx , uint32_t s1_idx , uint32_t s0_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );

	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 4
	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );

	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 3
	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );

	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 2
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 1
	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// stage 0
	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];
	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);

#if defined(_SAME_OUTPUT_WITH_REF_)
	// make compatible data layout with ref code
	uint8_t t0[64] __attribute__((aligned(32)));
	memcpy( t0 , v0 , 64);
	for(int i=0;i<32;i++) {
		v0[2*i] = t0[i];
		v0[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v1 , 64);
	for(int i=0;i<32;i++) {
		v1[2*i] = t0[i];
		v1[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v2 , 64);
	for(int i=0;i<32;i++) {
		v2[2*i] = t0[i];
		v2[2*i+1] = t0[32+i];
	}
	memcpy( t0 , v3 , 64);
	for(int i=0;i<32;i++) {
		v3[2*i] = t0[i];
		v3[2*i+1] = t0[32+i];
	}
#endif
}

///////////////////////

static inline
void ibtfy_s012345_64ele( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s0_idx , uint32_t s1_idx , uint32_t s2_idx , uint32_t s3_idx , uint32_t s4_idx , uint32_t s5_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 0
	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 1
	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 2
	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 3
	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s3_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 4
	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s4_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s5_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);
}

static inline
void ibtfy_s012345_64ele_5( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s0_idx , uint32_t s1_idx , uint32_t s2_idx , uint32_t s3_idx , uint32_t s4_idx , uint32_t s5_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 0
	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 1
	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 2
	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 3
	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s3_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 4
	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s4_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);
}

static inline
void ibtfy_s012345_64ele_4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s0_idx , uint32_t s1_idx , uint32_t s2_idx , uint32_t s3_idx , uint32_t s4_idx , uint32_t s5_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 0
	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 1
	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 2
	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 3
	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s3_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 4
	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);
}

static inline
void ibtfy_s012345_64ele_3( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 ,
	uint32_t s0_idx , uint32_t s1_idx , uint32_t s2_idx , uint32_t s3_idx , uint32_t s4_idx , uint32_t s5_idx )
{
	__m256i _0x20 = _mm256_set1_epi8( 0x20 );
	ymm_x4 r;
	__m256i c0, c1 , c2, c3;

	// input
	__m256i v0l = _mm256_loadu_si256( (__m256i*)(v0)  );
	__m256i v1l = _mm256_loadu_si256( (__m256i*)(v1)  );
	__m256i v2l = _mm256_loadu_si256( (__m256i*)(v2)  );
	__m256i v3l = _mm256_loadu_si256( (__m256i*)(v3)  );
	__m256i v0h = _mm256_loadu_si256( (__m256i*)(v0+32)  );
	__m256i v1h = _mm256_loadu_si256( (__m256i*)(v1+32)  );
	__m256i v2h = _mm256_loadu_si256( (__m256i*)(v2+32)  );
	__m256i v3h = _mm256_loadu_si256( (__m256i*)(v3+32)  );

	// stage 0
	c0 = _mm256_set1_epi8( s0_idx&0xff ) ^ _s0_ymm;
	c1 = _mm256_set1_epi8( (s0_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s0_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s0_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	__m256i _0xff00 = _mm256_set1_epi16(0xff00);
	r.val[0] = _mm256_blendv_epi8( v0l , _mm256_slli_epi16(v0h,8) , _0xff00 );
	r.val[1] = _mm256_blendv_epi8( v1l , _mm256_slli_epi16(v1h,8) , _0xff00 );
	r.val[2] = _mm256_blendv_epi8( v2l , _mm256_slli_epi16(v2h,8) , _0xff00 );
	r.val[3] = _mm256_blendv_epi8( v3l , _mm256_slli_epi16(v3h,8) , _0xff00 );
	v0h = _mm256_blendv_epi8( _mm256_srli_epi16(v0l,8) , v0h , _0xff00 );
	v1h = _mm256_blendv_epi8( _mm256_srli_epi16(v1l,8) , v1h , _0xff00 );
	v2h = _mm256_blendv_epi8( _mm256_srli_epi16(v2l,8) , v2h , _0xff00 );
	v3h = _mm256_blendv_epi8( _mm256_srli_epi16(v3l,8) , v3h , _0xff00 );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 1
	c0 = _mm256_set1_epi8( s1_idx&0xff ) ^ _s1_ymm;
	c1 = _mm256_set1_epi8( (s1_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s1_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s1_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi16( v0l , _mm256_slli_epi32(v0h,16) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi16( v1l , _mm256_slli_epi32(v1h,16) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi16( v2l , _mm256_slli_epi32(v2h,16) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi16( v3l , _mm256_slli_epi32(v3h,16) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi16( _mm256_srli_epi32(v0l,16) , v0h , 0xaa );
	v1h = _mm256_blend_epi16( _mm256_srli_epi32(v1l,16) , v1h , 0xaa );
	v2h = _mm256_blend_epi16( _mm256_srli_epi32(v2l,16) , v2h , 0xaa );
	v3h = _mm256_blend_epi16( _mm256_srli_epi32(v3l,16) , v3h , 0xaa );
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 2
	c0 = _mm256_set1_epi8( s2_idx&0xff ) ^ _s2_ymm;
	c1 = _mm256_set1_epi8( (s2_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s2_idx>>16)&0xff );
	c3 = _mm256_set1_epi8( (s2_idx>>24) );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul( v0h, v1h, v2h, v3h, c0 , c1 , c2, c3 , _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_epi64(v0h,32) , 0xaa ); // 1010,1010
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_epi64(v1h,32) , 0xaa ); // 1010,1010
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_epi64(v2h,32) , 0xaa ); // 1010,1010
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_epi64(v3h,32) , 0xaa ); // 1010,1010
	v0h = _mm256_blend_epi32( _mm256_srli_epi64(v0l,32) , v0h , 0xaa );  // 1010,1010
	v1h = _mm256_blend_epi32( _mm256_srli_epi64(v1l,32) , v1h , 0xaa );  // 1010,1010
	v2h = _mm256_blend_epi32( _mm256_srli_epi64(v2l,32) , v2h , 0xaa );  // 1010,1010
	v3h = _mm256_blend_epi32( _mm256_srli_epi64(v3l,32) , v3h , 0xaa );  // 1010,1010
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 3
	c0 = _mm256_set1_epi8( s3_idx&0xff ) ^ _s3_ymm;
	c1 = _mm256_set1_epi8( (s3_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s3_idx>>16)&0xff );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// 64-bit transpose. use blend_epi32
	r.val[0] = _mm256_blend_epi32( v0l , _mm256_slli_si256(v0h,8) , 0xcc ); // 1100,1100
	r.val[1] = _mm256_blend_epi32( v1l , _mm256_slli_si256(v1h,8) , 0xcc ); // 1100,1100
	r.val[2] = _mm256_blend_epi32( v2l , _mm256_slli_si256(v2h,8) , 0xcc ); // 1100,1100
	r.val[3] = _mm256_blend_epi32( v3l , _mm256_slli_si256(v3h,8) , 0xcc ); // 1100,1100
	v0h = _mm256_blend_epi32( _mm256_srli_si256(v0l,8) , v0h , 0xcc ); // 1100,1100
	v1h = _mm256_blend_epi32( _mm256_srli_si256(v1l,8) , v1h , 0xcc ); // 1100,1100
	v2h = _mm256_blend_epi32( _mm256_srli_si256(v2l,8) , v2h , 0xcc ); // 1100,1100
	v3h = _mm256_blend_epi32( _mm256_srli_si256(v3l,8) , v3h , 0xcc ); // 1100,1100
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 4
	c0 = _mm256_set1_epi8( s4_idx&0xff ) ^ _s4_ymm;
	c1 = _mm256_set1_epi8( (s4_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s4_idx>>16)&0xff );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	r.val[0] = _mm256_permute2x128_si256(v0l,v0h,0x20);
	r.val[1] = _mm256_permute2x128_si256(v1l,v1h,0x20);
	r.val[2] = _mm256_permute2x128_si256(v2l,v2h,0x20);
	r.val[3] = _mm256_permute2x128_si256(v3l,v3h,0x20);
	v0h = _mm256_permute2x128_si256(v0l,v0h,0x31);
	v1h = _mm256_permute2x128_si256(v1l,v1h,0x31);
	v2h = _mm256_permute2x128_si256(v2l,v2h,0x31);
	v3h = _mm256_permute2x128_si256(v3l,v3h,0x31);
	v0l = r.val[0];
	v1l = r.val[1];
	v2l = r.val[2];
	v3l = r.val[3];

	// stage 5
	c0 = _mm256_set1_epi8( s5_idx&0xff );
	c1 = _mm256_set1_epi8( (s5_idx>>8)&0xff );
	c2 = _mm256_set1_epi8( (s5_idx>>16)&0xff );

	v0h ^= v0l;
	v1h ^= v1l;
	v2h ^= v2l;
	v3h ^= v3l;
	r = _gf256t4v_mul_gf256x3( v0h, v1h, v2h, v3h, c0 , c1 , c2, _0x20 );
	v0l ^= r.val[0];
	v1l ^= r.val[1];
	v2l ^= r.val[2];
	v3l ^= r.val[3];

	// output
	_mm256_storeu_si256((__m256i*)(v0), v0l);
	_mm256_storeu_si256((__m256i*)(v1), v1l);
	_mm256_storeu_si256((__m256i*)(v2), v2l);
	_mm256_storeu_si256((__m256i*)(v3), v3l);
	_mm256_storeu_si256((__m256i*)(v0+32), v0h);
	_mm256_storeu_si256((__m256i*)(v1+32), v1h);
	_mm256_storeu_si256((__m256i*)(v2+32), v2h);
	_mm256_storeu_si256((__m256i*)(v3+32), v3h);
}

////////////////////////////////////////////

static
void btfy_s543210_gf256t4_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned len , uint32_t idx_offset )
{
	uint32_t s0_offset = cidx_to_gf256t4( idx_offset );
	uint32_t s1_offset = cidx_to_gf256t4( idx_offset>>1 );
	uint32_t s2_offset = cidx_to_gf256t4( idx_offset>>2 );
	uint32_t s3_offset = cidx_to_gf256t4( idx_offset>>3 );
	uint32_t s4_offset = cidx_to_gf256t4( idx_offset>>4 );
	uint32_t s5_offset = cidx_to_gf256t4( idx_offset>>5 );

	if(s5_offset >= (1<<24)) {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];	
			btfy_s543210_64ele( v0 , v1 , v2 , v3 , s5_idx , s4_idx , s3_idx , s2_idx , s1_idx , s0_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	} else if (s4_offset >= (1<<24)) {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];	
			btfy_s543210_64ele_5( v0 , v1 , v2 , v3 , s5_idx , s4_idx , s3_idx , s2_idx , s1_idx , s0_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	} else if (s3_offset >= (1<<24)) {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];	
			btfy_s543210_64ele_4( v0 , v1 , v2 , v3 , s5_idx , s4_idx , s3_idx , s2_idx , s1_idx , s0_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	} else {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];	
			btfy_s543210_64ele_3( v0 , v1 , v2 , v3 , s5_idx , s4_idx , s3_idx , s2_idx , s1_idx , s0_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	}

}

static
void ibtfy_s012345_gf256t4_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned len , uint32_t idx_offset )
{
	uint32_t s0_offset = cidx_to_gf256t4( idx_offset );
	uint32_t s1_offset = cidx_to_gf256t4( idx_offset>>1 );
	uint32_t s2_offset = cidx_to_gf256t4( idx_offset>>2 );
	uint32_t s3_offset = cidx_to_gf256t4( idx_offset>>3 );
	uint32_t s4_offset = cidx_to_gf256t4( idx_offset>>4 );
	uint32_t s5_offset = cidx_to_gf256t4( idx_offset>>5 );

	if(s5_offset >= (1<<24)) {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];

			ibtfy_s012345_64ele( v0 , v1 , v2 , v3 , s0_idx , s1_idx , s2_idx , s3_idx , s4_idx , s5_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	} else if(s4_offset >= (1<<24)) {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];

			ibtfy_s012345_64ele_5( v0 , v1 , v2 , v3 , s0_idx , s1_idx , s2_idx , s3_idx , s4_idx , s5_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	} else if(s3_offset >= (1<<24)) {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];

			ibtfy_s012345_64ele_4( v0 , v1 , v2 , v3 , s0_idx , s1_idx , s2_idx , s3_idx , s4_idx , s5_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	}else {
		for(unsigned i=0;i<len;i+=64) {
			unsigned idx = i>>6; // div 64
			uint32_t s0_idx = s0_offset ^ cidx_to_gf256x2_64x[idx];
			uint32_t s1_idx = s1_offset ^ cidx_to_gf256x2_32x[idx];
			uint32_t s2_idx = s2_offset ^ cidx_to_gf256x2_16x[idx];
			uint32_t s3_idx = s3_offset ^ cidx_to_gf256x2_8x[idx];
			uint32_t s4_idx = s4_offset ^ cidx_to_gf256x2_4x[idx];
			uint32_t s5_idx = s5_offset ^ cidx_to_gf256x2_2x[idx];

			ibtfy_s012345_64ele_3( v0 , v1 , v2 , v3 , s0_idx , s1_idx , s2_idx , s3_idx , s4_idx , s5_idx );
			v0 += 64;
			v1 += 64;
			v2 += 64;
			v3 += 64;
		}
	}
}

///////////////////////////////////////////////////

static inline
void btfy_si_gf256t4_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned si , unsigned len , uint32_t offset_gf )
{
	unsigned unit = 1<<(si+1);
	unsigned n_unit = len/unit;
	unsigned unit_2 = unit/2;

	__m256i _0x20 = _mm256_set1_epi8(0x20);
	__m256i _c0 = _mm256_set1_epi8(offset_gf&0xff);
	__m256i _c1 = _mm256_set1_epi8((offset_gf>>8)&0xff);
	__m256i _c2 = _mm256_set1_epi8((offset_gf>>16)&0xff);
	__m256i _c3 = _mm256_set1_epi8((offset_gf>>24)&0xff);

	unsigned idx_unit = 0;
	for(;idx_unit<n_unit;idx_unit++) {
		uint8_t c_idx = cidx_to_gf256x2_2x[idx_unit];
		__m256i c0 = _mm256_set1_epi8( c_idx )^_c0;

		for(unsigned i=0;i<unit_2;i+=32) {
			__m256i v0il = _mm256_loadu_si256( (__m256i*)(v0+i)  );
			__m256i v1il = _mm256_loadu_si256( (__m256i*)(v1+i)  );
			__m256i v2il = _mm256_loadu_si256( (__m256i*)(v2+i)  );
			__m256i v3il = _mm256_loadu_si256( (__m256i*)(v3+i)  );
			__m256i v0ih = _mm256_loadu_si256( (__m256i*)(v0+unit_2+i)  );
			__m256i v1ih = _mm256_loadu_si256( (__m256i*)(v1+unit_2+i)  );
			__m256i v2ih = _mm256_loadu_si256( (__m256i*)(v2+unit_2+i)  );
			__m256i v3ih = _mm256_loadu_si256( (__m256i*)(v3+unit_2+i)  );

			ymm_x4 t0 = _gf256t4v_mul( v0ih , v1ih , v2ih , v3ih , c0 , _c1 , _c2 , _c3 , _0x20 );

			v0il ^= t0.val[0];
			v1il ^= t0.val[1];
			v2il ^= t0.val[2];
			v3il ^= t0.val[3];
			v0ih ^= v0il;
			v1ih ^= v1il;
			v2ih ^= v2il;
			v3ih ^= v3il;

			_mm256_storeu_si256((__m256i*)(v0+i), v0il);
			_mm256_storeu_si256((__m256i*)(v1+i), v1il);
			_mm256_storeu_si256((__m256i*)(v2+i), v2il);
			_mm256_storeu_si256((__m256i*)(v3+i), v3il);
			_mm256_storeu_si256((__m256i*)(v0+unit_2+i), v0ih);
			_mm256_storeu_si256((__m256i*)(v1+unit_2+i), v1ih);
			_mm256_storeu_si256((__m256i*)(v2+unit_2+i), v2ih);
			_mm256_storeu_si256((__m256i*)(v3+unit_2+i), v3ih);
		}
		v0 += unit;
		v1 += unit;
		v2 += unit;
		v3 += unit;
	}
}

static inline
void btfy_si_gf256x3_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned si , unsigned len , uint32_t offset_gf )
{
	unsigned unit = 1<<(si+1);
	unsigned n_unit = len/unit;
	unsigned unit_2 = unit/2;

	__m256i _0x20 = _mm256_set1_epi8(0x20);
	__m256i _c0 = _mm256_set1_epi8(offset_gf&0xff);
	__m256i _c1 = _mm256_set1_epi8((offset_gf>>8)&0xff);
	__m256i _c2 = _mm256_set1_epi8((offset_gf>>16)&0xff);

	unsigned idx_unit = 0;
	for(;idx_unit<n_unit;idx_unit++) {
		uint8_t c_idx = cidx_to_gf256x2_2x[idx_unit];
		__m256i c0 = _mm256_set1_epi8( c_idx )^_c0;

		for(unsigned i=0;i<unit_2;i+=32) {
			__m256i v0il = _mm256_loadu_si256( (__m256i*)(v0+i)  );
			__m256i v1il = _mm256_loadu_si256( (__m256i*)(v1+i)  );
			__m256i v2il = _mm256_loadu_si256( (__m256i*)(v2+i)  );
			__m256i v3il = _mm256_loadu_si256( (__m256i*)(v3+i)  );
			__m256i v0ih = _mm256_loadu_si256( (__m256i*)(v0+unit_2+i)  );
			__m256i v1ih = _mm256_loadu_si256( (__m256i*)(v1+unit_2+i)  );
			__m256i v2ih = _mm256_loadu_si256( (__m256i*)(v2+unit_2+i)  );
			__m256i v3ih = _mm256_loadu_si256( (__m256i*)(v3+unit_2+i)  );

			ymm_x4 t0 = _gf256t4v_mul_gf256x3( v0ih , v1ih , v2ih , v3ih , c0 , _c1 , _c2 , _0x20 );

			v0il ^= t0.val[0];
			v1il ^= t0.val[1];
			v2il ^= t0.val[2];
			v3il ^= t0.val[3];
			v0ih ^= v0il;
			v1ih ^= v1il;
			v2ih ^= v2il;
			v3ih ^= v3il;

			_mm256_storeu_si256((__m256i*)(v0+i), v0il);
			_mm256_storeu_si256((__m256i*)(v1+i), v1il);
			_mm256_storeu_si256((__m256i*)(v2+i), v2il);
			_mm256_storeu_si256((__m256i*)(v3+i), v3il);
			_mm256_storeu_si256((__m256i*)(v0+unit_2+i), v0ih);
			_mm256_storeu_si256((__m256i*)(v1+unit_2+i), v1ih);
			_mm256_storeu_si256((__m256i*)(v2+unit_2+i), v2ih);
			_mm256_storeu_si256((__m256i*)(v3+unit_2+i), v3ih);
		}
		v0 += unit;
		v1 += unit;
		v2 += unit;
		v3 += unit;
	}
}

static inline
void btfy_si_gf256x2_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned si , unsigned len , uint32_t offset_gf )
{
	unsigned unit = 1<<(si+1);
	unsigned n_unit = len/unit;
	unsigned unit_2 = unit/2;

	__m256i _0x20 = _mm256_set1_epi8(0x20);
	__m256i _c0 = _mm256_set1_epi8(offset_gf&0xff);
	__m256i _c1 = _mm256_set1_epi8((offset_gf>>8)&0xff);

	unsigned idx_unit = 0;
	for(;idx_unit<n_unit;idx_unit++) {
		uint8_t c_idx = cidx_to_gf256x2_2x[idx_unit];
		__m256i c0 = _mm256_set1_epi8( c_idx )^_c0;

		for(unsigned i=0;i<unit_2;i+=32) {
			__m256i v0il = _mm256_loadu_si256( (__m256i*)(v0+i)  );
			__m256i v1il = _mm256_loadu_si256( (__m256i*)(v1+i)  );
			__m256i v2il = _mm256_loadu_si256( (__m256i*)(v2+i)  );
			__m256i v3il = _mm256_loadu_si256( (__m256i*)(v3+i)  );
			__m256i v0ih = _mm256_loadu_si256( (__m256i*)(v0+unit_2+i)  );
			__m256i v1ih = _mm256_loadu_si256( (__m256i*)(v1+unit_2+i)  );
			__m256i v2ih = _mm256_loadu_si256( (__m256i*)(v2+unit_2+i)  );
			__m256i v3ih = _mm256_loadu_si256( (__m256i*)(v3+unit_2+i)  );

			ymm_x4 t0 = _gf256t4v_mul_gf256x2( v0ih , v1ih , v2ih , v3ih , c0 , _c1 , _0x20 );

			v0il ^= t0.val[0];
			v1il ^= t0.val[1];
			v2il ^= t0.val[2];
			v3il ^= t0.val[3];
			v0ih ^= v0il;
			v1ih ^= v1il;
			v2ih ^= v2il;
			v3ih ^= v3il;

			_mm256_storeu_si256((__m256i*)(v0+i), v0il);
			_mm256_storeu_si256((__m256i*)(v1+i), v1il);
			_mm256_storeu_si256((__m256i*)(v2+i), v2il);
			_mm256_storeu_si256((__m256i*)(v3+i), v3il);
			_mm256_storeu_si256((__m256i*)(v0+unit_2+i), v0ih);
			_mm256_storeu_si256((__m256i*)(v1+unit_2+i), v1ih);
			_mm256_storeu_si256((__m256i*)(v2+unit_2+i), v2ih);
			_mm256_storeu_si256((__m256i*)(v3+unit_2+i), v3ih);
		}
		v0 += unit;
		v1 += unit;
		v2 += unit;
		v3 += unit;
	}
}

static inline
void ibtfy_si_gf256t4_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned si , unsigned len , uint32_t offset_gf )
{
	unsigned unit = 1<<(si+1);
	unsigned n_unit = len/unit;
	unsigned unit_2 = unit/2;

	__m256i _0x20 = _mm256_set1_epi8(0x20);
	__m256i _c0 = _mm256_set1_epi8(offset_gf&0xff);
	__m256i _c1 = _mm256_set1_epi8((offset_gf>>8)&0xff);
	__m256i _c2 = _mm256_set1_epi8((offset_gf>>16)&0xff);
	__m256i _c3 = _mm256_set1_epi8((offset_gf>>24)&0xff);

	unsigned idx_unit = 0;
	for(;idx_unit<n_unit;idx_unit++) {
		uint8_t c_idx = cidx_to_gf256x2_2x[idx_unit];
		__m256i c0 = _mm256_set1_epi8( c_idx )^_c0;

		for(unsigned i=0;i<unit_2;i+=32) {
			__m256i v0il = _mm256_loadu_si256( (__m256i*)(v0+i)  );
			__m256i v1il = _mm256_loadu_si256( (__m256i*)(v1+i)  );
			__m256i v2il = _mm256_loadu_si256( (__m256i*)(v2+i)  );
			__m256i v3il = _mm256_loadu_si256( (__m256i*)(v3+i)  );
			__m256i v0ih = _mm256_loadu_si256( (__m256i*)(v0+unit_2+i)  );
			__m256i v1ih = _mm256_loadu_si256( (__m256i*)(v1+unit_2+i)  );
			__m256i v2ih = _mm256_loadu_si256( (__m256i*)(v2+unit_2+i)  );
			__m256i v3ih = _mm256_loadu_si256( (__m256i*)(v3+unit_2+i)  );

			v0ih ^= v0il;
			v1ih ^= v1il;
			v2ih ^= v2il;
			v3ih ^= v3il;

			ymm_x4 t0 = _gf256t4v_mul( v0ih , v1ih , v2ih , v3ih , c0 , _c1 , _c2 , _c3 , _0x20 );

			v0il ^= t0.val[0];
			v1il ^= t0.val[1];
			v2il ^= t0.val[2];
			v3il ^= t0.val[3];

			_mm256_storeu_si256((__m256i*)(v0+i), v0il);
			_mm256_storeu_si256((__m256i*)(v1+i), v1il);
			_mm256_storeu_si256((__m256i*)(v2+i), v2il);
			_mm256_storeu_si256((__m256i*)(v3+i), v3il);
			_mm256_storeu_si256((__m256i*)(v0+unit_2+i), v0ih);
			_mm256_storeu_si256((__m256i*)(v1+unit_2+i), v1ih);
			_mm256_storeu_si256((__m256i*)(v2+unit_2+i), v2ih);
			_mm256_storeu_si256((__m256i*)(v3+unit_2+i), v3ih);
		}
		v0 += unit;
		v1 += unit;
		v2 += unit;
		v3 += unit;
	}
}

static inline
void ibtfy_si_gf256x3_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned si , unsigned len , uint32_t offset_gf )
{
	unsigned unit = 1<<(si+1);
	unsigned n_unit = len/unit;
	unsigned unit_2 = unit/2;

	__m256i _0x20 = _mm256_set1_epi8(0x20);
	__m256i _c0 = _mm256_set1_epi8(offset_gf&0xff);
	__m256i _c1 = _mm256_set1_epi8((offset_gf>>8)&0xff);
	__m256i _c2 = _mm256_set1_epi8((offset_gf>>16)&0xff);

	unsigned idx_unit = 0;
	for(;idx_unit<n_unit;idx_unit++) {
		uint8_t c_idx = cidx_to_gf256x2_2x[idx_unit];
		__m256i c0 = _mm256_set1_epi8( c_idx )^_c0;

		for(unsigned i=0;i<unit_2;i+=32) {
			__m256i v0il = _mm256_loadu_si256( (__m256i*)(v0+i)  );
			__m256i v1il = _mm256_loadu_si256( (__m256i*)(v1+i)  );
			__m256i v2il = _mm256_loadu_si256( (__m256i*)(v2+i)  );
			__m256i v3il = _mm256_loadu_si256( (__m256i*)(v3+i)  );
			__m256i v0ih = _mm256_loadu_si256( (__m256i*)(v0+unit_2+i)  );
			__m256i v1ih = _mm256_loadu_si256( (__m256i*)(v1+unit_2+i)  );
			__m256i v2ih = _mm256_loadu_si256( (__m256i*)(v2+unit_2+i)  );
			__m256i v3ih = _mm256_loadu_si256( (__m256i*)(v3+unit_2+i)  );

			v0ih ^= v0il;
			v1ih ^= v1il;
			v2ih ^= v2il;
			v3ih ^= v3il;

			ymm_x4 t0 = _gf256t4v_mul_gf256x3( v0ih , v1ih , v2ih , v3ih , c0 , _c1 , _c2 , _0x20 );

			v0il ^= t0.val[0];
			v1il ^= t0.val[1];
			v2il ^= t0.val[2];
			v3il ^= t0.val[3];

			_mm256_storeu_si256((__m256i*)(v0+i), v0il);
			_mm256_storeu_si256((__m256i*)(v1+i), v1il);
			_mm256_storeu_si256((__m256i*)(v2+i), v2il);
			_mm256_storeu_si256((__m256i*)(v3+i), v3il);
			_mm256_storeu_si256((__m256i*)(v0+unit_2+i), v0ih);
			_mm256_storeu_si256((__m256i*)(v1+unit_2+i), v1ih);
			_mm256_storeu_si256((__m256i*)(v2+unit_2+i), v2ih);
			_mm256_storeu_si256((__m256i*)(v3+unit_2+i), v3ih);
		}
		v0 += unit;
		v1 += unit;
		v2 += unit;
		v3 += unit;
	}
}

static inline
void ibtfy_si_gf256x2_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned si , unsigned len , uint32_t offset_gf )
{
	unsigned unit = 1<<(si+1);
	unsigned n_unit = len/unit;
	unsigned unit_2 = unit/2;

	__m256i _0x20 = _mm256_set1_epi8(0x20);
	__m256i _c0 = _mm256_set1_epi8(offset_gf&0xff);
	__m256i _c1 = _mm256_set1_epi8((offset_gf>>8)&0xff);

	unsigned idx_unit = 0;
	for(;idx_unit<n_unit;idx_unit++) {
		uint8_t c_idx = cidx_to_gf256x2_2x[idx_unit];
		__m256i c0 = _mm256_set1_epi8( c_idx )^_c0;

		for(unsigned i=0;i<unit_2;i+=32) {
			__m256i v0il = _mm256_loadu_si256( (__m256i*)(v0+i)  );
			__m256i v1il = _mm256_loadu_si256( (__m256i*)(v1+i)  );
			__m256i v2il = _mm256_loadu_si256( (__m256i*)(v2+i)  );
			__m256i v3il = _mm256_loadu_si256( (__m256i*)(v3+i)  );
			__m256i v0ih = _mm256_loadu_si256( (__m256i*)(v0+unit_2+i)  );
			__m256i v1ih = _mm256_loadu_si256( (__m256i*)(v1+unit_2+i)  );
			__m256i v2ih = _mm256_loadu_si256( (__m256i*)(v2+unit_2+i)  );
			__m256i v3ih = _mm256_loadu_si256( (__m256i*)(v3+unit_2+i)  );

			v0ih ^= v0il;
			v1ih ^= v1il;
			v2ih ^= v2il;
			v3ih ^= v3il;

			ymm_x4 t0 = _gf256t4v_mul_gf256x2( v0ih , v1ih , v2ih , v3ih , c0 , _c1 , _0x20 );

			v0il ^= t0.val[0];
			v1il ^= t0.val[1];
			v2il ^= t0.val[2];
			v3il ^= t0.val[3];

			_mm256_storeu_si256((__m256i*)(v0+i), v0il);
			_mm256_storeu_si256((__m256i*)(v1+i), v1il);
			_mm256_storeu_si256((__m256i*)(v2+i), v2il);
			_mm256_storeu_si256((__m256i*)(v3+i), v3il);
			_mm256_storeu_si256((__m256i*)(v0+unit_2+i), v0ih);
			_mm256_storeu_si256((__m256i*)(v1+unit_2+i), v1ih);
			_mm256_storeu_si256((__m256i*)(v2+unit_2+i), v2ih);
			_mm256_storeu_si256((__m256i*)(v3+unit_2+i), v3ih);
		}
		v0 += unit;
		v1 += unit;
		v2 += unit;
		v3 += unit;
	}
}


/////////////////////////////////////

void btfy_gf256t4_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
{
	unsigned len = 1<<n_stage;
	// si = n_stage-1 ... 6
	for(unsigned si=n_stage-1;si>5;si--) {
		uint32_t offset_gf = cidx_to_gf256t4(idx_offset>>si);
		if(offset_gf<(1<<16))       { btfy_si_gf256x2_gfni( v0 , v1 , v2 , v3 , si , len , offset_gf );}
		else if (offset_gf<(1<<24)) { btfy_si_gf256x3_gfni( v0 , v1 , v2 , v3 , si , len , offset_gf );}
		else                        { btfy_si_gf256t4_gfni( v0 , v1 , v2 , v3 , si , len , offset_gf ); }
	}
	btfy_s543210_gf256t4_gfni( v0 , v1 , v2 , v3 , len , idx_offset );
}

void ibtfy_gf256t4_gfni( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
{
	unsigned len = 1<<n_stage;
	ibtfy_s012345_gf256t4_gfni( v0 , v1 , v2 , v3 , len , idx_offset );
	// si = n_stage-1 ... 6
	for(unsigned si=6;si<n_stage;si++) {
		uint32_t offset_gf = cidx_to_gf256t4(idx_offset>>si);
		if(offset_gf<(1<<16))       { ibtfy_si_gf256x2_gfni( v0 , v1 , v2 , v3 , si , len , offset_gf ); }
		else if (offset_gf<(1<<24)) { ibtfy_si_gf256x3_gfni( v0 , v1 , v2 , v3 , si , len , offset_gf ); }
		else                        { ibtfy_si_gf256t4_gfni( v0 , v1 , v2 , v3 , si , len , offset_gf ); }
	}
}

//////////////////// avx2-gfni code end ////////////////////////////////

void btfy_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
{
	if( 5 >= n_stage ) { btfy_gf256t4_ref( v0 , v1 , v2 , v3 , n_stage , idx_offset ); return; }

	if((1<<n_stage) > MAX_BTFY_LEN) { btfy_gf256t4_ref( v0 , v1 , v2 , v3 , n_stage , idx_offset ); return; }

	btfy_gf256t4_gfni( v0 , v1 , v2 , v3 , n_stage , idx_offset );
}

void ibtfy_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset )
{
	if( 5 >= n_stage ) { ibtfy_gf256t4_ref( v0 , v1 , v2 , v3 , n_stage , idx_offset ); return; }

	if((1<<n_stage) > MAX_BTFY_LEN) { ibtfy_gf256t4_ref( v0 , v1 , v2 , v3 , n_stage , idx_offset ); return; }

	ibtfy_gf256t4_gfni( v0 , v1 , v2 , v3 , n_stage , idx_offset );
}






