

#include <stdint.h>
#include <string.h>


#include "gf264.h"
#include "cantor_to_gf264.h"
#include "btfy.h"
#include "gf264_neon.h"

#include <arm_neon.h>

static inline unsigned min(unsigned a,unsigned b) { return (a<b)?a:b; }

//////////////////////////////////

static inline
void btfy_s0( uint64_t * poly , uint64_t scalar_a )
{
	uint64_t extra_a = cantor_to_gf264(scalar_a);
	uint64_t p0 = poly[0];
	uint64_t p1 = poly[1];
	p0 ^= _gf264_mulx1_neon( p1 , extra_a );
	p1 ^= p0;
	poly[0] = p0;
	poly[1] = p1;
}

static inline
void i_btfy_s0( uint64_t * poly , uint64_t scalar_a )
{
	uint64_t extra_a = cantor_to_gf264(scalar_a);
	uint64_t p0 = poly[0];
	uint64_t p1 = poly[1];
	p1 ^= p0;
	p0 ^= _gf264_mulx1_neon( p1 , extra_a );
	poly[0] = p0;
	poly[1] = p1;
}

///////////////////////////////


// (log_n > 1)
// (n_terms <= SIZE_TBL_CANTOR2X*2)
static void btfy_64_median( uint64_t * poly , unsigned log_n , uint64_t scalar_a );
static void ibtfy_64_median( uint64_t * poly , unsigned log_n , uint64_t scalar_a );

// (log_n > 1)
static void btfy_64_large( uint64_t * poly , unsigned log_n , uint64_t scalar_a );
static void ibtfy_64_large( uint64_t * poly , unsigned log_n , uint64_t scalar_a );


void btfy_64( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
	if( 0 == log_n ) {
		return;
	} else if( 1 == log_n ) {
		btfy_s0(poly,scalar_a);
	} else if ( (1<<log_n) <= SIZE_TBL_CANTOR2X*2 ) {
		btfy_64_median(poly,log_n,scalar_a);
	} else {
		btfy_64_large(poly,log_n,scalar_a);
	}
}

void ibtfy_64( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
	if( 0 == log_n ) {
		return;
	} else if( 1 == log_n ) {
		i_btfy_s0(poly,scalar_a);
	} else if ( (1<<log_n) <= SIZE_TBL_CANTOR2X*2 ) {
		ibtfy_64_median(poly,log_n,scalar_a);
	} else {
		ibtfy_64_large(poly,log_n,scalar_a);
	}
}


////////////////////////////////

static inline
void butterfly_64( uint64_t * poly , unsigned unit ,  uint64_t a )
{
	uint64x2_t mask_0x1b = vdupq_n_u64(0x1b);
	uint64x2_t aa = vdupq_n_u64(a);
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		uint64x2_t p0 = vld1q_u64( poly+i );
		uint64x2_t p1 = vld1q_u64( poly+unit_2+i );
		p0 ^= _gf264_mul_neon( aa , p1 , mask_0x1b );
		p1 ^= p0;
		vst1q_u64( poly+i , p0 );
		vst1q_u64( poly+unit_2+i , p1 );
	}
}

// 2 stages
static inline
void butterfly_2s_64( uint64_t * poly , unsigned unit ,  uint64x2_t s1a , uint64x2_t s0a )
{
	uint64x2_t mask_0x1b = get_mask_0x1b();
	uint64x2_t s0a0 = vtrn1q_u64(s0a,s0a);
	uint64x2_t s0a1 = vtrn2q_u64(s0a,s0a);
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		uint64x2_t p0_l = vld1q_u64( poly+i );
		uint64x2_t p0_h = vld1q_u64( poly+unit_2+i );
		uint64x2_t p1_l = vld1q_u64( poly+unit+i );
		uint64x2_t p1_h = vld1q_u64( poly+unit+unit_2+i );

		// stage 1
		p0_h ^= _gf264_mul_neon( s1a , p1_h , mask_0x1b );
		uint64x2_t p1_l_s1a_0 = _vmull_p64( p1_l , s1a );
		uint64x2_t p1_l_s1a_1 = _vmull_high_p64( p1_l , s1a );
		p1_l ^= p0_l;
		p1_h ^= p0_h;
		// delayed:
		// _p0_l_0 ^= p1_l_s1a_0;
		// _p0_l_1 ^= p1_l_s1a_1;
		// _p1_l_0 ^= p1_l_s1a_0;
		// _p1_l_1 ^= p1_l_s1a_1;

		// stage 0
		uint64x2_t p0_h_s0a_0 = _vmull_p64( p0_h , s0a0 )     ^p1_l_s1a_0;
		uint64x2_t p0_h_s0a_1 = _vmull_high_p64( p0_h , s0a0 )^p1_l_s1a_1;
		uint64x2_t p1_h_s0a_0 = _vmull_p64( p1_h , s0a1 )     ^p1_l_s1a_0;
		uint64x2_t p1_h_s0a_1 = _vmull_high_p64( p1_h , s0a1 )^p1_l_s1a_1;
		p0_l ^= _gf264_reduce_neon( p0_h_s0a_0 , p0_h_s0a_1 , mask_0x1b );
		p1_l ^= _gf264_reduce_neon( p1_h_s0a_0 , p1_h_s0a_1 , mask_0x1b );
		p0_h ^= p0_l;
		p1_h ^= p1_l;

		vst1q_u64( poly+i , p0_l );
		vst1q_u64( poly+unit_2+i , p0_h );
		vst1q_u64( poly+unit+i , p1_l );
		vst1q_u64( poly+unit+unit_2+i , p1_h );
	}
}

static inline
void i_butterfly_64( uint64_t * poly , unsigned unit , uint64_t a )
{
	uint64x2_t mask_0x1b = vdupq_n_u64(0x1b);
	uint64x2_t aa = vdupq_n_u64(a);
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		uint64x2_t p0 = vld1q_u64( poly+i );
		uint64x2_t p1 = vld1q_u64( poly+unit_2+i );
		p1 ^= p0;
		p0 ^= _gf264_mul_neon( aa , p1 , mask_0x1b );
		vst1q_u64( poly+i , p0 );
		vst1q_u64( poly+unit_2+i , p1 );
	}
}

static inline
void i_butterfly_2s_64( uint64_t * poly , unsigned unit ,  uint64x2_t s1a , uint64x2_t s0a )
{
	uint64x2_t mask_0x1b = get_mask_0x1b();
	uint64x2_t s0a0 = vtrn1q_u64(s0a,s0a);
	uint64x2_t s0a1 = vtrn2q_u64(s0a,s0a);
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		uint64x2_t p0_l = vld1q_u64( poly+i );
		uint64x2_t p0_h = vld1q_u64( poly+unit_2+i );
		uint64x2_t p1_l = vld1q_u64( poly+unit+i );
		uint64x2_t p1_h = vld1q_u64( poly+unit+unit_2+i );

		// stage 0
		p0_h ^= p0_l;
		p1_h ^= p1_l;

		uint64x2_t p0_l_r0 = _vmull_p64( p0_h , s0a0 );
		uint64x2_t p0_l_r1 = _vmull_high_p64( p0_h , s0a0 );

		uint64x2_t p1_l_r0 = _vmull_p64( p1_h , s0a1 );
		uint64x2_t p1_l_r1 = _vmull_high_p64( p1_h , s0a1 );
		// delayed:
		// _p0_l_0 = p0_l_0 ^ p0_l_r0;
		// _p0_l_1 = p0_l_1 ^ p0_l_r1;

		// stage 1
		p1_l ^= p0_l ^ _gf264_reduce_neon( p1_l_r0^p0_l_r0 , p1_l_r1^p0_l_r1 , mask_0x1b );
		p1_h ^= p0_h;

		uint64x2_t p0_l_rr0 = _vmull_p64( p1_l , s1a );
		uint64x2_t p0_l_rr1 = _vmull_high_p64( p1_l , s1a );

		p0_l ^= _gf264_reduce_neon( p0_l_rr0^p0_l_r0 , p0_l_rr1^p0_l_r1 , mask_0x1b );
		p0_h ^= _gf264_mul_neon( p1_h , s1a , mask_0x1b );
		vst1q_u64( poly+i , p0_l );
		vst1q_u64( poly+unit_2+i , p0_h );
		vst1q_u64( poly+unit+i , p1_l );
		vst1q_u64( poly+unit+unit_2+i , p1_h );
	}
}

/////////////////////////////////////////////////////////

static inline
void btfy_s1s0( uint64_t * poly , uint64x2_t s1_a , uint64x2_t s0_a01 )
{
	uint64x2_t zero = vdupq_n_u64(0);
	uint64x2_t mask_0x1b = get_mask_0x1b();
	uint64x2_t t01 = vld1q_u64( poly );
	uint64x2_t t23 = vld1q_u64( poly + 2 );

	// stage 1
	uint64x2_t t2s1a = _vmull_p64( t23 , s1_a );
	uint64x2_t _t3s1a = _vmull_high_p64(t23,s1_a);
	//uint64x2_t t3s1a = vextq_u64(zero,_gf264_reduce_x1_neon(_t3s1a,mask_0x1b),1);
	uint64x2_t t3s1a = vtrn1q_u64(zero,_gf264_reduce_x1_neon(_t3s1a,mask_0x1b));
	t01 ^= t3s1a;
	t23 ^= t01;

	// stage 0
	uint64x2_t t1s0a = t2s1a ^ _vmull_high_p64( t01 , vextq_u64(s0_a01,s0_a01,1) );
	uint64x2_t t3s0a = t2s1a ^ _vmull_high_p64( t23 , s0_a01 );
	uint64x2_t t0t2 = _gf264_reduce_neon( t1s0a , t3s0a , mask_0x1b ) ^ vtrn1q_u64(t01,t23);
	uint64x2_t t1t3 = t0t2 ^ vtrn2q_u64(t01,t23);

	uint64x2x2_t p;
	p.val[0] = t0t2;
	p.val[1] = t1t3;
	vst2q_u64( poly , p );
}


static inline
void i_btfy_s0s1( uint64_t * poly , uint64x2_t s1_a , uint64x2_t s0_a01 )
{
	uint64x2_t zero = vdupq_n_u64(0);
	uint64x2_t mask_0x1b = get_mask_0x1b();

	uint64x2x2_t p = vld2q_u64( poly );
	uint64x2_t t02 = p.val[0];
	uint64x2_t t13 = p.val[1];
	// stage 0
	t13 ^= t02;
	uint64x2_t t1s0a = _vmull_p64( t13 , s0_a01 );
	uint64x2_t t3s0a = _vmull_high_p64( t13 , s0_a01 );
	uint64x2_t rt2 = _gf264_reduce_x1_neon(t1s0a^t3s0a,mask_0x1b);

	// stage 1
	uint64x2_t t01 = vtrn1q_u64(t02,t13);
	uint64x2_t t23 = vtrn2q_u64(t02,t13) ^ t01 ^ vtrn1q_u64(rt2,zero);

	uint64x2_t t2s1a = _vmull_p64( t23 , s1_a ) ^ t1s0a;
	uint64x2_t t3s1a = _vmull_high_p64(t23,s1_a );
	t01 ^= _gf264_reduce_neon( t2s1a , t3s1a , mask_0x1b );

	vst1q_u64( poly , t01 );
	vst1q_u64( poly + 2 , t23 );
}

/////////////////////////////////////////////////////////

// assert( log_n > 1 )
// assert(n_terms <= SIZE_TBL_CANTOR2X*2)
static
void btfy_64_median( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
	// assert( log_n > 2 )
	// assert( n_terms <= SIZE_TBL_CANTOR2X*2 )

	unsigned n_terms = 1<<log_n;
	unsigned si=log_n-1;

	// if si is even
	if (0==(si&1)) {
		unsigned unit = 1<<(si+1);
		unsigned num  = 1<<(log_n-(si+1)); // n_terms / unit

		uint64_t extra_a = cantor_to_gf264(scalar_a>>si);
		for(unsigned j=0;j<num;j++) {
			uint64_t a = cantor_to_gf264_2x[j]^extra_a;
			butterfly_64( poly + j*unit , unit , a );
		}
		si -= 1;
	}
	for (;si>1;si-=2) {
		unsigned unit = 1<<(si);
		unsigned num  = 1<<(log_n-(si)); // n_terms / unit

		uint64x2_t extra_a1 = vdupq_n_u64(cantor_to_gf264(scalar_a>>si));
		uint64x2_t extra_a0 = vdupq_n_u64(cantor_to_gf264(scalar_a>>(si-1)));

		for(unsigned j=0;j<num;j+=2) {
			//uint64x2_t s1a = vdupq_n_u64( cantor_to_gf264_2x[j>>1] ) ^ extra_a1;
			//uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[j] ) ^ extra_a0;
			uint64x2_t s0a = vld1q_u64( btfy_consts_2stages + j*2 ) ^ extra_a0;
			uint64x2_t s1a = vld1q_u64( btfy_consts_2stages + j*2+2 ) ^ extra_a1;
			butterfly_2s_64( poly + j*unit , unit , s1a , s0a );
		}
	}
	// si = 1 and 0
	do {
		unsigned num_s0 = n_terms/2;
		uint64x2_t extra_s1a = vdupq_n_u64(cantor_to_gf264(scalar_a>>1));
		uint64x2_t extra_s0a = vdupq_n_u64(cantor_to_gf264(scalar_a));
		for(unsigned j=0;j<num_s0;j+=2) {
			//uint64x2_t s1a = vdupq_n_u64(cantor_to_gf264_2x[j>>1])^extra_s1a;
			//uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[j] )^ extra_s0a;
			uint64x2_t s0a = vld1q_u64( btfy_consts_2stages + j*2 ) ^ extra_s0a;
			uint64x2_t s1a = vld1q_u64( btfy_consts_2stages + j*2+2 ) ^ extra_s1a;
			btfy_s1s0( poly + j*2 , s1a , s0a );
		}
	} while(0);
}

// assert( log_n > 1 )
// assert(n_terms <= SIZE_TBL_CANTOR2X*2)
static
void ibtfy_64_median( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
    unsigned n_terms = 1<<log_n;
	// si = 1 and 0
	do {
		unsigned num_s0 = n_terms/2;
		uint64x2_t extra_s1a = vdupq_n_u64(cantor_to_gf264(scalar_a>>1));
		uint64x2_t extra_s0a = vdupq_n_u64(cantor_to_gf264(scalar_a));
		for(unsigned j=0;j<num_s0;j+=2) {
			//uint64x2_t s1a = vdupq_n_u64(cantor_to_gf264_2x[j>>1])^extra_s1a;
			//uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[j] )^ extra_s0a;
			uint64x2_t s0a = vld1q_u64( btfy_consts_2stages + j*2 ) ^ extra_s0a;
			uint64x2_t s1a = vld1q_u64( btfy_consts_2stages + j*2+2 ) ^ extra_s1a;
			i_btfy_s0s1( poly + j*2 , s1a , s0a );
		}
	} while(0);
	unsigned si=2;
	for( ; si+1<log_n; si += 2 ) {
		unsigned unit = 1<<(si+1);
		unsigned num  = 1<<(log_n-(si+1)); // n_terms / unit

		uint64x2_t extra_a1 = vdupq_n_u64(cantor_to_gf264(scalar_a>>(si+1)));
		uint64x2_t extra_a0 = vdupq_n_u64(cantor_to_gf264(scalar_a>>(si)));
		for(unsigned j=0;j<num;j+=2) {
			//uint64x2_t s1a = vdupq_n_u64( cantor_to_gf264_2x[j>>1] ) ^ extra_a1;
			//uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[j] ) ^ extra_a0;
			uint64x2_t s0a = vld1q_u64( btfy_consts_2stages + j*2 ) ^ extra_a0;
			uint64x2_t s1a = vld1q_u64( btfy_consts_2stages + j*2+2 ) ^ extra_a1;
			i_butterfly_2s_64( poly + j*unit , unit , s1a , s0a );
		}
	}
	if ( si<log_n ) {
		unsigned unit = 1<<(si+1);
		unsigned num  = 1<<(log_n-(si+1)); // n_terms / unit

		uint64_t extra_a = cantor_to_gf264(scalar_a>>si);
		for(unsigned j=0;j<num;j++) {
			uint64_t a = cantor_to_gf264_2x[j]^extra_a;
			i_butterfly_64( poly + j*unit , unit , a );
		}
		si += 1;
	}
}

//////////////////////////////////////////////////////

// assert( log_n > 1 )
static
void btfy_64_large( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
	// assert( log_n > 2 )
	// assert( n_terms <= SIZE_TBL_CANTOR2X*2 )

	unsigned n_terms = 1<<log_n;
	unsigned si=log_n-1;

	// if si is even
	if (0==(si&1)) {
		unsigned unit = 1<<(si+1);
		unsigned num  = 1<<(log_n-(si+1)); // n_terms / unit

		uint64_t extra_a = cantor_to_gf264(scalar_a>>si);
		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);

		for(unsigned j=0;j<num;j+=step_size) {  // process constants outside of the cantor_to_gf264_2x[]
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64_t extra_a_j = extra_a ^ cantor_to_gf264(j<<1);
			for(unsigned k=0;k<step_size;k++) {
				uint64_t a = cantor_to_gf264_2x[k]^extra_a_j;
				butterfly_64( poly + (j+k)*unit , unit , a );
			}
		}
		si -= 1;
	}
	for (;si>1;si-=2) {
		unsigned unit = 1<<(si);
		unsigned num  = 1<<(log_n-(si)); // n_terms / unit

		uint64x2_t extra_a1 = vdupq_n_u64(cantor_to_gf264(scalar_a>>si));
		uint64x2_t extra_a0 = vdupq_n_u64(cantor_to_gf264(scalar_a>>(si-1)));
		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);
		for(unsigned j=0;j<num;j+=step_size) {  // process constants outside of the cantor_to_gf264_2x[]
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64x2_t extra_a1_j = extra_a1 ^ vdupq_n_u64(cantor_to_gf264(j));
			uint64x2_t extra_a0_j = extra_a0 ^ vdupq_n_u64(cantor_to_gf264(j<<1));
			for(unsigned k=0;k<step_size;k+=2) {
				uint64x2_t s1a = vdupq_n_u64( cantor_to_gf264_2x[k>>1]) ^ extra_a1_j;
				uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[k] ) ^ extra_a0_j;
				butterfly_2s_64( poly + (j+k)*unit , unit , s1a , s0a );
			}
		}
	}
	// si = 1 and 0
	do {
		unsigned num = n_terms/2;
		uint64x2_t extra_s1a = vdupq_n_u64(cantor_to_gf264(scalar_a>>1));
		uint64x2_t extra_s0a = vdupq_n_u64(cantor_to_gf264(scalar_a));
		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);
		for(unsigned j=0;j<num;j+=step_size) {  // process constants outside of the cantor_to_gf264_2x[]
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64x2_t extra_a1_j = extra_s1a ^ vdupq_n_u64(cantor_to_gf264(j));
			uint64x2_t extra_a0_j = extra_s0a ^ vdupq_n_u64(cantor_to_gf264(j<<1));
			for(unsigned k=0;k<step_size;k+=2) {
				uint64x2_t s1a = vdupq_n_u64( cantor_to_gf264_2x[k>>1]) ^ extra_a1_j;
				uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[k] ) ^ extra_a0_j;
				btfy_s1s0( poly + (j+k)*2 , s1a , s0a );
			}
		}
	} while(0);
}

// assert( log_n > 1 )
static
void ibtfy_64_large( uint64_t * poly , unsigned log_n , uint64_t scalar_a )
{
    unsigned n_terms = 1<<log_n;

	// si = 1 and 0
	do {
		unsigned num = n_terms/2;
		uint64x2_t extra_s1a = vdupq_n_u64(cantor_to_gf264(scalar_a>>1));
		uint64x2_t extra_s0a = vdupq_n_u64(cantor_to_gf264(scalar_a));
		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);
		for(unsigned j=0;j<num;j+=step_size) {  // process constants outside of the cantor_to_gf264_2x[]
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64x2_t extra_a1_j = extra_s1a ^ vdupq_n_u64(cantor_to_gf264(j));
			uint64x2_t extra_a0_j = extra_s0a ^ vdupq_n_u64(cantor_to_gf264(j<<1));
			for(unsigned k=0;k<step_size;k+=2) {
				uint64x2_t s1a = vdupq_n_u64( cantor_to_gf264_2x[k>>1]) ^ extra_a1_j;
				uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[k] ) ^ extra_a0_j;
				i_btfy_s0s1( poly + (j+k)*2 , s1a , s0a );
			}
		}
	} while(0);
	unsigned si=2;
	for( ; si+1<log_n; si += 2 ) {
		unsigned unit = 1<<(si+1);
		unsigned num  = 1<<(log_n-(si+1)); // n_terms / unit

		uint64x2_t extra_a1 = vdupq_n_u64(cantor_to_gf264(scalar_a>>(si+1)));
		uint64x2_t extra_a0 = vdupq_n_u64(cantor_to_gf264(scalar_a>>(si)));
		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);
		for(unsigned j=0;j<num;j+=step_size) {  // process constants outside of the cantor_to_gf264_2x[]
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64x2_t extra_a1_j = extra_a1 ^ vdupq_n_u64(cantor_to_gf264(j));
			uint64x2_t extra_a0_j = extra_a0 ^ vdupq_n_u64(cantor_to_gf264(j<<1));
			for(unsigned k=0;k<step_size;k+=2) {
				uint64x2_t s1a = vdupq_n_u64( cantor_to_gf264_2x[k>>1]) ^ extra_a1_j;
				uint64x2_t s0a = vld1q_u64( &cantor_to_gf264_2x[k] ) ^ extra_a0_j;
				i_butterfly_2s_64( poly + (j+k)*unit , unit , s1a , s0a );
			}
		}

	}
	if ( si<log_n ) {
		unsigned unit = 1<<(si+1);
		unsigned num  = 1<<(log_n-(si+1)); // n_terms / unit

		uint64_t extra_a = cantor_to_gf264(scalar_a>>si);
		unsigned step_size = min(SIZE_TBL_CANTOR2X,num);
		for(unsigned j=0;j<num;j+=step_size) {  // process constants outside of the cantor_to_gf264_2x[]
			//unsigned idx = ((j+k)<<(si+1)) + scalar_a
			uint64_t extra_a_j = extra_a ^ cantor_to_gf264(j<<1);
			for(unsigned k=0;k<step_size;k++) {
				uint64_t a = cantor_to_gf264_2x[k]^extra_a_j;
				i_butterfly_64( poly + (j+k)*unit , unit , a );
			}
		}
		si += 1;
	}
}


