
#include "stdint.h"
#include "string.h"

#include "polymul.h"

#include <arm_neon.h>

static inline void _mull_64( uint64_t * c0 , uint64_t * c1 , uint64_t a , uint64_t b )
{
    // poly128_t	vmull_p64	(poly64_t a, poly64_t b)
    //uint64x2_t cc = vreinterpretq_u64_p128(vmull_p64(vcreate_p64(a),vcreate_p64(b)));
    uint64x2_t cc = vreinterpretq_u64_p128(vmull_p64((poly64_t)(a),(poly64_t)(b)));
    c0[0] = vgetq_lane_u64(cc, 0);;
    c1[0] = vgetq_lane_u64(cc, 1);
}


// 128x128
static inline void karat_mult_128( uint64_t *C, const uint64_t *A, const uint64_t *B) {
    uint64_t a0 = A[0];
    uint64_t a1 = A[1];
    uint64_t b0 = B[0];
    uint64_t b1 = B[1];

    uint64_t a0a1 = a0^a1;
    uint64_t b0b1 = b0^b1;

    uint64_t t0,t1;
    _mull_64(&t0, &t1, a0a1, b0b1);

    uint64_t c0, c1, c2, c3;
    _mull_64(&c0, &c1, a0, b0);
    _mull_64(&c2, &c3, a1, b1);
    t0 ^= c0 ^ c2;
    t1 ^= c1 ^ c3;
    c1 ^= t0;
    c2 ^= t1;

    C[0] = c0;
    C[1] = c1;
    C[2] = c2;
    C[3] = c3;
}

// 256x256
static inline void karat_mult_256( uint64_t *C, const uint64_t *A, const uint64_t *B) {
#define LEN  (4)
#define HALF (LEN/2)
    uint64_t t0[LEN], a0a1[HALF], b0b1[HALF];

    for(int i=0;i<HALF;i++) { a0a1[i] = A[i]^A[HALF+i]; }
    for(int i=0;i<HALF;i++) { b0b1[i] = B[i]^B[HALF+i]; }

    karat_mult_128(t0, a0a1, b0b1);

    karat_mult_128(C, A, B);
    karat_mult_128(C+LEN, A+HALF, B+HALF);

    for(int i=0;i<LEN;i++) { t0[i] ^= C[i] ^ C[LEN+i]; }
    for(int i=0;i<LEN;i++) { C[i+HALF] ^= t0[i]; }
#undef LEN
#undef HALF
}

// 512x512
static inline void karat_mult_512( uint64_t *C, const uint64_t *A, const uint64_t *B) {
#define LEN  (8)
#define HALF (LEN/2)
    uint64_t t0[LEN], a0a1[HALF], b0b1[HALF];

    for(int i=0;i<HALF;i++) { a0a1[i] = A[i]^A[HALF+i]; }
    for(int i=0;i<HALF;i++) { b0b1[i] = B[i]^B[HALF+i]; }

    karat_mult_256(t0, a0a1, b0b1);

    karat_mult_256(C, A, B);
    karat_mult_256(C+LEN, A+HALF, B+HALF);

    for(int i=0;i<LEN;i++) { t0[i] ^= C[i] ^ C[LEN+i]; }
    for(int i=0;i<LEN;i++) { C[i+HALF] ^= t0[i]; }
#undef LEN
#undef HALF
}


// 1024x1024
static inline void karat_mult_1024( uint64_t *C, const uint64_t *A, const uint64_t *B) {
#define LEN  (16)
#define HALF (LEN/2)
    uint64_t t0[LEN], a0a1[HALF], b0b1[HALF];

    for(int i=0;i<HALF;i++) { a0a1[i] = A[i]^A[HALF+i]; }
    for(int i=0;i<HALF;i++) { b0b1[i] = B[i]^B[HALF+i]; }

    karat_mult_512(t0, a0a1, b0b1);

    karat_mult_512(C, A, B);
    karat_mult_512(C+LEN, A+HALF, B+HALF);

    for(int i=0;i<LEN;i++) { t0[i] ^= C[i] ^ C[LEN+i]; }
    for(int i=0;i<LEN;i++) { C[i+HALF] ^= t0[i]; }
#undef LEN
#undef HALF
}


// 1536x1536
static inline void karat_mult_1536( uint64_t *C, const uint64_t *A, const uint64_t *B) {
#define LEN  (24)
#define HALF (8)
    uint64_t a0a1[HALF], a1a2[HALF], a0a2[HALF], b0b1[HALF], b1b2[HALF], b0b2[HALF];

    for(int i=0;i<HALF;i++) { a0a1[i] = A[i]^A[HALF+i]; }
    for(int i=0;i<HALF;i++) { a1a2[i] = A[i+HALF]^A[2*HALF+i]; }
    for(int i=0;i<HALF;i++) { a0a2[i] = A[i]^A[2*HALF+i]; }

    for(int i=0;i<HALF;i++) { b0b1[i] = B[i]^B[HALF+i]; }
    for(int i=0;i<HALF;i++) { b1b2[i] = B[i+HALF]^B[2*HALF+i]; }
    for(int i=0;i<HALF;i++) { b0b2[i] = B[i]^B[2*HALF+i]; }

    uint64_t D0[2*HALF], D1[2*HALF], D2[2*HALF], D3[2*HALF], D4[2*HALF], D5[2*HALF];

	karat_mult_512(D0, A, B);
	karat_mult_512(D1, A+HALF, B+HALF);
	karat_mult_512(D2, A+2*HALF, B+2*HALF);

	karat_mult_512(D3, a0a1, b0b1);
	karat_mult_512(D4, a0a2, b0b2);
	karat_mult_512(D5, a1a2, b1b2);

    for(int i=0;i<HALF;i++) {
        int j = i + HALF;
        uint64_t middle0 = D0[i] ^ D1[i] ^ D0[j];
        C[i] = D0[i];
        C[j] = D3[i] ^ middle0;
        C[j + HALF] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle0;
        middle0 = D1[j] ^ D2[i] ^ D2[j];
        C[j + HALF*2] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle0;
        C[i + HALF*4] = D5[j] ^ middle0;
        C[j + HALF*4] = D2[j];
    }
#undef LEN
#undef HALF
}


// 2048x2048
static inline void karat_mult_2048( uint64_t *C, const uint64_t *A, const uint64_t *B) {
#define LEN  (32)
#define HALF (LEN/2)
    uint64_t t0[LEN], a0a1[HALF], b0b1[HALF];

    for(int i=0;i<HALF;i++) { a0a1[i] = A[i]^A[HALF+i]; }
    for(int i=0;i<HALF;i++) { b0b1[i] = B[i]^B[HALF+i]; }

    karat_mult_1024(t0, a0a1, b0b1);

    karat_mult_1024(C, A, B);
    karat_mult_1024(C+LEN, A+HALF, B+HALF);

    for(int i=0;i<LEN;i++) { t0[i] ^= C[i] ^ C[LEN+i]; }
    for(int i=0;i<LEN;i++) { C[i+HALF] ^= t0[i]; }
#undef LEN
#undef HALF
}

// 4096x4096
static inline void karat_mult_4096( uint64_t *C, const uint64_t *A, const uint64_t *B) {
#define LEN  (64)
#define HALF (LEN/2)
    uint64_t t0[LEN], a0a1[HALF], b0b1[HALF];

    for(int i=0;i<HALF;i++) { a0a1[i] = A[i]^A[HALF+i]; }
    for(int i=0;i<HALF;i++) { b0b1[i] = B[i]^B[HALF+i]; }

    karat_mult_2048(t0, a0a1, b0b1);

    karat_mult_2048(C, A, B);
    karat_mult_2048(C+LEN, A+HALF, B+HALF);

    for(int i=0;i<LEN;i++) { t0[i] ^= C[i] ^ C[LEN+i]; }
    for(int i=0;i<LEN;i++) { C[i+HALF] ^= t0[i]; }
#undef LEN
#undef HALF
}



#include "btfy.h"
#include "gf264.h"
#include "dencoder.h"
#include "bc_1.h"






///
/// @brief input transform for multiplying two bit-polynomials of size < 17920 bits
///
/// @param a_fft [out] size : 32768 bits + 280u64 = 792 u64
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 280 u64
void polymul_17920_input( uint64_t * a_fft , const uint64_t * a )
{
    polymul_280U64_input( a_fft , a );
    uint64_t * fft_end = a_fft + 512;  // offset 32768 bits
    memcpy( fft_end , a , 280*8 );
}


///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 560 u64
/// @param a_fft [in] a transformed bit-polynomial. size : 512+280 u64
/// @param b_fft [in] a transformed bit-polynomial. size : 512+280 u64
void polymul_17920_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft )
{
    polymul_280U64_mul( c , a_fft+512 , b_fft+512 , a_fft , b_fft );
}


///
/// @brief input transform for multiplying two bit-polynomials of size < (16384+1536) bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 65536 bits = 1024 u64  = 8192 bytes
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 256+24 u64 = 2048+192 bytes
void polymul_280U64_input( uint64_t * a_fft , const uint64_t * a )
{
    uint64_t fft_end[256];
    memcpy( fft_end , a , 256*8 );  // 256xu64 = 16384 bits
    bc_1( fft_end , 256*8 );
    encode_64(a_fft,512,fft_end,32);
    btfy_64(a_fft,8+1,1ULL<<(32+8+1));
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 512+48 u64 = 4480=(4096+384) bytes
/// @param a [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param b [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
void polymul_280U64_mul( uint64_t * c , const uint64_t * a , const uint64_t * b , const uint64_t * a_fft , const uint64_t * b_fft )
{
    uint64_t temp[512];
    gf264v_mul( temp , a_fft , b_fft , 512 );
    // output transform
    ibtfy_64( temp,8+1,1ULL<<(32+8+1));
    decode_64( c , temp , 512 );
    ibc_1( c , 512*8 );

    const uint64_t * a_poly = a;  // 280 u64
    const uint64_t * b_poly = b;  // 280 u64
    const uint64_t * a_rem = a_poly + 256;  // 24 u64 = 3 x 8 u64
    const uint64_t * b_rem = b_poly + 256;  // 24 u64 = 3 x 8 u64

    karat_mult_1536( c+512 , a_rem , b_rem );

    uint64_t tmp0[48];
    uint64_t tmp1[48];
    for( int i=0;i<240;i+=24) {
        karat_mult_1536( tmp0 , a_poly + i , b_rem );
        karat_mult_1536( tmp1 , b_poly + i , a_rem );
        for (int j=0;j<48;j++) {
            c[256+i+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
    {
        int i=240;
        karat_mult_1024( tmp0 , a_poly + i , b_rem );
        karat_mult_1024( tmp1 , b_poly + i , a_rem );
        for (int j=0;j<32;j++) {
            c[256+i+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
    const uint64_t * a_h = a_poly + 240;  // 16 u64
    const uint64_t * b_h = b_poly + 240;  // 16 u64
    a_rem += 16;
    b_rem += 16;
    for(int k=0;k<16;k+=8) {
        karat_mult_512( tmp0 , a_h + k , b_rem );
        karat_mult_512( tmp1 , b_h + k , a_rem );
        for (int j=0;j<16;j++) {
            c[240+256+16+k+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
}

////////////////////////


///
/// @brief input transform for multiplying two bit-polynomials of 36860 bits
///
/// @param a_fft [out] size : 65536 + 36864 bits = 1024+512+64 u64
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 512+64 u64
void polymul_36864_input( uint64_t * a_fft , const uint64_t * a )
{
    polymul_576U64_input( a_fft , a );
    uint64_t * fft_end = a_fft + 1024;  // offset 65536 bits
    memcpy( fft_end , a , 576*8 );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 65536 + 8192 bits = 1152 u64
/// @param a_fft [in] a transformed bit-polynomial. size : 1024+576 u64
/// @param b_fft [in] a transformed bit-polynomial. size : 1024+576 u64
void polymul_36864_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft )
{
    polymul_576U64_mul( c , a_fft+1024 , b_fft+1024 , a_fft , b_fft );
}


///
/// @brief input transform for multiplying two bit-polynomials of size < 32768+4096 bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 2048 u64  = 16384 bytes
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 512+64 u64 = 4608=(4096+512) bytes
void polymul_576U64_input( uint64_t * a_fft , const uint64_t * a )
{
    uint64_t fft_end[512];
    // input transform
    memcpy( fft_end , a , 512*8 );  // 512xu64 = 32768 bits
    bc_1( fft_end , 512*8 );
    encode_64(a_fft,1024,fft_end,32);
    btfy_64(a_fft,9+1,1ULL<<(32+9+1));
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 1024+128 u64 = 8976=(8192+1024) bytes
/// @param a [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param b [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
void polymul_576U64_mul( uint64_t * c , const uint64_t * a , const uint64_t * b , const uint64_t * a_fft , const uint64_t * b_fft )
{
    uint64_t temp[1024];
    gf264v_mul( temp , a_fft , b_fft , 1024 );
    // output transform
    ibtfy_64( temp,9+1,1ULL<<(32+9+1));
    decode_64( c , temp , 1024 );
    ibc_1( c , 1024*8 );

    const uint64_t * a_poly = a;  // 576 u64
    const uint64_t * b_poly = b;  // 576 u64
    const uint64_t * a_rem = a_poly + 512;  // 64 u64
    const uint64_t * b_rem = b_poly + 512;  // 64 u64

    karat_mult_4096( c+1024 , a_rem , b_rem );

    uint64_t tmp0[128];
    uint64_t tmp1[128];
    for( int i=0;i<512;i+=64) {
        karat_mult_4096( tmp0 , a_poly + i , b_rem );
        karat_mult_4096( tmp1 , b_poly + i , a_rem );
        for (int j=0;j<128;j++) {
            c[512+i+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
}


