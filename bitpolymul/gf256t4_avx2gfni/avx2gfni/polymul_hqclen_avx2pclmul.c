
#include "stdint.h"
#include "string.h"

#include "polymul.h"


#include <immintrin.h>

// copy ymm code from hqc's opt implementation.

// 256x256
static inline void karat_mult_256(__m128i *C, __m128i *A, __m128i *B) {
	__m128i D1[2];
	__m128i D0[2], D2[2];
	__m128i Al = _mm_load_si128(A);
	__m128i Ah = _mm_load_si128(A + 1);
	__m128i Bl = _mm_load_si128(B);
	__m128i Bh = _mm_load_si128(B + 1);

	//	Compute Al.Bl=D0
	__m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);
	__m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);
	__m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));
	__m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e));
	__m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	//	Compute Ah.Bh=D2
	DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);
	DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);
	AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));
	BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	// Compute AlpAh.BlpBh=D1
	// Initialisation of AlpAh and BlpBh
	__m128i AlpAh = _mm_xor_si128(Al,Ah);
	__m128i BlpBh = _mm_xor_si128(Bl,Bh);
	DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);
	DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);
	AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));
	BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	// Final comutation of C
	__m128i middle = _mm_xor_si128(D0[1], D2[0]);
	C[0] = D0[0];
	C[1] = middle ^ D0[0] ^ D1[0];
	C[2] = middle ^ D1[1] ^ D2[1];
	C[3] = D2[1];
}




// 512x512
static inline void karat_mult_512(__m256i *C, __m256i *A, __m256i *B) {
	__m256i D0[2], D1[2], D2[2], SAA, SBB;
	__m128i *A128 = (__m128i *)A, *B128 = (__m128i *)B;

	karat_mult_256((__m128i *) D0, A128, B128);
	karat_mult_256((__m128i *) D2, A128 + 2, B128 + 2);

	SAA = _mm256_xor_si256(A[0], A[1]);
	SBB = _mm256_xor_si256(B[0], B[1]);

	karat_mult_256((__m128i *) D1,(__m128i *) &SAA,(__m128i *) &SBB);
	__m256i middle = _mm256_xor_si256(D0[1], D2[0]);

	C[0] = D0[0];
	C[1] = middle ^ D0[0] ^ D1[0];
	C[2] = middle ^ D1[1] ^ D2[1];
	C[3] = D2[1];
}


// 1024x1024
static inline void karat_mult_1024(__m256i *C, __m256i *A, __m256i *B) {
	__m256i D0[4], D1[4], D2[4], SAA[2], SBB[2];

	karat_mult_512(D0, A, B);
	karat_mult_512(D2, A + 2, B + 2);

	SAA[0] = A[0] ^ A[2];
	SBB[0] = B[0] ^ B[2];
	SAA[1] = A[1] ^ A[3];
	SBB[1] = B[1] ^ B[3];

	karat_mult_512( D1, SAA, SBB);

	__m256i middle0 = _mm256_xor_si256(D0[2], D2[0]);
	__m256i middle1 = _mm256_xor_si256(D0[3], D2[1]);

	C[0] = D0[0];
	C[1] = D0[1];
	C[2] = middle0 ^ D0[0] ^ D1[0];
	C[3] = middle1 ^ D0[1] ^ D1[1];
	C[4] = middle0 ^ D1[2] ^ D2[2];
	C[5] = middle1 ^ D1[3] ^ D2[3];
	C[6] = D2[2];
	C[7] = D2[3];
}


// 1536x1536
static inline void karat_mult_1536(__m256i *Out, __m256i *A, __m256i *B) {
	__m256i *a0, *b0, *a1, *b1, *a2, *b2;
	__m256i aa01[2], bb01[2], aa02[2], bb02[2], aa12[2], bb12[2];
	__m256i D0[4], D1[4], D2[4], D3[4], D4[4], D5[4];
	__m256i ro256[3 * 4];

	a0 = A;
	a1 = A + 2;
	a2 = A + 4;

	b0 = B;
	b1 = B + 2;
	b2 = B + 4;

	for(int32_t i = 0; i < 2; i++) {
		aa01[i] = a0[i] ^ a1[i];
		bb01[i] = b0[i] ^ b1[i];

		aa12[i] = a2[i] ^ a1[i];
		bb12[i] = b2[i] ^ b1[i];

		aa02[i] = a0[i] ^ a2[i];
		bb02[i] = b0[i] ^ b2[i];
	}

	karat_mult_512(D0, a0, b0);
	karat_mult_512(D1, a1, b1);
	karat_mult_512(D2, a2, b2);

	karat_mult_512(D3, aa01, bb01);
	karat_mult_512(D4, aa02, bb02);
	karat_mult_512(D5, aa12, bb12);

	for(int32_t i = 0; i < 2; i++) {
		int32_t j = i + 2;
		__m256i middle0 = D0[i] ^ D1[i] ^ D0[j];
		ro256[i] = D0[i];
		ro256[j] = D3[i] ^ middle0;
		ro256[j + 2] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle0;
		middle0 = D1[j] ^ D2[i] ^ D2[j];
		ro256[j + 4] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle0;
		ro256[i + 8] = D5[j] ^ middle0;
		ro256[j + 8] = D2[j];
	}

	for(int32_t i = 0; i < 12; i++) {
		Out[i] = ro256[i];
	}
}


// 2048x2048
static inline void karat_mult_2048(__m256i *C, __m256i *A, __m256i *B) {
	__m256i D0[8], D1[8], D2[8], SAA[4], SBB[4];

	karat_mult_1024(D0, A, B);
	karat_mult_1024(D2, A + 4, B + 4);

	for(int32_t i = 0; i < 4; i++) {
		int32_t is = i + 4;
		SAA[i] = A[i] ^ A[is];
		SBB[i] = B[i] ^ B[is];
	}

	karat_mult_1024(D1, SAA, SBB);

	for(int32_t i = 0; i < 4; i++) {
		int32_t is = i + 4;
		int32_t is2 = is + 4;
		int32_t is3 = is2 + 4;

		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);

		C[i]   = D0[i];
		C[is]  = middle ^ D0[i] ^ D1[i];
		C[is2] = middle ^ D1[is] ^ D2[is];
		C[is3] = D2[is];
	}
}

// 4096x4096
static inline void karat_mult_4096(__m256i *C, __m256i *A, __m256i *B) {
	__m256i D0[16], D1[16], D2[16], SAA[8], SBB[8];

	karat_mult_2048(D0, A, B);
	karat_mult_2048(D2, A + 8, B + 8);

	for(int32_t i = 0; i < 8; i++) {
		int32_t is = i + 8;
		SAA[i] = A[i] ^ A[is];
		SBB[i] = B[i] ^ B[is];
	}

	karat_mult_2048(D1, SAA, SBB);

	for(int32_t i = 0; i < 8; i++) {
		int32_t is = i + 8;
		int32_t is2 = is + 8;
		int32_t is3 = is2 + 8;

		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);

		C[i]   = D0[i];
		C[is]  = middle ^ D0[i] ^ D1[i];
		C[is2] = middle ^ D1[is] ^ D2[is];
		C[is3] = D2[is];
	}
}



#include "polymul.h"




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
    polymul_input_transform( a_fft , a , 256  );
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
    polymul_output( c , a_fft , b_fft , 256 );
////////////////

    __m256i _a[70]; memcpy( _a , a , 70*32 );
    __m256i _b[70]; memcpy( _b , b , 70*32 );
    __m256i tmp0[12];
    __m256i tmp1[12];

    const __m256i * a_poly = _a;
    const __m256i * b_poly = _b;
    const __m256i * a_rem = a_poly + 64;  // 6 ymm
    const __m256i * b_rem = b_poly + 64;  // 6 ymm
    __m256i * cc = (__m256i*) (c);

    {
        //karat_mult_1536( cc+128 , a_rem , b_rem );
        karat_mult_1536( tmp0 , a_rem , b_rem );
        for (int j=0;j<12;j++) { _mm256_storeu_si256( cc+128+j , tmp0[j] ); }
    }

    for( int i=0;i<60;i+=6) {
        karat_mult_1536( tmp0 , a_poly + i , b_rem );
        karat_mult_1536( tmp1 , b_poly + i , a_rem );
        for (int j=0;j<12;j++) {
            __m256i tmp = _mm256_loadu_si256( cc+64+i+j ) ^ tmp0[j]^tmp1[j];
            _mm256_storeu_si256( cc+64+i+j , tmp );
            //cc[64+i+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
    {
        int i=60;
        karat_mult_1024( tmp0 , a_poly + i , b_rem );
        karat_mult_1024( tmp1 , b_poly + i , a_rem );
        for (int j=0;j<8;j++) {
            __m256i tmp = _mm256_loadu_si256( cc+64+i+j ) ^ tmp0[j]^tmp1[j];
            _mm256_storeu_si256( cc + 64+i+j , tmp );
            //cc[64+i+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
    const __m256i * a_h = a_poly + 60;  // 4 ymm
    const __m256i * b_h = b_poly + 60;  // 4 ymm
    a_rem += 4;
    b_rem += 4;
    for(int k=0;k<4;k+=2) {
        karat_mult_512( tmp0 , a_h + k , b_rem );
        karat_mult_512( tmp1 , b_h + k , a_rem );
        for (int j=0;j<4;j++) {
            __m256i tmp = _mm256_loadu_si256( cc+60+64+4+k+j ) ^ tmp0[j]^tmp1[j];
            _mm256_storeu_si256( cc+60+64+4+k+j , tmp );
            //cc[60+64+4+k+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
}

//////////////////////////////////

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
    polymul_input_transform( a_fft , a , 512  );
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
    polymul_output( c , a_fft , b_fft , 512 );
////////////////

    __m256i _a[144]; memcpy( _a , a , 144*32 );
    __m256i _b[144]; memcpy( _b , b , 144*32 );
    __m256i tmp0[32];
    __m256i tmp1[32];

    const __m256i * a_poly = _a;
    const __m256i * b_poly = _b;
    const __m256i * a_rem = a_poly + 128;  // 16 ymm
    const __m256i * b_rem = b_poly + 128;  // 16 ymm
    __m256i * cc = (__m256i*) (c);

    {
        //karat_mult_4096( c+1024 , a_rem , b_rem );
        karat_mult_4096( tmp0 , a_rem , b_rem );
        for (int j=0;j<32;j++) { _mm256_storeu_si256( cc+256+j , tmp0[j] ); }
    }
    for( int i=0;i<128;i+=16) {
        karat_mult_4096( tmp0 , a_poly + i , b_rem );
        karat_mult_4096( tmp1 , b_poly + i , a_rem );
        for (int j=0;j<32;j++) {
            __m256i tmp = _mm256_loadu_si256( cc+128+i+j ) ^ tmp0[j]^tmp1[j];
            _mm256_storeu_si256( cc+128+i+j , tmp );
            //cc[128+i+j] ^= tmp0[j] ^ tmp1[j];
        }
    }
}


