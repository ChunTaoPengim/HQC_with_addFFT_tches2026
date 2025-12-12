#include "stdint.h"
#include "string.h"

#include "polymul.h"


#include "ringmul.h"

#include "gf256.h"

///
/// @brief input transform for multiplying two bit-polynomials of size < 17920 bits
///
/// @param a_fft [out] size : 65536 + 3072 bits = 1024+48 u64  = 8192+384 bytes
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 256+24 u64 = 2048+192 bytes
void polymul_17920_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s12_input_2240( (uint8_t*)a_fft , ((uint8_t*)a_fft)+4096 ,  (uint8_t*)a );
    memcpy( ((uint8_t*)a_fft)+4096*2 , ((uint8_t*)a) , 384 );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 560 u64 = 4480=(4096+384) bytes
/// @param a_fft [in] a transformed bit-polynomial. size : 1072 u64 = 8192+384 bytes
/// @param b_fft [in] a transformed bit-polynomial. size : 1072 u64 = 8192+384 bytes
void polymul_17920_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft )
{
    polymul_280U64_mul( c , a_fft+(POLYMUL_280U64_FFTSIZE_U64) , b_fft+(POLYMUL_280U64_FFTSIZE_U64) , a_fft , b_fft );
}

/////////////////////

///
/// @brief input transform for multiplying two bit-polynomials of size < (16384+1536) bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 65536 bits = 1024 u64  = 8192 bytes
/// @param a [in] a bit-polynomial of size < 16384 + 1536 bits = 256+24 u64 = 2048+192 bytes
void polymul_280U64_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s12_input_2240( (uint8_t*)a_fft , ((uint8_t*)a_fft)+(POLYMUL_280U64_FFTSIZE_U64*4) ,  (uint8_t*)a );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 16384+1536 bits.
///
/// @param c [out] product bit-polynomials of size < 32768 + 3072 bits = 512+48 u64 = 4480=(4096+384) bytes
/// @param a [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param b [in] the input bit-polynomial. size : < 280 u64 = 2240 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 1024 u64 = 8192 bytes
void polymul_280U64_mul( uint64_t * _c , const uint64_t * _a , const uint64_t * _b , const uint64_t * _a_fft , const uint64_t * _b_fft )
{
#define LEN0 (4096)
#define LEN1 (384)
    uint8_t * c = (uint8_t*)_c;
    uint8_t * a = (uint8_t*)_a;
    uint8_t * b = (uint8_t*)_b;
    const uint8_t * a_fft = (const uint8_t*)_a_fft;
    const uint8_t * b_fft = (const uint8_t*)_b_fft;
    uint8_t pc_m384_l[LEN1+32];
    uint8_t pc_m384_h[LEN1+32];
    ringmul_mul_384( pc_m384_l , pc_m384_h , a , b );
    pc_m384_l[LEN1] = 0;
    pc_m384_h[LEN1] = 0;
    uint8_t pc_s12_l[LEN0+LEN1+32];
    uint8_t pc_s12_h[LEN0+LEN1+32];
    ringmul_s12_mul( pc_s12_l , pc_s12_h , a_fft , a_fft+LEN0 , b_fft , b_fft+LEN0 );
    memset( pc_s12_l+LEN0 , 0 , LEN1+32 );
    memset( pc_s12_h+LEN0 , 0 , LEN1+32 );

    static const int invs12[] = {0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 270, 300, 330, 360};
    uint8_t k_s12_div_x_l[LEN1];
    uint8_t k_s12_div_x_h[LEN1];

    //  #k*(s12/x) = (pc_m384+pc_ms12)/x   mod  x^384
    //  k_s12_div_x = xor_list(pc_m384,pc_ms12)[1:384+1]
    gf256v_add( k_s12_div_x_l, pc_s12_l+1 , pc_m384_l+1 , LEN1 );
    gf256v_add( k_s12_div_x_h, pc_s12_h+1 , pc_m384_h+1 , LEN1 );

    uint8_t k_l[LEN1];
    uint8_t k_h[LEN1];
    // #k = (s12/x)^-1 * k_s12_div_x
    // k = poly_mul_gf2poly( k_s12_div_x , invs12 )[:384]
    memcpy( k_l , k_s12_div_x_l , LEN1 );
    memcpy( k_h , k_s12_div_x_h , LEN1 );
    for( int i = 1 ; i < (int)(sizeof(invs12)/sizeof(int)) ; i++ ) {
        int raise_deg = invs12[i];
        gf256v_add( k_l+raise_deg , k_l+raise_deg , k_s12_div_x_l , LEN1-raise_deg );
        gf256v_add( k_h+raise_deg , k_h+raise_deg , k_s12_div_x_h , LEN1-raise_deg );
    }

    //# f = pc_ms12 + k*s12   mod (lcm(x^384, x^4096+x^256+x^16+x))
    //f = xor_list( pc_ms12 , poly_mul_gf2poly( k , [4096,256,16,1] ) )
    uint8_t * f_l= pc_s12_l;
    uint8_t * f_h= pc_s12_h;
    { int r_deg = 1;    gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 16;   gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 256;  gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 4096; gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }

    // rr = poly_mod_gf2poly( f , [4096+384-1,256+384-1,16+384-1,1+384-1] )  # lcm(x^384, x^4096+x^256+x^16+x)
    uint8_t red_l = f_l[LEN0+LEN1-1]; f_l[LEN0+LEN1-1] = 0;
    uint8_t red_h = f_h[LEN0+LEN1-1]; f_h[LEN0+LEN1-1] = 0;
    f_l[256+LEN1-1] ^= red_l;
    f_h[256+LEN1-1] ^= red_h;
    f_l[16+LEN1-1]  ^= red_l;
    f_h[16+LEN1-1]  ^= red_h;
    f_l[1+LEN1-1]   ^= red_l;
    f_h[1+LEN1-1]   ^= red_h;

    c[0] = f_l[0];
    gf256v_add( c+1 , f_l+1 , f_h , LEN0+LEN1-1 );
#undef LEN0
#undef LEN1
}


////////////////////////

///
/// @brief input transform for multiplying two bit-polynomials of size < 32768+4096 bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 2048 u64  = 16384 bytes
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 512+64 u64 = 4608=(4096+512) bytes
void polymul_36864_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s13_input_4608( (uint8_t*)a_fft , ((uint8_t*)a_fft)+8192 ,  (uint8_t*)a );
    memcpy( ((uint8_t*)a_fft)+8192*2 , ((uint8_t*)a) , 1024 );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 65536 + 8192 bits = 1152 u64 = 8192+1024 bytes
/// @param a_fft [in] a transformed bit-polynomial. size : 2176 u64 = 16384+1024 bytes
/// @param b_fft [in] a transformed bit-polynomial. size : 2176 u64 = 16384+1024 bytes
void polymul_36864_mul( uint64_t * c , const uint64_t * a_fft , const uint64_t * b_fft )
{
    polymul_576U64_mul( c , a_fft+(POLYMUL_576U64_FFTSIZE_U64) , b_fft+(POLYMUL_576U64_FFTSIZE_U64) , a_fft , b_fft );
}



///
/// @brief input transform for multiplying two bit-polynomials of size < 32768+4096 bits
///
/// @param a_fft [out] size : transformed values of the partial input polynomial: 2048 u64  = 16384 bytes
/// @param a [in] a bit-polynomial of size < 32768 + 4096 bits = 512+64 u64 = 4608=(4096+512) bytes
void polymul_576U64_input( uint64_t * a_fft , const uint64_t * a )
{
    ringmul_s13_input_4608( (uint8_t*)a_fft , ((uint8_t*)a_fft)+(POLYMUL_576U64_FFTSIZE_U64*4) ,  (uint8_t*)a );
    //ringmul_s13_input_4608( (uint8_t*)a_fft , ((uint8_t*)a_fft)+8192 ,  (uint8_t*)a );
}

///
/// @brief compute c = a * b. a and b are bit-polynomials of size < 32768+4096 bits.
///
/// @param c [out] product bit-polynomials of size < 1024+128 u64 = 8976=(8192+1024) bytes
/// @param a [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param b [in] the input bit-polynomial. size : < 512+64 u64 = 4608 bytes
/// @param a_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
/// @param b_fft [in] a transformed partial bit-polynomial. size : 2048 u64 = 16384 bytes
void polymul_576U64_mul( uint64_t * _c , const uint64_t * _a , const uint64_t * _b , const uint64_t * _a_fft , const uint64_t * _b_fft )
{
#define LEN0 (8192)
#define LEN1 (1024)
    uint8_t * c = (uint8_t*)_c;
    uint8_t * a = (uint8_t*)_a;
    uint8_t * b = (uint8_t*)_b;
    const uint8_t * a_fft = (const uint8_t*)_a_fft;
    const uint8_t * b_fft = (const uint8_t*)_b_fft;
    uint8_t pc_m1024_l[LEN1+32];
    uint8_t pc_m1024_h[LEN1+32];
    ringmul_mul_1024( pc_m1024_l , pc_m1024_h , a , b );
    pc_m1024_l[LEN1] = 0;
    pc_m1024_h[LEN1] = 0;
    uint8_t pc_s13_l[LEN0+LEN1+32];
    uint8_t pc_s13_h[LEN0+LEN1+32];
    ringmul_s13_mul( pc_s13_l , pc_s13_h , a_fft , a_fft+LEN0 , b_fft , b_fft+LEN0 );
    memset( pc_s13_l+LEN0 , 0 , LEN1+32 );
    memset( pc_s13_h+LEN0 , 0 , LEN1+32 );

    static const int invs13[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28, 32, 33, 36, 37, 40, 41, 44, 48, 52, 56, 64, 65, 66, 67, 72, 73, 74, 80, 82, 88, 96, 97, 104, 112, 128, 129, 130, 131, 132, 133, 134, 144, 146, 148, 160, 161, 164, 176, 192, 193, 194, 208, 224, 256, 258, 260, 262, 264, 266, 268, 288, 292, 296, 320, 322, 328, 352, 384, 386, 388, 416, 448, 512, 513, 516, 517, 520, 521, 524, 528, 532, 536, 576, 577, 584, 592, 640, 641, 644, 656, 704, 768, 772, 776, 832, 896};
    uint8_t k_s13_div_x_l[1024];
    uint8_t k_s13_div_x_h[1024];

    //  #k*(s13/x) = (pc_m1024+pc_ms13)/x   mod  x^1024
    //  k_s13_div_x = xor_list(pc_m1024,pc_ms13)[1:1024+1]
    gf256v_add( k_s13_div_x_l, pc_s13_l+1 , pc_m1024_l+1 , LEN1 );
    gf256v_add( k_s13_div_x_h, pc_s13_h+1 , pc_m1024_h+1 , LEN1 );

    uint8_t k_l[LEN1];
    uint8_t k_h[LEN1];
    // #k = (s13/x)^-1 * k_s13_div_x
    // k = poly_mul_gf2poly( k_s13_div_x , invs13 )[:1024]
    memcpy( k_l , k_s13_div_x_l , LEN1 );
    memcpy( k_h , k_s13_div_x_h , LEN1 );
    for( int i = 1 ; i < (int)(sizeof(invs13)/sizeof(int)) ; i++ ) {
        int raise_deg = invs13[i];
        gf256v_add( k_l+raise_deg , k_l+raise_deg , k_s13_div_x_l , LEN1-raise_deg );
        gf256v_add( k_h+raise_deg , k_h+raise_deg , k_s13_div_x_h , LEN1-raise_deg );
    }

    //# f = pc_ms13 + k*s13   mod (lcm(x^1024, s13))
    //f = xor_list( pc_ms13 , poly_mul_gf2poly( k , [8192,4096,512,256,32,16,2,1] ) )
    uint8_t * f_l= pc_s13_l;
    uint8_t * f_h= pc_s13_h;
    { int r_deg = 1;    gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 2;    gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 16;   gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 32;   gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 256;  gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 512;  gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 4096; gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }
    { int r_deg = 8192; gf256v_add( f_l+r_deg , f_l+r_deg , k_l , LEN1 ); gf256v_add( f_h+r_deg , f_h+r_deg , k_h , LEN1 ); }

    // rr = poly_mod_gf2poly( f , [4096+384-1,256+384-1,16+384-1,1+384-1] )  # lcm(x^1024, s13 )
    uint8_t red_l = f_l[LEN0+LEN1-1]; f_l[LEN0+LEN1-1] = 0;
    uint8_t red_h = f_h[LEN0+LEN1-1]; f_h[LEN0+LEN1-1] = 0;
    f_l[4096+LEN1-1] ^= red_l;
    f_h[4096+LEN1-1] ^= red_h;
    f_l[512+LEN1-1] ^= red_l;
    f_h[512+LEN1-1] ^= red_h;
    f_l[256+LEN1-1] ^= red_l;
    f_h[256+LEN1-1] ^= red_h;
    f_l[32+LEN1-1] ^= red_l;
    f_h[32+LEN1-1] ^= red_h;
    f_l[16+LEN1-1] ^= red_l;
    f_h[16+LEN1-1] ^= red_h;
    f_l[2+LEN1-1] ^= red_l;
    f_h[2+LEN1-1] ^= red_h;
    f_l[1+LEN1-1] ^= red_l;
    f_h[1+LEN1-1] ^= red_h;

    c[0] = f_l[0];
    gf256v_add( c+1 , f_l+1 , f_h , LEN0+LEN1-1 );
#undef LEN0
#undef LEN1
}



