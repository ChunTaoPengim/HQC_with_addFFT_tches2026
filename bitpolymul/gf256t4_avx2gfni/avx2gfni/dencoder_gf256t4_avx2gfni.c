
#include "dencoder_gf256t4.h"

/////////////  ref code starts. //////////////////////////////////////////

static uint32_t _enc_mat[32] = {
1,0x1f6fa,0x8456,0x845658ee,0xe02,0xe02bfa6,0x315,0x315ee26,
0x5a68,0x5a68449d,0xc149,0xc149b5be,0x939c,0x939cfa1c,0xa8cd,0xa8cd1324,
0xae46,0xae464233,0x6241,0x624191bb,0x24,0x241cb7,0xd6d,0xd6d8e51,
0xe36c,0xe36cdc1e,0x61af,0x61afdf65,0xc614,0xc6145364,0x7adc,0x7adcb3e4,
};

static uint32_t _dec_mat[32] = {
1,0x54415404,0x54145005,0x5451115,0x115510,0x54045005,0x5514054,0x4440011,
0x11505554,0x45555000,0x10145501,0x1005115,0x14004505,0x44455401,0x4454454,0x15014440,
0x10441452,0xfd93ec18,0xbd6ca10e,0x5e8e336e,0x23ab70,0xe85ce11b,0xab3c4b8,0x589c0173,
0x63e0afe8,0x9bfae541,0x716caa03,0x4745a36b,0x6d148b0a,0xdc9efd06,0x4c9fdca8,0x7a47ddc5,
};

static void encode_gf256t4_ref( uint8_t * v0 , uint8_t * v1, uint8_t * v2, uint8_t * v3, unsigned n,
    const uint32_t * bitvec , unsigned src_bits )
{
    unsigned batch = n/8;  // process 8 elements at a time
    for(unsigned i=0;i<batch;i++){ // process 8 elements at a time
        uint32_t rr[8] = {0};
        const uint8_t * ptr = ((const uint8_t *)bitvec)+i;
        for(unsigned j=0;j<src_bits;j++){
            uint8_t v = ptr[0];
            for(int k=0;k<8;k++) rr[k] ^= (-((v>>k)&1))&_enc_mat[j];
            ptr += batch;
        }
        // output
        for(int k=0;k<8;k++){
            v0[k] = rr[k]&0xff;
            v1[k] = (rr[k]>>8)&0xff;
            v2[k] = (rr[k]>>16)&0xff;
            v3[k] = (rr[k]>>24)&0xff;
        }
        v0+=8;
        v1+=8;
        v2+=8;
        v3+=8;
    }
}

#define SWAPMOVE(R1,R2,TMP,MASK,N_BITS) do { \
  TMP = (R2^(R1>>(N_BITS)))&(MASK); \
  R2 = R2^TMP; \
  R1 = R1^(TMP<<(N_BITS)); \
} while(0)

static void transpose_8x8( uint8_t * mat0 )
{
    uint8_t tmp;
    SWAPMOVE(mat0[0],mat0[4],tmp,0x0f,4);
    SWAPMOVE(mat0[1],mat0[5],tmp,0x0f,4);
    SWAPMOVE(mat0[2],mat0[6],tmp,0x0f,4);
    SWAPMOVE(mat0[3],mat0[7],tmp,0x0f,4);

    SWAPMOVE(mat0[0],mat0[2],tmp,0x33,2);
    SWAPMOVE(mat0[1],mat0[3],tmp,0x33,2);
    SWAPMOVE(mat0[4],mat0[6],tmp,0x33,2);
    SWAPMOVE(mat0[5],mat0[7],tmp,0x33,2);

    SWAPMOVE(mat0[0],mat0[1],tmp,0x55,1);
    SWAPMOVE(mat0[2],mat0[3],tmp,0x55,1);
    SWAPMOVE(mat0[4],mat0[5],tmp,0x55,1);
    SWAPMOVE(mat0[6],mat0[7],tmp,0x55,1);
}

static void decode_gf256t4_ref( uint32_t * bitvec 
    , const uint8_t * v0, const uint8_t * v1, const uint8_t * v2, const uint8_t * v3, unsigned n )
{
    unsigned step = n/8; // process 8 elements in one step
    for(unsigned i=0;i<step;i++){
        uint32_t rr[8] = {0};
        for(int j=0;j<8;j++) { // bit idx of input
            for(int k=0;k<8;k++) { rr[k] ^= (-((v0[8*i+k]>>j)&1))&_dec_mat[j]; }
            for(int k=0;k<8;k++) { rr[k] ^= (-((v1[8*i+k]>>j)&1))&_dec_mat[j+8]; }
            for(int k=0;k<8;k++) { rr[k] ^= (-((v2[8*i+k]>>j)&1))&_dec_mat[j+16]; }
            for(int k=0;k<8;k++) { rr[k] ^= (-((v3[8*i+k]>>j)&1))&_dec_mat[j+24]; }
        }

        // output
        uint8_t * ptr = ((uint8_t *)bitvec)+i;
        for(int k=0;k<4;k++) { // 4 bytes in one uint32_t
            uint8_t mat0[8];
            for(int j=0;j<8;j++) { mat0[j] = *(((uint8_t*)&rr[j])+k); }
            transpose_8x8(mat0);
            for(int j=0;j<8;j++){
                ptr[(k*8+j)*step] = mat0[j];
            }
        }
    }
}

/////////////  end of ref code. avx2 code starts. //////////////////////////////////////////

#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

typedef struct {
    __m256i v[2];
} ymm_x2;

typedef struct {
    __m256i v[4];
} ymm_x4;


static inline void _byte_transpose_8x8_x4( __m256i * row )
{
    __m256i t0 = _mm256_blend_epi32( row[0] , _mm256_slli_epi64(row[4],32) , 0xaa ); // 1010,1010
    __m256i t4 = _mm256_blend_epi32( _mm256_srli_epi64(row[0],32) , row[4] , 0xaa ); // 1010,1010
    __m256i t1 = _mm256_blend_epi32( row[1] , _mm256_slli_epi64(row[5],32) , 0xaa ); // 1010,1010
    __m256i t5 = _mm256_blend_epi32( _mm256_srli_epi64(row[1],32) , row[5] , 0xaa ); // 1010,1010
    __m256i t2 = _mm256_blend_epi32( row[2] , _mm256_slli_epi64(row[6],32) , 0xaa ); // 1010,1010
    __m256i t6 = _mm256_blend_epi32( _mm256_srli_epi64(row[2],32) , row[6] , 0xaa ); // 1010,1010
    __m256i t3 = _mm256_blend_epi32( row[3] , _mm256_slli_epi64(row[7],32) , 0xaa ); // 1010,1010
    __m256i t7 = _mm256_blend_epi32( _mm256_srli_epi64(row[3],32) , row[7] , 0xaa ); // 1010,1010

    __m256i q0 = _mm256_blend_epi16( t0 , _mm256_slli_epi32(t2,16) , 0xaa ); // 1010,1010
    __m256i q2 = _mm256_blend_epi16( _mm256_srli_epi32(t0,16) , t2 , 0xaa ); // 1010,1010
    __m256i q1 = _mm256_blend_epi16( t1 , _mm256_slli_epi32(t3,16) , 0xaa ); // 1010,1010
    __m256i q3 = _mm256_blend_epi16( _mm256_srli_epi32(t1,16) , t3 , 0xaa ); // 1010,1010
    __m256i q4 = _mm256_blend_epi16( t4 , _mm256_slli_epi32(t6,16) , 0xaa ); // 1010,1010
    __m256i q6 = _mm256_blend_epi16( _mm256_srli_epi32(t4,16) , t6 , 0xaa ); // 1010,1010
    __m256i q5 = _mm256_blend_epi16( t5 , _mm256_slli_epi32(t7,16) , 0xaa ); // 1010,1010
    __m256i q7 = _mm256_blend_epi16( _mm256_srli_epi32(t5,16) , t7 , 0xaa ); // 1010,1010

    __m256i _00ff     = _mm256_set1_epi16(0x00ff);
    row[0] = (q0&_00ff)|_mm256_slli_epi16(q1,8);
    row[1] = _mm256_andnot_si256( _00ff , q1 )|_mm256_srli_epi16(q0,8);
    row[2] = (q2&_00ff)|_mm256_slli_epi16(q3,8);
    row[3] = _mm256_andnot_si256( _00ff , q3 )|_mm256_srli_epi16(q2,8);
    row[4] = (q4&_00ff)|_mm256_slli_epi16(q5,8);
    row[5] = _mm256_andnot_si256( _00ff , q5 )|_mm256_srli_epi16(q4,8);
    row[6] = (q6&_00ff)|_mm256_slli_epi16(q7,8);
    row[7] = _mm256_andnot_si256( _00ff , q7 )|_mm256_srli_epi16(q6,8);
}

static inline void _bit_transpose_8x8_x32( __m256i * row )
{
    __m256i _0x0f = _mm256_set1_epi8(0xf);
    __m256i _0x33 = _mm256_set1_epi8(0x33);
    __m256i _0x55 = _mm256_set1_epi8(0x55);

    __m256i tmp;
    tmp = (_mm256_srli_epi16(row[0],4)^row[4])&_0x0f;
    row[0] ^= _mm256_slli_epi16(tmp,4);
    row[4] ^= tmp;
    tmp = (_mm256_srli_epi16(row[1],4)^row[5])&_0x0f;
    row[1] ^= _mm256_slli_epi16(tmp,4);
    row[5] ^= tmp;
    tmp = (_mm256_srli_epi16(row[2],4)^row[6])&_0x0f;
    row[2] ^= _mm256_slli_epi16(tmp,4);
    row[6] ^= tmp;
    tmp = (_mm256_srli_epi16(row[3],4)^row[7])&_0x0f;
    row[3] ^= _mm256_slli_epi16(tmp,4);
    row[7] ^= tmp;

    tmp = (_mm256_srli_epi16(row[0],2)^row[2])&_0x33;
    row[0] ^= _mm256_slli_epi16(tmp,2);
    row[2] ^= tmp;
    tmp = (_mm256_srli_epi16(row[1],2)^row[3])&_0x33;
    row[1] ^= _mm256_slli_epi16(tmp,2);
    row[3] ^= tmp;
    tmp = (_mm256_srli_epi16(row[4],2)^row[6])&_0x33;
    row[4] ^= _mm256_slli_epi16(tmp,2);
    row[6] ^= tmp;
    tmp = (_mm256_srli_epi16(row[5],2)^row[7])&_0x33;
    row[5] ^= _mm256_slli_epi16(tmp,2);
    row[7] ^= tmp;

    tmp = (_mm256_srli_epi16(row[0],1)^row[1])&_0x55;
    row[0] ^= _mm256_slli_epi16(tmp,1);
    row[1] ^= tmp;
    tmp = (_mm256_srli_epi16(row[2],1)^row[3])&_0x55;
    row[2] ^= _mm256_slli_epi16(tmp,1);
    row[3] ^= tmp;
    tmp = (_mm256_srli_epi16(row[4],1)^row[5])&_0x55;
    row[4] ^= _mm256_slli_epi16(tmp,1);
    row[5] ^= tmp;
    tmp = (_mm256_srli_epi16(row[6],1)^row[7])&_0x55;
    row[6] ^= _mm256_slli_epi16(tmp,1);
    row[7] ^= tmp;
}

///////////////////////////////////////////////////////////////////////


#define _enc_0_0 _mm256_set1_epi64x(0x41beec0a46aa0e2aULL)
#define _enc_0_1 _mm256_set1_epi64x(0x60f2b6b82aa28aa6ULL)
#define _enc_0_2 _mm256_set1_epi64x(0x8228880088000800ULL)
#define _enc_0_3 _mm256_set1_epi64x(0x80a0282000000008ULL)
#define _enc_1_0 _mm256_set1_epi64x(0x4608fa7f3a89455aULL)
#define _enc_1_1 _mm256_set1_epi64x(0x9cb10a61b968277cULL)
#define _enc_1_2 _mm256_set1_epi64x(0x8800a0aa20028aa0ULL)
#define _enc_1_3 _mm256_set1_epi64x(0x2822008222800aa8ULL)
#define _enc_2_0 _mm256_set1_epi64x(0xee2b7148aa7ac528ULL)
#define _enc_2_1 _mm256_set1_epi64x(0x4887e1e128050689ULL)
#define _enc_2_2 _mm256_set1_epi64x(0x8802a28000a08a00ULL)
#define _enc_2_3 _mm256_set1_epi64x(0x800a8282000a0802ULL)
#define _enc_3_0 _mm256_set1_epi64x(0x0c06ff4752ade9c4ULL)
#define _enc_3_1 _mm256_set1_epi64x(0xadf91a4aeac57f9bULL)
#define _enc_3_2 _mm256_set1_epi64x(0x0808aa8aa00a8288ULL)
#define _enc_3_3 _mm256_set1_epi64x(0x0aa22080808aaa22ULL)

static inline
ymm_x4 encode_core( __m256i d0 , __m256i d1 )
{
    ymm_x4 r;
    r.v[0] = _mm256_gf2p8affine_epi64_epi8( d0 , _enc_0_0 , 0 );
    r.v[1] = _mm256_gf2p8affine_epi64_epi8( d0 , _enc_0_1 , 0 );
    r.v[2] = _mm256_gf2p8affine_epi64_epi8( d0 , _enc_0_2 , 0 );
    r.v[3] = _mm256_gf2p8affine_epi64_epi8( d0 , _enc_0_3 , 0 );
    r.v[0] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _enc_1_0 , 0 );
    r.v[1] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _enc_1_1 , 0 );
    r.v[2] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _enc_1_2 , 0 );
    r.v[3] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _enc_1_3 , 0 );
    return r;
}



static inline
void collect_128( __m256i * temp , const uint8_t * inp , unsigned blocklen_in_byte )
{
    union{
    __m256i ymm[8];
    __m128i xmm[16];
    } u;
    for(int i=0;i<8;i++) { u.xmm[i*2] = _mm_loadu_si128((__m128i*)(inp+blocklen_in_byte*i)); }  // first byte
    for(int i=0;i<8;i++) { u.xmm[i*2+1] = _mm_loadu_si128((__m128i*)(inp+blocklen_in_byte*(i+8))); }  // second byte

    _bit_transpose_8x8_x32( u.ymm );
    _byte_transpose_8x8_x4( u.ymm );

    __m256i t0,t1,t2,t3,t4,t5,t6,t7;
    t0 = _mm256_unpacklo_epi64( u.ymm[0] , u.ymm[1] );
    t4 = _mm256_unpackhi_epi64( u.ymm[0] , u.ymm[1] );
    t1 = _mm256_unpacklo_epi64( u.ymm[2] , u.ymm[3] );
    t5 = _mm256_unpackhi_epi64( u.ymm[2] , u.ymm[3] );
    t2 = _mm256_unpacklo_epi64( u.ymm[4] , u.ymm[5] );
    t6 = _mm256_unpackhi_epi64( u.ymm[4] , u.ymm[5] );
    t3 = _mm256_unpacklo_epi64( u.ymm[6] , u.ymm[7] );
    t7 = _mm256_unpackhi_epi64( u.ymm[6] , u.ymm[7] );

    // permute 128 and output
    temp[0] = _mm256_permute2x128_si256( t0 , t1 , 0x20 );
    temp[1] = _mm256_permute2x128_si256( t0 , t1 , 0x31 );
    temp[2] = _mm256_permute2x128_si256( t2 , t3 , 0x20 );
    temp[3] = _mm256_permute2x128_si256( t2 , t3 , 0x31 );
    temp[4] = _mm256_permute2x128_si256( t4 , t5 , 0x20 );
    temp[5] = _mm256_permute2x128_si256( t4 , t5 , 0x31 );
    temp[6] = _mm256_permute2x128_si256( t6 , t7 , 0x20 );
    temp[7] = _mm256_permute2x128_si256( t6 , t7 , 0x31 );
}

static
void encode_gf256t4_gfni( uint8_t * v0, uint8_t * v1, uint8_t * v2, uint8_t * v3, unsigned n , const uint32_t * bitvec )
{
    unsigned blocklen = n/8; // Total vector len: n*4 bytes. one block len : total_len/32 = n/8
    __m256i temp[8];
    for(unsigned i=0;i<n;i+=128){
        // collect 16 128-bit chunks. transform it into 128 2-byte elements (byte-slice form).
        collect_128( temp , ((uint8_t*)bitvec)+(i/8) , blocklen );   // i/8 = (i/128)*16

        for(int j=0;j<4;j++){
            ymm_x4 t = encode_core( temp[j*2], temp[j*2+1] );
            _mm256_storeu_si256( (__m256i*)(v0+i+j*32) , t.v[0] );
            _mm256_storeu_si256( (__m256i*)(v1+i+j*32) , t.v[1] );
            _mm256_storeu_si256( (__m256i*)(v2+i+j*32) , t.v[2] );
            _mm256_storeu_si256( (__m256i*)(v3+i+j*32) , t.v[3] );
        }
    }
}

/////////////////////////////////////////////////////////////////////////////

#if 0
static
void encode_gf256t4_256_gfni( uint8_t * v0, uint8_t * v1, uint8_t * v2, uint8_t * v3, unsigned n , const uint32_t * bitvec , unsigned src_bits )
{
    unsigned blocklen = n/8; // Total vector len: n*4 bytes. one block len : total_len/32 = n/8
    __m256i temp[8];
    __m256i temp0[8];
    __m256i temp1[8];
    __m256i temp2[8];
    for(unsigned i=0;i<n;i+=128){
        // collect src_bits 256-bit chunks. transform it into 256 byte-slice elements.
        collect_256( temp0 , ((uint8_t*)bitvec)+(i/8) , blocklen , src_bits );   // i/8 = (i/128)*16

        for(int j=0;j<4;j++){
            ymm_x4 t = encode_core( temp[j*2], temp[j*2+1] , src_bits );
            _mm256_storeu_si256( (__m256i*)(v0+i+j*32) , t.v[0] );
            _mm256_storeu_si256( (__m256i*)(v1+i+j*32) , t.v[1] );
            _mm256_storeu_si256( (__m256i*)(v2+i+j*32) , t.v[2] );
            _mm256_storeu_si256( (__m256i*)(v3+i+j*32) , t.v[3] );
        }
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////


#define _dec_0_0 _mm256_set1_epi64x(0xad006e00d8004000ULL)
#define _dec_0_1 _mm256_set1_epi64x(0x180012003e007600ULL)
#define _dec_0_2 _mm256_set1_epi64x(0x5a00ac005400ca00ULL)
#define _dec_0_3 _mm256_set1_epi64x(0x4800ee0026002600ULL)
#define _dec_1_0 _mm256_set1_epi64x(0x3c0059004900c100ULL)
#define _dec_1_1 _mm256_set1_epi64x(0x1d00f5002f00ff00ULL)
#define _dec_1_2 _mm256_set1_epi64x(0xe200660007006300ULL)
#define _dec_1_3 _mm256_set1_epi64x(0x8b00f20095002200ULL)
#define _dec_2_0 _mm256_set1_epi64x(0xa0ad0c6ef3d89940ULL)
#define _dec_2_1 _mm256_set1_epi64x(0xbc184312093e6276ULL)
#define _dec_2_2 _mm256_set1_epi64x(0x525aadace25425caULL)
#define _dec_2_3 _mm256_set1_epi64x(0x6480eee8f26aa26ULL)
#define _dec_3_0 _mm256_set1_epi64x(0x8e3ca05900498bc1ULL)
#define _dec_3_1 _mm256_set1_epi64x(0xbb1de3f5e02fe2ffULL)
#define _dec_3_2 _mm256_set1_epi64x(0xc8e2fc6672078f63ULL)
#define _dec_3_3 _mm256_set1_epi64x(0x1f8b78f2a695fd22ULL)

static inline
ymm_x4 decode_core( __m256i d0 , __m256i d1 , __m256i d2 , __m256i d3 )
{
    ymm_x4 r;
    r.v[0] = _mm256_gf2p8affine_epi64_epi8( d0 , _dec_0_0 , 0 );
    r.v[1] = _mm256_gf2p8affine_epi64_epi8( d0 , _dec_0_1 , 0 );
    r.v[2] = _mm256_gf2p8affine_epi64_epi8( d0 , _dec_0_2 , 0 );
    r.v[3] = _mm256_gf2p8affine_epi64_epi8( d0 , _dec_0_3 , 0 );
    r.v[0] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _dec_1_0 , 0 );
    r.v[1] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _dec_1_1 , 0 );
    r.v[2] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _dec_1_2 , 0 );
    r.v[3] ^= _mm256_gf2p8affine_epi64_epi8( d1 , _dec_1_3 , 0 );
    r.v[0] ^= _mm256_gf2p8affine_epi64_epi8( d2 , _dec_2_0 , 0 );
    r.v[1] ^= _mm256_gf2p8affine_epi64_epi8( d2 , _dec_2_1 , 0 );
    r.v[2] ^= _mm256_gf2p8affine_epi64_epi8( d2 , _dec_2_2 , 0 );
    r.v[3] ^= _mm256_gf2p8affine_epi64_epi8( d2 , _dec_2_3 , 0 );
    r.v[0] ^= _mm256_gf2p8affine_epi64_epi8( d3 , _dec_3_0 , 0 );
    r.v[1] ^= _mm256_gf2p8affine_epi64_epi8( d3 , _dec_3_1 , 0 );
    r.v[2] ^= _mm256_gf2p8affine_epi64_epi8( d3 , _dec_3_2 , 0 );
    r.v[3] ^= _mm256_gf2p8affine_epi64_epi8( d3 , _dec_3_3 , 0 );
    return r;
}



// scatter 128 4-byte element to 32 128-bit chunks of blocklen.
static inline void tr_scatter_32_x128( uint8_t * rfx , __m256i * vv , unsigned blocklen )
{
    for(int k=0;k<2;k++){ // process 2 different 128-bit lanes simultaneously.
        __m256i v[8];
        v[0] = _mm256_unpacklo_epi64( vv[k*8+0] , vv[k*8+4] );
        v[1] = _mm256_unpackhi_epi64( vv[k*8+0] , vv[k*8+4] );
        v[2] = _mm256_unpacklo_epi64( vv[k*8+1] , vv[k*8+5] );
        v[3] = _mm256_unpackhi_epi64( vv[k*8+1] , vv[k*8+5] );
        v[4] = _mm256_unpacklo_epi64( vv[k*8+2] , vv[k*8+6] );
        v[5] = _mm256_unpackhi_epi64( vv[k*8+2] , vv[k*8+6] );
        v[6] = _mm256_unpacklo_epi64( vv[k*8+3] , vv[k*8+7] );
        v[7] = _mm256_unpackhi_epi64( vv[k*8+3] , vv[k*8+7] );
        _byte_transpose_8x8_x4( v );
        _bit_transpose_8x8_x32( v );

        // scatter.
        for(int j=0;j<8;j++){
            __m128i j_l = _mm256_castsi256_si128( v[j]);
            __m128i j_h = _mm256_extracti128_si256( v[j], 1 );
            _mm_storeu_si128( (__m128i*)(rfx+((k*16 + j)  *blocklen)) , j_l );
            _mm_storeu_si128( (__m128i*)(rfx+((k*16 + 8 + j)*blocklen)) , j_h );
        }
    }
}

static
void decode_gf256t4_gfni( uint32_t * bitvec , const uint8_t * v0, const uint8_t * v1, const uint8_t * v2, const uint8_t * v3, unsigned n )
{
    unsigned blocklen = n/8; // Total vector len: n*4 bytes. one block len : total_len/32 = n/8
    __m256i temp[4*4];
    for(unsigned i=0;i<n;i+=128){
        for(int j=0;j<4;j++){
            ymm_x4 t = decode_core(_mm256_loadu_si256((const __m256i*)(v0+i+j*32)),_mm256_loadu_si256((const __m256i*)(v1+i+j*32)),_mm256_loadu_si256((const __m256i*)(v2+i+j*32)),_mm256_loadu_si256((const __m256i*)(v3+i+j*32)));
            // store vectors of 0-byte and 1-byte in different 128-bit lanes
            temp[j*2 ]  = _mm256_permute2x128_si256( t.v[0] , t.v[1] , 0x20 );
            temp[j*2+1] = _mm256_permute2x128_si256( t.v[0] , t.v[1] , 0x31 );
            temp[j*2+8] = _mm256_permute2x128_si256( t.v[2] , t.v[3] , 0x20 );
            temp[j*2+9] = _mm256_permute2x128_si256( t.v[2] , t.v[3] , 0x31 );
        }

        // scatter 128 4-byte element to 32 128-bit chunks of blocklen.
        tr_scatter_32_x128( ((uint8_t*)bitvec)+(i/8) , temp , blocklen );   // i/8 = (i/128)*16
    }
}

///////////////////////////////////////////////

/// @brief encode bit-vector input of size (n*src_bits) bits into n gf256t4 elements
/// @param v0 0-th byte of gf256t4 vector 
/// @param v1 1-st byte of gf256t4 vector
/// @param v2 2-nd byte of gf256t4 vector
/// @param v3 3-rd byte of gf256t4 vector
/// @param n number of output gf256t4 elements
/// @param bitvec input of a bit vector
/// @param src_bits number of meaningful bits of gf256t4 elements, optimizing for src_bits=16.
void encode_gf256t4( uint8_t * v0 , uint8_t * v1, uint8_t * v2, uint8_t * v3, unsigned n, const uint32_t * bitvec , unsigned src_bits )
{
    if ( (0==(n&0x7f)) && (16==src_bits)    ) {
        encode_gf256t4_gfni(v0,v1,v2,v3,n,bitvec);
        //encode_gf256t4_ref(v0,v1,v2,v3,n,bitvec,16);
        return;
    }
    encode_gf256t4_ref(v0,v1,v2,v3,n,bitvec,src_bits);
}

/// @brief reverse operation of encode_gf256t4()
/// @param bitvec output of a bit vector
/// @param v0 0-th byte of gf256t4 vector
/// @param v1 1-st byte of gf256t4 vector
/// @param v2 2-nd byte of gf256t4 vector
/// @param v3 3-rd byte of gf256t4 vector
/// @param n number of input gf256t4 elements, equal to (n*32)-bit output.
void decode_gf256t4( uint32_t * bitvec , const uint8_t * v0, const uint8_t * v1, const uint8_t * v2, const uint8_t * v3, unsigned n )
{
    if ( 0==(n&0x7f) ) {
        decode_gf256t4_gfni(bitvec,v0,v1,v2,v3,n);
        return;
    }
    decode_gf256t4_ref(bitvec,v0,v1,v2,v3,n);
}

//////////////////////////////////////////////


