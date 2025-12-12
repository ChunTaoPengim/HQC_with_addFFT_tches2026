
#include "dencoder.h"

static uint64_t _enc_mat[64] = {
1,0xed57ce778f0d6244,0xb66864e6ec14b4d2,0x9cedebc39773a213,
0x54e5bf3774b3f850,0xaa41ec72aca4d601,0x4943d209fdc449b8,0x6e9a7c0e6d89dc84,
0xeb42e79f91f49f8c,0x5dc314ada2848ba2,0x503cdedab981d32c,0x21d93ab94919cfd3,
0x4553b41bda2fbec8,0xcb47a62d21df0b4d,0x4639691aa527dbba,0xe032e7a43c3d150f,
0xb0f502e4cd60039a,0xf8287767660ad2b4,0x1dbf8d7727fc654e,0x19b12a51dacba79e,
0xe691d7c0794c614d,0x28cf418faf97a29f,0xfc5c2bca45c931c,0x4cb433e14292028e,
0x189b29840c130e7f,0x315f35290c617710,0x1f4a9de7438026b,0x6aea5df783dd08ef,
0xb36ae8e530273635,0x837b54ac6006165d,0x49216693bb420e26,0xfd50e246ba0fe1a8,
0x57a3104fcd0e5f34,0xbfbbcc4e6e818ce9,0x4ff85cdbb8b5a385,0xe5836ac7d88636ff,
0xacaee33d357e7115,0xd5591c1e1fcd94ac,0x5a80ddb4eb880892,0x557feec300cdaab7,
0x5a3fd6353ee07e37,0x5c467f7d2caf0e9f,0xeaac97005f61500b,0x9554e988c2331d19,
0x46fc62831b042eeb,0xe89bd07ebe5cbadc,0xe442a47d4ab744e,0x492fcd9722316b93,
0xb86431a85814f803,0xa3c96aff342c82b8,0x4716ccf6cbc75a1a,0x5b71bea40e882374,
0x54ac9ee9b21ebe6e,0x1aac7f922b29c823,0x49a5f74a2ecb3e44,0x4a5a968cc3b566d1,
0x641ddcf92a60aa5,0xee004d8f989dba16,0x1c2242e610c9cd6a,0x59f1654505010e32,
0xd8df39e76c0e7b7,0x8cf8e69eb14ecfb4,0xcb8c5396572fc44,0x3b278b84a916e44c,
};

static uint64_t _dec_mat[64] = {
1,0x8961b8a7841bfc6,0xfd56dc6306e45ec6,0x77332469d0404105,
0xd110eb4a970feaf5,0x9c94931de38226fe,0x52bf8c5ba0978896,0xbce2b823bebf0166,
0x7af3356b5d818e41,0x1e7b94c935b698d0,0xbcb73d9aabdd6ddf,0x5f75970481f366a8,
0xb33f7074d9b6e3fe,0x2adb64d29a93ef96,0x59dab92a69394e60,0x3c8004790328459c,
0xadcb0001cdaa3d05,0x9bb38fd76d5e0c0b,0xabcadd426cb54b4d,0xf042a8f93e8a48b9,
0x47b25aab146986f6,0xe750955be42931a1,0x9bf1d4a1535a0d17,0x91564f9a406d933c,
0xded3254b42d33aaa,0xbd1f32c59e84e45d,0xfa018660e3165396,0xa2f911a00f5baa7b,
0xe958695bc8926acd,0xc8c10606a8ec8d6b,0x8e6cfef54b2e9ac6,0x88007d210e507cf4,
0x17a89b9796a48397,0x1bdd636a99f218be,0x7cb821d83481a449,0xff39584957728d46,
0x9cd7c0f7b4d02de7,0x779fca946b676edb,0x5235a7728b715838,0x70ef0482c08984a5,
0xbb1c3dbfe2b5150b,0xee767356051fe0e1,0x1b5c98d75cc93fe8,0x4c25ee1fb1c8ceb6,
0xb7310079067f9e51,0x4be2a5ae359e2594,0xfe86864026631c3,0x5b4cd59958434683,
0xe550b2539a848707,0x6e50d13b6a7a5997,0x7fd8fb00a5aef99a,0xcdbd262ac16e806b,
0xc021427b4f7646e0,0xd7f262605a9bcdd0,0xb046630a7efeddc,0xa869f3dfcbe24fab,
0xa00150323be9b1de,0x1294d8724ae3d0d,0x4ab517f07f03a4b1,0xfdc5af3bb370e266,
0x2aff63efa8285238,0xfba0ace42227357a,0xc208107e1c3e5ac1,0xb2dbb5cb68d0dc74,
};



static inline
void collect_inputbits_x8( uint8_t* dest , const uint8_t * src , unsigned src_bits , unsigned offset )
{
    for(unsigned j=0;j<src_bits;j++){
        dest[j] = src[j*offset];
    }
}

static inline
void linear_map_x8( uint64_t * rrx8 , const uint8_t * inbits , unsigned src_bits , const uint64_t * mat )
{
    uint64_t rr[8] = {0};
    for(unsigned j=0;j<src_bits;j++){
        uint8_t v = inbits[j];
        for(int k=0;k<8;k++) rr[k] ^= (-((v>>k)&1))&mat[j];
    }
    for(int k=0;k<8;k++) {
        rrx8[k] = rr[k];
    }
}


// sizeof output = n_u64 x 64 bits  // assert(n_u64>=8)
void encode_64( uint64_t * rfx , unsigned n_u64, const uint64_t * fx , unsigned src_bits )
{
    unsigned batch = n_u64/8;  // process 8 elements at a time
    for(unsigned i=0;i<batch;i++){ // process 8 elements at a time
        uint8_t inbits[64];
        collect_inputbits_x8( inbits , ((const uint8_t *)fx)+i , src_bits , batch );

        linear_map_x8( rfx+i*8 , inbits , src_bits , _enc_mat );
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


// sizeof output = sizeof input // assert(n_u64>=8)
void decode_64( uint64_t * rfx , const uint64_t * fx , unsigned n_u64 )
{
    unsigned step = n_u64/8;
    for(unsigned i=0;i<step;i++){  // processing 8 elements for one iteration
        const uint64_t * inp = fx+i*8;
        uint64_t rr[8] = {0};
        for(int j=0;j<64;j++) {
            for(int k=0;k<8;k++) { rr[k] ^= (-((inp[k]>>j)&1))&_dec_mat[j]; }  // linear map for 8 elements
        }

        // output
        uint8_t * ptr = ((uint8_t *)rfx)+i;
        for(int k=0;k<8;k++) {
            uint8_t mat0[8];
            uint8_t * p8 = ((uint8_t*)&rr[0])+k;  // processing k-th byte of 64-bit elements
            for(int j=0;j<8;j++) { mat0[j] = p8[j*8]; }
            transpose_8x8(mat0);
            for(int j=0;j<8;j++){
                ptr[0] = mat0[j];
                ptr += step;
            }
        }
    }
}


