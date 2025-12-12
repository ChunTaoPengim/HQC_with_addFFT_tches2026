
#include "dencoder_gf256t4.h"

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

/// @brief encode bit-vector input of size (n*src_bits) bits into n gf256t4 elements
/// @param v0 0-th byte of gf256t4 vector 
/// @param v1 1-st byte of gf256t4 vector
/// @param v2 2-nd byte of gf256t4 vector
/// @param v3 3-rd byte of gf256t4 vector
/// @param n number of output gf256t4 elements
/// @param bitvec input of a bit vector
/// @param src_bits number of meaningful bits of gf256t4 elements, optimizing for src_bits=16.
void encode_gf256t4( uint8_t * v0 , uint8_t * v1, uint8_t * v2, uint8_t * v3, unsigned n,
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

/// @brief reverse operation of encode_gf256t4()
/// @param bitvec output of a bit vector
/// @param v0 0-th byte of gf256t4 vector
/// @param v1 1-st byte of gf256t4 vector
/// @param v2 2-nd byte of gf256t4 vector
/// @param v3 3-rd byte of gf256t4 vector
/// @param n number of input gf256t4 elements, equal to (n*32)-bit output.
void decode_gf256t4( uint32_t * bitvec 
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

