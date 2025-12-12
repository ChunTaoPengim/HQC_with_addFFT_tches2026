
#include <stdint.h>


static inline
int choose_si( int loglen )
{
  int si = 1<<0;
  for( int i=1; (1<<i) < loglen ; i++ ) {
    si = 1<<i;
  }
  return si;
}

/////////////////

static inline
void div_blk( uint32_t *poly, int si_h, int si_l, int polylen )
{
  int deg_diff = si_h-si_l;
  for(int i=polylen-1;i>=si_h;i--) {
      poly[(i-deg_diff)*8]   ^= poly[i*8];
      poly[(i-deg_diff)*8+1] ^= poly[i*8+1];
      poly[(i-deg_diff)*8+2] ^= poly[i*8+2];
      poly[(i-deg_diff)*8+3] ^= poly[i*8+3];
      poly[(i-deg_diff)*8+4] ^= poly[i*8+4];
      poly[(i-deg_diff)*8+5] ^= poly[i*8+5];
      poly[(i-deg_diff)*8+6] ^= poly[i*8+6];
      poly[(i-deg_diff)*8+7] ^= poly[i*8+7];
  }
}

static
void rep_in_si( uint32_t *data, int datalen, int logsize_blk, int polyloglen_blk,  int si  )
{
  for(int i=polyloglen_blk-1;i>=si;i--) {
    int polylen = (1<<(i+logsize_blk+1));
    int si_h = (1<<(i+logsize_blk));
    int si_l = (1<<(i+logsize_blk-si));
    for(int j=0;j<datalen;j+=polylen) div_blk( data+j*8 , si_h, si_l, polylen );
  }
}

static
void cvt( uint32_t *data, int datalen, int logsize_blk, int polyloglen_blk )
{
  if( 1 >= polyloglen_blk ) return;
  int si = choose_si(polyloglen_blk);
  rep_in_si( data, datalen , logsize_blk , polyloglen_blk , si );

  cvt( data , datalen , logsize_blk , si );
  cvt( data , datalen , logsize_blk+si , polyloglen_blk-si );
}


////////

static inline
void idiv_blk( uint32_t *poly, int si_h, int si_l, int polylen )
{
  int deg_diff = si_h-si_l;
  for(int i=si_h;i<polylen;i++) {
      poly[(i-deg_diff)*8]   ^= poly[i*8];
      poly[(i-deg_diff)*8+1] ^= poly[i*8+1];
      poly[(i-deg_diff)*8+2] ^= poly[i*8+2];
      poly[(i-deg_diff)*8+3] ^= poly[i*8+3];
      poly[(i-deg_diff)*8+4] ^= poly[i*8+4];
      poly[(i-deg_diff)*8+5] ^= poly[i*8+5];
      poly[(i-deg_diff)*8+6] ^= poly[i*8+6];
      poly[(i-deg_diff)*8+7] ^= poly[i*8+7];
  }
}

static
void irep_in_si( uint32_t *data, int datalen, int logsize_blk, int polyloglen_blk,  int si  )
{
  for(int i=si;i<polyloglen_blk;i++) {
    int polylen = (1<<(i+logsize_blk+1));
    int si_h = (1<<(i+logsize_blk));
    int si_l = (1<<(i+logsize_blk-si));
    for(int j=0;j<datalen;j+=polylen) idiv_blk( data+j*8 , si_h, si_l, polylen );
  }
}

static
void icvt( uint32_t *data, int datalen, int logsize_blk, int polyloglen_blk )
{
  if( 1 >= polyloglen_blk ) return;
  int si = choose_si(polyloglen_blk);

  icvt( data , datalen , logsize_blk+si , polyloglen_blk-si );
  icvt( data , datalen , logsize_blk , si );

  irep_in_si( data, datalen , logsize_blk , polyloglen_blk , si );
}








/////////////////////////////////////////
//
// Optimized version for n_256 = 64, 128, 256
//
/////////////////////////////////////////



static
void bc_256_16384( uint32_t *poly )
{
  const unsigned w = 16384;

  uint32_t *ptr;
//layer 6: [16384] s4^2:(8192,512) suggest unit:256
#define DEG_H 8192
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 7: [8192] s4^1:(4096,256) suggest unit:256
#define DEG_H 4096
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 8: [16384] s1^1:(8192,4096) suggest unit:4096
#define DEG_H 8192
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 9: [4096] s2^2:(2048,512) suggest unit:256
#define DEG_H 2048
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 10: [2048] s2^1:(1024,256) suggest unit:256
#define DEG_H 1024
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 11: [4096] s1^1:(2048,1024) suggest unit:1024
#define DEG_H 2048
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 12: [1024] s1^1:(512,256) suggest unit:256
#define DEG_H 512
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

}


static
void ibc_256_16384( uint32_t *poly )
{
  const unsigned w = 16384;

  uint32_t *ptr;

//layer 12: [1024] s1^1:(512,256) suggest unit:256
#define DEG_H 512
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 11: [4096] s1^1:(2048,1024) suggest unit:1024
#define DEG_H 2048
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 10: [2048] s2^1:(1024,256) suggest unit:256
#define DEG_H 1024
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 9: [4096] s2^2:(2048,512) suggest unit:256
#define DEG_H 2048
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 8: [16384] s1^1:(8192,4096) suggest unit:4096
#define DEG_H 8192
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 7: [8192] s4^1:(4096,256) suggest unit:256
#define DEG_H 4096
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 6: [16384] s4^2:(8192,512) suggest unit:256
#define DEG_H 8192
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

}




static
void bc_256_32768( uint32_t *poly )
{
  const unsigned w = 32768;

  uint32_t *ptr;
//layer 7: [32768] s4^4:(16384,1024) suggest unit:256
#define DEG_H 16384
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 8: [16384] s4^2:(8192,512) suggest unit:256
#define DEG_H 8192
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 9: [8192] s4^1:(4096,256) suggest unit:256
#define DEG_H 4096
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 10: [32768] s2^1:(16384,4096) suggest unit:4096
#define DEG_H 16384
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 11: [16384] s1^1:(8192,4096) suggest unit:4096
#define DEG_H 8192
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 12: [4096] s2^2:(2048,512) suggest unit:256
#define DEG_H 2048
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 13: [2048] s2^1:(1024,256) suggest unit:256
#define DEG_H 1024
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 14: [4096] s1^1:(2048,1024) suggest unit:1024
#define DEG_H 2048
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 15: [1024] s1^1:(512,256) suggest unit:256
#define DEG_H 512
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L
}


static
void ibc_256_32768( uint32_t *poly )
{
  const unsigned w = 32768;

  uint32_t *ptr;

//layer 15: [1024] s1^1:(512,256) suggest unit:256
#define DEG_H 512
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 14: [4096] s1^1:(2048,1024) suggest unit:1024
#define DEG_H 2048
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 13: [2048] s2^1:(1024,256) suggest unit:256
#define DEG_H 1024
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 12: [4096] s2^2:(2048,512) suggest unit:256
#define DEG_H 2048
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 11: [16384] s1^1:(8192,4096) suggest unit:4096
#define DEG_H 8192
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 10: [32768] s2^1:(16384,4096) suggest unit:4096
#define DEG_H 16384
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 9: [8192] s4^1:(4096,256) suggest unit:256
#define DEG_H 4096
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 8: [16384] s4^2:(8192,512) suggest unit:256
#define DEG_H 8192
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 7: [32768] s4^4:(16384,1024) suggest unit:256
#define DEG_H 16384
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

}



static
void bc_256_65536( uint32_t *poly )
{
  const unsigned w = 65536;

  uint32_t *ptr;
//layer 8: [65536] s4^8:(32768,2048) suggest unit:256
#define DEG_H 32768
#define DEG_L 2048
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 9: [32768] s4^4:(16384,1024) suggest unit:256
#define DEG_H 16384
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 10: [16384] s4^2:(8192,512) suggest unit:256
#define DEG_H 8192
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 11: [8192] s4^1:(4096,256) suggest unit:256
#define DEG_H 4096
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 12: [65536] s2^2:(32768,8192) suggest unit:4096
#define DEG_H 32768
#define DEG_L 8192
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 13: [32768] s2^1:(16384,4096) suggest unit:4096
#define DEG_H 16384
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 14: [65536] s1^1:(32768,16384) suggest unit:16384
#define DEG_H 32768
#define DEG_L 16384
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 15: [16384] s1^1:(8192,4096) suggest unit:4096
#define DEG_H 8192
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 16: [4096] s2^2:(2048,512) suggest unit:256
#define DEG_H 2048
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 17: [2048] s2^1:(1024,256) suggest unit:256
#define DEG_H 1024
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 18: [4096] s1^1:(2048,1024) suggest unit:1024
#define DEG_H 2048
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 19: [1024] s1^1:(512,256) suggest unit:256
#define DEG_H 512
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=(DEG_H/32)-4;i>=0;i-=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L
}

static
void ibc_256_65536( uint32_t *poly )
{
  const unsigned w = 65536;
  uint32_t *ptr;

//layer 19: [1024] s1^1:(512,256) suggest unit:256
#define DEG_H 512
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 18: [4096] s1^1:(2048,1024) suggest unit:1024
#define DEG_H 2048
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 17: [2048] s2^1:(1024,256) suggest unit:256
#define DEG_H 1024
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 16: [4096] s2^2:(2048,512) suggest unit:256
#define DEG_H 2048
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 15: [16384] s1^1:(8192,4096) suggest unit:4096
#define DEG_H 8192
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 14: [65536] s1^1:(32768,16384) suggest unit:16384
#define DEG_H 32768
#define DEG_L 16384
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 13: [32768] s2^1:(16384,4096) suggest unit:4096
#define DEG_H 16384
#define DEG_L 4096
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 12: [65536] s2^2:(32768,8192) suggest unit:4096
#define DEG_H 32768
#define DEG_L 8192
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 11: [8192] s4^1:(4096,256) suggest unit:256
#define DEG_H 4096
#define DEG_L 256
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 10: [16384] s4^2:(8192,512) suggest unit:256
#define DEG_H 8192
#define DEG_L 512
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 9: [32768] s4^4:(16384,1024) suggest unit:256
#define DEG_H 16384
#define DEG_L 1024
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L

//layer 8: [65536] s4^8:(32768,2048) suggest unit:256
#define DEG_H 32768
#define DEG_L 2048
  ptr = poly;
  for(int j=(w/(DEG_H*2));j>0;j--) {
    for(int i=0;i<(DEG_H/32);i+=4) {
      uint32_t p0 = ptr[i+(DEG_H/32)];
      uint32_t p1 = ptr[i+(DEG_H/32)+1];
      uint32_t p2 = ptr[i+(DEG_H/32)+2];
      uint32_t p3 = ptr[i+(DEG_H/32)+3];
      uint32_t q0 = ptr[i+(DEG_L/32)];
      uint32_t q1 = ptr[i+(DEG_L/32)+1];
      uint32_t q2 = ptr[i+(DEG_L/32)+2];
      uint32_t q3 = ptr[i+(DEG_L/32)+3];
      ptr[i+(DEG_L/32)] = q0^p0;
      ptr[i+(DEG_L/32)+1] = q1^p1;
      ptr[i+(DEG_L/32)+2] = q2^p2;
      ptr[i+(DEG_L/32)+3] = q3^p3;
    }
    ptr += (DEG_H*2/32);
  }
#undef DEG_H
#undef DEG_L
}



#include "bc_256.h"

void bc_256( void * poly, unsigned n_256 )
{
  if (64==n_256){
    bc_256_16384( (uint32_t *)poly );
  } else if (128==n_256){
    bc_256_32768( (uint32_t *)poly );
  } else if (256==n_256){
    bc_256_65536( (uint32_t *)poly );
  } else {
    cvt( (uint32_t *)poly , n_256 , 0 , __builtin_ctz(n_256)  );
  }
}

void ibc_256( void * poly, unsigned n_256 )
{
  if (64==n_256){
    ibc_256_16384( (uint32_t *)poly );
  } else if (128==n_256){
    ibc_256_32768( (uint32_t *)poly );
  } else if (256==n_256){
    ibc_256_65536( (uint32_t *)poly );
  } else {
    icvt( (uint32_t *)poly , n_256 , 0 , __builtin_ctz(n_256)  );
  }
}



