
#ifndef _BTFY_H_
#define _BTFY_H_


#include <stdint.h>


#ifdef  __cplusplus
extern  "C" {
#endif



void btfy_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset );

void ibtfy_gf256t4( uint8_t * v0 , uint8_t * v1 , uint8_t * v2 , uint8_t * v3 , unsigned n_stage , uint32_t idx_offset );



#ifdef  __cplusplus
}
#endif


#endif
