#ifndef _COMBINE_H_
#define _COMBINE_H_


#include <stdint.h>


#ifdef  __cplusplus
extern  "C" {
#endif

/// @brief combine two polynomials in F216[X]/1024 AND F216[X]/s13
/// @param c0 [out] low  byte of F216. size : 9215 bytes
/// @param c1 [out] high byte of F216. size : 9215 bytes
void ringmul_combine_9215( uint8_t * out , const uint8_t * pc_m1024_low , const uint8_t * pc_m1024_high, 
    const uint8_t *c0, const uint8_t *c1);
void ringmul_combine_4479( uint8_t * out , const uint8_t * pc_m384_low , const uint8_t * pc_m384_high,
    const uint8_t *c0, const uint8_t *c1);
 
#ifdef  __cplusplus
}
#endif


#endif
