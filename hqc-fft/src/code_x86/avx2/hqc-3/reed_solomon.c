/**
 * @file reed_solomon.c
 * @brief Constant time implementation of Reed-Solomon codes
 */

#include "reed_solomon.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "crypto_memset.h"
#include "fft.h"
#include "gf.h"
#include "parameters.h"
#ifdef VERBOSE
#include <stdbool.h>
#include <stdio.h>
#endif

/**
 * @brief Number of 16-bit words required to hold a syndrome of length 2·PARAM_DELTA bits
 *
 * For the 256-bit implementation, each word is 16 bits, so we round
 * up the total bit-length (2·PARAM_DELTA) to the next multiple of 16 bits.
 */
#define SYND_SIZE_256 (CEIL_DIVIDE(2 * PARAM_DELTA, 16))

static uint16_t mod(uint16_t i, uint16_t modulus);
static void compute_syndromes(__m256i *syndromes, uint8_t *cdw);
static uint16_t compute_elp(uint16_t *sigma, const uint16_t *syndromes);
static void compute_roots(uint8_t *error, uint16_t *sigma);
static void compute_z_poly(uint16_t *z, const uint16_t *sigma, uint16_t degree, const uint16_t *syndromes);
static void compute_error_values(uint16_t *error_values, const uint16_t *z, const uint8_t *error);
static void correct_errors(uint8_t *cdw, const uint16_t *error_values);

/**
 * @brief Precomputed 256-bit vectors of field elements αᵢʲ for GF(2^8) operations (first half).
 *
 * Each entry packs four 64-bit words into a __m256i, corresponding to the
 * powers of the primitive element α raised to the exponents i·j.
 *
 * The table length (45) matches the number of coefficients needed in the
 * αᵢʲ power table for the chosen code parameters.
 */
static const __m256i alpha_ij256_1[55] = {
    {0x0010000800040002, 0x001d008000400020, 0x00cd00e80074003a, 0x004c002600130087},
    {0x001d004000100004, 0x004c001300cd0074, 0x008f00ea00b4002d, 0x009d006000180006},
    {0x00cd003a00400008, 0x008f0075002d0026, 0x002500270060000c, 0x004600c100b50035},
    {0x004c00cd001d0010, 0x009d0018008f00b4, 0x004600ee006a0025, 0x005f00b9005d0014},
    {0x00b4002600740020, 0x006a009c00600003, 0x00b900a0000500c1, 0x00fd000f005e00be},
    {0x008f002d00cd0040, 0x004600b500250060, 0x0065006100b90050, 0x00d900df006b0078},
    {0x0018007500130080, 0x005d008c00b5009c, 0x006b003c005e00a1, 0x0081001a004300a3},
    {0x009d008f004c001d, 0x005f005d0046006a, 0x00d900fe00fd0065, 0x0085003b0081000d},
    {0x0025000c002d003a, 0x006500a1005000c1, 0x00d0008600df00e7, 0x00a800a9006600ed},
    {0x006a006000b40074, 0x00fd005e00b90005, 0x003b0067001100df, 0x00e600550084002e},
    {0x00ee002700ea00e8, 0x00fe003c006100a0, 0x00b8007600670086, 0x00e3009100390054},
    {0x00460025008f00cd, 0x00d9006b006500b9, 0x00a800b8003b00d0, 0x0082009600fc00e4},
    {0x0014003500060087, 0x000d00a3007800be, 0x00e40054002e00ed, 0x00510064006200e5},
    {0x005d00b500180013, 0x00810043006b005e, 0x00fc003900840066, 0x0012005900c80062},
    {0x00b900c100600026, 0x003b001a00df000f, 0x00960091005500a9, 0x002c002400590064},
    {0x005f0046009d004c, 0x0085008100d900fd, 0x008200e300e600a8, 0x0002002c00120051},
    {0x0099000a004e0098, 0x004f0093004400d6, 0x00dd00dc00d70092, 0x00980001000b0045},
    {0x006500500025002d, 0x00a8006600d000df, 0x00c30007009600bf, 0x0027002600ad00fb},
    {0x001e00ba0094005a, 0x0049006d003e00e2, 0x003d00a200ae00b3, 0x008c006000e80083},
    {0x00fd00b9006a00b4, 0x00e60084003b0011, 0x002c00ac001c0096, 0x00be00c100030020},
    {0x006b00a100b50075, 0x00fc00290066001a, 0x00ad00f500590057, 0x00e700b90035002d},
    {0x00fe006100ee00ea, 0x00e3003900b80067, 0x003a00b000ac0007, 0x00af000f002800c0},
    {0x005b002f009f00c9, 0x009500d10021007c, 0x0075004700f400a6, 0x001f00df00c200ee},
    {0x00d900650046008f, 0x008200fc00a8003b, 0x0027003a002c00c3, 0x0017001a00e700ba},
    {0x0011000f00050003, 0x001c00ff00550033, 0x00c100b4006c0024, 0x004d003b00e2005e},
    {0x000d007800140006, 0x0051006200e4002e, 0x00ba00c0002000fb, 0x00d100a900bd00bb},
    {0x00d000e70050000c, 0x00c3005700bf00a9, 0x002f00b50026007d, 0x00db005500c500d9},
    {0x0081006b005d0018, 0x001200c800fc0084, 0x00e70028000300ad, 0x00190091009e00bd},
    {0x00f8007f00690030, 0x00f700e000f1004d, 0x00b6005f009c0040, 0x00a2009600aa00ec},
    {0x003b00df00b90060, 0x002c005900960055, 0x001a000f00c10026, 0x00240064009100a9},
    {0x009700b600de00c0, 0x001b009b006e0072, 0x00ed00b100a0008f, 0x00580059004b0052},
    {0x008500d9005f009d, 0x00020012008200e6, 0x001700af00be0027, 0x00040024001900d1},
    {0x00b8008600610027, 0x003a00f500070091, 0x001500d0000f00b5, 0x002d002c00a600f1},
    {0x004f00440099004e, 0x0098000b00dd00d7, 0x0092009300d6000a, 0x004e0001004500dc},
    {0x0084001a005e009c, 0x000300e9005900ff, 0x0091002e00e200b9, 0x0005002600eb001c},
    {0x00a800d000650025, 0x002700ad00c30096, 0x00db0015001a002f, 0x00610060003600f2},
    {0x005200ce0089004a, 0x00d40010008a0037, 0x00570049007c0078, 0x00d300c1001d0048},
    {0x0049003e001e0094, 0x008c00e8003d00ae, 0x003800630033007f, 0x004300b900ea0016},
    {0x00e400ed00780035, 0x00ba002d00fb0064, 0x00f200f100a900d9, 0x003e000f002500ad},
    {0x00e6003b00fd006a, 0x00be0003002c001c, 0x00240037004d001a, 0x002e00df00050074},
    {0x00c600c500d300d4, 0x00ca009d00cf00a7, 0x008b00c80072003e, 0x009a001a005f00c9},
    {0x00fc0066006b00b5, 0x00e7003500ad0059, 0x003600a6009100c5, 0x00bf003b00780025},
    {0x007b001700b10077, 0x00e1009f000800ef, 0x0040002b00ff00b8, 0x00ab00a9005b008c},
    {0x00e300b800fe00ee, 0x00af0028003a00ac, 0x002d007a00370015, 0x00320055003400de},
    {0x009600a900df00c1, 0x001a00b900260024, 0x0060002c00640055, 0x00590091003b000f},
    {0x00950021005b009f, 0x001f00c2007500f4, 0x00b500d800a70073, 0x0048009600da00fe},
    {0x00a5001500710023, 0x00760089000c00eb, 0x0050008000ef00fc, 0x00b0006400520022},
    {0x008200a800d90046, 0x001700e70027002c, 0x0061002d002400db, 0x0008005900bf003e},
    {0x00c800290043008c, 0x009e00fe003500e9, 0x0078003000eb006e, 0x005a002400e300cc},
    {0x001c005500110005, 0x004d00e200c1006c, 0x00df006a00e90064, 0x009c002c00ae0084},
    {0x00dd00920044000a, 0x00920044000a0001, 0x0044000a000100dd, 0x000a000100dd0092},
    {0x005100e4000d0014, 0x00d100bd00ba0020, 0x003e00de007400f2, 0x00c20026002b003f},
    {0x0079007300340028, 0x00e500f800a10074, 0x006600ca00b4008a, 0x00bb006000f7004b},
    {0x00c300bf00d00050, 0x00db00c5002f0026, 0x0021006b006000f5, 0x008600c100cf0082},
    {0x00ac0091006700a0, 0x0037002e000f00b4, 0x005500e2006a002c, 0x007c00b9002000a7}};

/**
 * @brief Precomputed 256-bit vectors of field elements αᵢʲ for GF(2^8) operations (second half).
 *
 * Same format as alpha_ij256_1, providing the remaining set of powers required
 * for optimized Reed–Solomon arithmetic on 256-bit words with AVX2.
 */
static const __m256i alpha_ij256_2[55] = {
    {0x00b4005a002d0098, 0x008f00c900ea0075, 0x0018000c00060003, 0x009d00c000600030},
    {0x006a00940025004e, 0x0046009f00ee00b5, 0x005d005000140005, 0x005f00de00b90069},
    {0x00b900ba0050000a, 0x0065002f006100a1, 0x006b00e70078000f, 0x00d900b600df007f},
    {0x00fd001e00650099, 0x00d9005b00fe006b, 0x008100d0000d0011, 0x00850097003b00f8},
    {0x001100e200df00d6, 0x003b007c0067001a, 0x008400a9002e0033, 0x00e600720055004d},
    {0x003b003e00d00044, 0x00a8002100b80066, 0x00fc00bf00e40055, 0x0082006e009600f1},
    {0x0084006d00660093, 0x00fc00d100390029, 0x00c80057006200ff, 0x0012009b005900e0},
    {0x00e6004900a8004f, 0x0082009500e300fc, 0x001200c30051001c, 0x0002001b002c00f7},
    {0x009600b300bf0092, 0x00c300a600070057, 0x00ad007d00fb0024, 0x0027008f00260040},
    {0x001c00ae009600d7, 0x002c00f400ac0059, 0x000300260020006c, 0x00be00a000c1009c},
    {0x00ac00a2000700dc, 0x003a004700b000f5, 0x002800b500c000b4, 0x00af00b1000f005f},
    {0x002c003d00c300dd, 0x00270075003a00ad, 0x00e7002f00ba00c1, 0x001700ed001a00b6},
    {0x0020008300fb0045, 0x00ba00ee00c0002d, 0x00bd00d900bb005e, 0x00d1005200a900ec},
    {0x000300e800ad000b, 0x00e700c200280035, 0x009e00c500bd00e2, 0x0019004b009100aa},
    {0x00c1006000260001, 0x001a00df000f00b9, 0x0091005500a9003b, 0x0024005900640096},
    {0x00be008c00270098, 0x0017001f00af00e7, 0x001900db00d1004d, 0x00040058002400a2},
    {0x00d60099000a004e, 0x0092004f00930044, 0x004500dd00dc00d7, 0x004e00980001000b},
    {0x001a007f002f000a, 0x00db0073001500c5, 0x003600f500f20064, 0x00610046006000cd},
    {0x00330034007f0099, 0x00380062006300a8, 0x00ea0008001600ac, 0x004300f000b900d4},
    {0x004d0033001a00d6, 0x002400a700370091, 0x00050060007400e9, 0x002e006700df005e},
    {0x009100a800c50044, 0x0036003d00a6006e, 0x007800ba00250026, 0x00bf0015003b0086},
    {0x0037006300150093, 0x002d00d8007a00a6, 0x0034006b00de006a, 0x0032007b00550085},
    {0x00a700620073004f, 0x00b5005a00d8003d, 0x00da00ce00fe00be, 0x004800e0009600d5},
    {0x0024003800db0092, 0x006100b5002d0036, 0x00bf0021003e00df, 0x000800fb0059006e},
    {0x00e900ac006400d7, 0x00df00be006a0026, 0x00ae00910084007c, 0x009c0074002c00ef},
    {0x0074001600f200dc, 0x003e00fe00de0025, 0x002b0082003f0084, 0x00c200d4002600fa},
    {0x0060000800f500dd, 0x002100ce006b00ba, 0x00cf005600820091, 0x0086006500c1002d},
    {0x000500ea00360045, 0x00bf00da00340078, 0x005a00cf002b00ae, 0x005c0088000f0023},
    {0x005e00d400cd000b, 0x006e00d500850086, 0x0023002d00fa00ef, 0x006300da001a001e},
    {0x00df00b900600001, 0x005900960055003b, 0x000f00c10026002c, 0x0064009100a9001a},
    {0x006700f000460098, 0x00fb00e0007b0015, 0x0088006500d40074, 0x009000c8009100da},
    {0x002e00430061004e, 0x00080048003200bf, 0x005c008600c2009c, 0x0010009000640063},
    {0x005500ed006b000a, 0x000c003600c300c4, 0x0073006600b600b9, 0x0025000800240082},
    {0x00d7004f00440099, 0x000a0098000b00dd, 0x00dc0092009300d6, 0x0099004e00010045},
    {0x00ae0072003b00d6, 0x000f006a00200024, 0x00ef0096004d0067, 0x001100be0060006c},
    {0x005900f100210044, 0x008600a1000c00cf, 0x007d00a600b300a9, 0x00b800d900b9008f},
    {0x00f4001900e40093, 0x00c500b1008c00cd, 0x004c00fb008d00e6, 0x00c600cc00df0028},
    {0x006c007900f1004f, 0x002900bd00bc0027, 0x00ee004000090037, 0x00c800b7003b00d3},
    {0x002600f500820092, 0x00b300b800b60050, 0x0065002700360059, 0x003d0057005500ce},
    {0x009c006c005900d7, 0x00640072007c000f, 0x001100b900b400eb, 0x002000ac00960084},
    {0x00a00013003d00dc, 0x005600ab009e00d9, 0x0085007f009f0020, 0x004a00d8005900e5},
    {0x000f002700cf00dd, 0x007d0038007300ed, 0x00e4003e00650060, 0x002f000c002c0007},
    {0x00e20014003a0045, 0x00cd001200310021, 0x00950015004300a0, 0x0022006900260090},
    {0x007c00bc000c000b, 0x0025008300e00073, 0x007900fc009700fd, 0x006d00e100c10002},
    {0x00a900df00c10001, 0x00b9002600240096, 0x002c00640055001a, 0x0091003b000f0060},
    {0x007200bd00a10098, 0x006b009400830038, 0x0087008a00e3002e, 0x008d00aa001a00d2},
    {0x00ff008500e7004e, 0x00d0006f0013008a, 0x00d4003600700072, 0x007a006200a900fe},
    {0x006400290086000a, 0x00b8006b0025007d, 0x002f0075003d0096, 0x004000f2009100ed},
    {0x00ef003f00ed0099, 0x00e400680069003a, 0x00af0046008e00a7, 0x009400fa0064009a},
    {0x00eb003700a900d6, 0x0096002e00fd0060, 0x0033000f000300f4, 0x005e00b4002400ff},
    {0x000100dd00920044, 0x00dd00920044000a, 0x00920044000a0001, 0x0044000a000100dd},
    {0x00b4000900b30093, 0x003d00e300970065, 0x00310017003c0003, 0x00da00d3006000f3},
    {0x006a00b00057004f, 0x00ad000e009a00b6, 0x00a200e400880005, 0x003f001f00b90080},
    {0x00b9004000a60092, 0x0075008a00fc003e, 0x008b00c40017000f, 0x000700a800df0025},
    {0x00fd0003002400d7, 0x00c100e900ae00a9, 0x0074005900720011, 0x00f400ff003b00be}};

/**
 * Coefficients of polynomial G
 * stored in 256-bit values
 **/
static const __m256i param256[3] = {{0x001800EF00D8002D, 0x0028001B006800FD, 0x00D200A30032006B, 0x009E00E0008600E3},
                                    {0x0001009E000D0077, 0x002B005200A400EE, 0x008E00F600E8000F, 0x00E8001D00BD0032},
                                    {1, 0, 0, 0}};

static const uint8_t gen_mat[24*32] __attribute__((aligned(32)))  = {
0x2d, 0xd8, 0xef, 0x18, 0xfd, 0x68, 0x1b, 0x28, 0x6b, 0x32, 0xa3, 0xd2, 0xe3, 0x86, 0xe0, 0x9e, 0x77, 0xd, 0x9e, 0x1, 0xee, 0xa4, 0x52, 0x2b, 0xf, 0xe8, 0xf6, 0x8e, 0x32, 0xbd, 0x1d, 0xe8, 
0x30, 0xad, 0x90, 0xda, 0xbb, 0x3a, 0x78, 0x44, 0xca, 0xcc, 0x76, 0xfd, 0xe, 0x84, 0x7f, 0xb2, 0xce, 0xb, 0x5f, 0x76, 0xa1, 0x8, 0xd7, 0x28, 0x9a, 0xe5, 0x7d, 0x82, 0x29, 0x9, 0xe7, 0xf7, 
0x6c, 0xd, 0x2f, 0xb0, 0x60, 0xd0, 0x1e, 0x18, 0x2b, 0x79, 0x55, 0x53, 0x6f, 0x10, 0x12, 0x41, 0x6, 0x29, 0x35, 0xa8, 0x3, 0xc7, 0x3b, 0xb3, 0x3c, 0xe7, 0xb0, 0x88, 0x31, 0x98, 0x25, 0x9a, 
0xc6, 0x46, 0x39, 0xde, 0xf8, 0x27, 0x92, 0x10, 0xec, 0xfd, 0xf2, 0xf2, 0x91, 0x86, 0x61, 0xa, 0x16, 0x6a, 0x31, 0xaf, 0x6, 0x69, 0xf2, 0x86, 0xf6, 0xe9, 0xb8, 0xfd, 0x5e, 0x30, 0xa1, 0xf0, 
0xaf, 0xc9, 0x73, 0x51, 0xad, 0x96, 0x42, 0x2a, 0x73, 0xc1, 0x2a, 0xd3, 0xf3, 0x3a, 0x8a, 0xa2, 0xe6, 0xd2, 0xa9, 0xc1, 0x6a, 0x3b, 0xf9, 0x47, 0xbf, 0x29, 0x44, 0xc0, 0xd0, 0xfb, 0x4f, 0x7e, 
0x37, 0xcc, 0x5, 0x17, 0xd2, 0xf7, 0x70, 0xee, 0xf2, 0x47, 0xc9, 0x62, 0x2d, 0x1b, 0x46, 0x6, 0xfb, 0xaa, 0x5e, 0xd7, 0x73, 0x5, 0x82, 0xd7, 0xf7, 0x14, 0xff, 0x7b, 0xf4, 0xa5, 0x4, 0xe4, 
0xf1, 0x7e, 0x34, 0x90, 0xdc, 0xcf, 0x53, 0xd2, 0xc2, 0x10, 0xc4, 0x26, 0x5e, 0x2c, 0x16, 0xd2, 0x8, 0xdb, 0x3e, 0xba, 0xcb, 0x76, 0x89, 0x11, 0x22, 0x89, 0x9d, 0x8d, 0x99, 0x80, 0x63, 0x7a, 
0x83, 0xd5, 0x29, 0x30, 0xc0, 0x3b, 0x45, 0x5f, 0xbb, 0x3e, 0xae, 0xe3, 0x73, 0x94, 0xf7, 0xd8, 0x4a, 0x70, 0x15, 0x44, 0x97, 0xe, 0x9a, 0xb, 0x9d, 0xe, 0xa0, 0xa0, 0x71, 0x22, 0xb, 0x4f, 
0x39, 0x3b, 0xc, 0xef, 0xf3, 0xbd, 0x2c, 0x12, 0xf3, 0xb4, 0xae, 0xca, 0x59, 0x98, 0xff, 0xda, 0x93, 0x66, 0x5d, 0x5a, 0xd2, 0xf7, 0x3e, 0x1c, 0xb9, 0xb4, 0x5e, 0x9, 0xaf, 0x98, 0x8a, 0x22, 
0x93, 0x96, 0x54, 0x1b, 0xde, 0xa2, 0xcc, 0x15, 0x25, 0x99, 0x36, 0x48, 0x20, 0x45, 0x14, 0xf4, 0x72, 0x34, 0x6d, 0x7f, 0x17, 0xbe, 0xc1, 0x61, 0xff, 0x38, 0xee, 0x4f, 0x63, 0xf6, 0x25, 0xb, 
0x2, 0x68, 0x91, 0xbc, 0xba, 0x1, 0x57, 0xe9, 0xd7, 0xfe, 0x10, 0x83, 0x3b, 0xf3, 0x2b, 0x2f, 0xf2, 0xd, 0xf, 0x66, 0x73, 0xaf, 0xe2, 0xf9, 0x8, 0xc9, 0xdc, 0x65, 0x94, 0x38, 0x39, 0x13, 
0x9d, 0x76, 0x12, 0x24, 0xcd, 0xcc, 0x81, 0x95, 0xaa, 0x86, 0xe4, 0xda, 0x2d, 0x24, 0x68, 0x81, 0x95, 0x35, 0xa7, 0x1c, 0xf, 0x10, 0x10, 0x15, 0x18, 0xb, 0x15, 0x5b, 0x34, 0x51, 0xd2, 0x3a, 
0xc, 0xbd, 0x64, 0x58, 0xc5, 0x35, 0xc8, 0x5f, 0x23, 0x4a, 0x97, 0x7d, 0xed, 0xfd, 0x5d, 0xf2, 0x95, 0x8a, 0xaf, 0x9d, 0x34, 0xb8, 0xc5, 0x80, 0x7e, 0xac, 0x69, 0x8, 0xbb, 0xf3, 0xc9, 0x66, 
0xa8, 0xe0, 0xc, 0x5d, 0xb, 0x36, 0xa6, 0x83, 0x6, 0x9d, 0xd1, 0xa0, 0x5e, 0xc9, 0x74, 0x40, 0x17, 0x61, 0x97, 0xc9, 0x4a, 0x80, 0xe2, 0x24, 0xb8, 0xe0, 0x42, 0x5a, 0xb6, 0x50, 0x29, 0x57, 
0xa6, 0x9f, 0x44, 0x97, 0x4e, 0xdf, 0x54, 0x16, 0xae, 0x83, 0x9e, 0xca, 0xc7, 0x79, 0x57, 0xc8, 0xb7, 0x83, 0xdd, 0xc0, 0x3a, 0xf1, 0x53, 0xab, 0x1e, 0xa4, 0x88, 0xe7, 0xdf, 0xc1, 0xdd, 0x35, 
0xba, 0x3a, 0x11, 0x86, 0x4, 0xd4, 0x42, 0xf, 0xd3, 0x5d, 0xad, 0xdd, 0x25, 0xe6, 0xc9, 0xb4, 0x1b, 0xe3, 0x60, 0xe8, 0x7b, 0x9f, 0x2d, 0x57, 0x95, 0x1b, 0xdd, 0x1c, 0x14, 0x8d, 0xe2, 0xd8, 
0x13, 0x39, 0x88, 0x9e, 0x98, 0x8c, 0x2e, 0xce, 0xf2, 0x7d, 0x22, 0xd, 0xa6, 0xc7, 0xe8, 0xa4, 0x81, 0xa, 0x8e, 0xb8, 0x82, 0x36, 0x37, 0xd4, 0xeb, 0x15, 0xfe, 0xb1, 0xb2, 0xe, 0x9d, 0x62, 
0x1c, 0xb8, 0x13, 0xd1, 0x1e, 0xd6, 0x73, 0xc5, 0x26, 0x84, 0x50, 0x7a, 0x85, 0xa0, 0xe9, 0xb7, 0x80, 0x41, 0x55, 0xec, 0xf0, 0x9c, 0x39, 0x7a, 0xd0, 0xf2, 0x4, 0xcf, 0xc7, 0x97, 0xa0, 0x84, 
0xb7, 0x53, 0xa9, 0xef, 0x7d, 0x9d, 0xbb, 0x6a, 0xd7, 0xd6, 0x71, 0x5e, 0x15, 0x93, 0x5e, 0x3, 0x73, 0x7a, 0xab, 0xd1, 0x79, 0xbe, 0xbb, 0xb1, 0x95, 0x7a, 0x9b, 0x46, 0x3f, 0xf1, 0xc5, 0xa, 
0x2f, 0x94, 0xbb, 0x59, 0xb3, 0xca, 0x73, 0xb6, 0xc3, 0x3e, 0xfc, 0x16, 0xce, 0x40, 0x1d, 0xfb, 0x72, 0x1, 0xdf, 0xa1, 0x33, 0x65, 0xb0, 0xa8, 0xd7, 0x4b, 0x68, 0x9e, 0xaf, 0xd9, 0x23, 0x1b, 
0xe8, 0xd5, 0xc5, 0xce, 0x93, 0xa2, 0x92, 0xec, 0x8a, 0x1f, 0x55, 0xe8, 0xf3, 0x95, 0x88, 0x33, 0xde, 0xdd, 0x2f, 0xc4, 0xeb, 0x19, 0x70, 0x2, 0x31, 0xc7, 0x74, 0xeb, 0x42, 0xeb, 0xdb, 0x33, 
0x54, 0x9e, 0x3, 0x57, 0x69, 0x64, 0x65, 0x39, 0x4e, 0xd5, 0xdc, 0xc0, 0x77, 0xe1, 0x5f, 0x8, 0xcf, 0xa4, 0x5d, 0x1c, 0x21, 0xb1, 0x34, 0x8e, 0x1e, 0x7e, 0xb0, 0xe3, 0xb4, 0xb9, 0x86, 0x94, 
0x5d, 0x1a, 0xd9, 0x62, 0x90, 0x24, 0xa4, 0xc6, 0xd5, 0xb9, 0xc2, 0x73, 0x39, 0xe9, 0xb9, 0xa0, 0xef, 0xe5, 0x5b, 0xc9, 0xcf, 0xfd, 0xdf, 0x36, 0x91, 0x92, 0xcc, 0xfa, 0x14, 0x9d, 0x26, 0xa, 
0x2f, 0x7e, 0xf2, 0x29, 0x3e, 0x27, 0xca, 0xa9, 0x6f, 0x3c, 0x93, 0xa5, 0xe3, 0x6c, 0x67, 0x1c, 0xd1, 0x9d, 0x40, 0x51, 0x2b, 0xd3, 0xf3, 0xcc, 0x50, 0x4f, 0x80, 0xc9, 0x13, 0xf2, 0x4f, 0xf8,
};

/**
 * Returns i modulo the given modulus.
 * i must be less than 2*modulus.
 * Therefore, the return value is either i or i-modulus.
 * @returns i mod (modulus)
 * @param[in] i The integer whose modulo is taken
 * @param[in] modulus The modulus
 */
static uint16_t mod(uint16_t i, uint16_t modulus) {
    uint16_t tmp = i - modulus;

    // mask = 0xffff if(i < PARAM_GF_MUL_ORDER)
    int16_t mask = -(tmp >> 15);

    return tmp + (mask & modulus);
}

/**
 * @brief Computes the generator polynomial of the primitive Reed-Solomon code with given parameters.
 *
 * Code length is 2^m-1. <br>
 * PARAM_DELTA is the targeted correction capacity of the code
 * and receives the real correction capacity (which is at least equal to the target). <br>
 * gf_exp and gf_log are arrays giving antilog and log of GF(2^m) elements.
 *
 * @param[out] poly Array of size (2*PARAM_DELTA + 1) receiving the coefficients of the generator polynomial
 */
void compute_generator_poly(uint16_t *poly) {
    poly[0] = 1;
    int tmp_degree = 0;

    for (uint16_t i = 1; i < (2 * PARAM_DELTA + 1); ++i) {
        for (size_t j = tmp_degree; j; --j) {
            poly[j] = gf_exp[mod(gf_log[poly[j]] + i, PARAM_GF_MUL_ORDER)] ^ poly[j - 1];
        }

        poly[0] = gf_exp[mod(gf_log[poly[0]] + i, PARAM_GF_MUL_ORDER)];
        poly[++tmp_degree] = 1;
    }

    printf("\n");
    for (int i = 0; i < (PARAM_G); ++i) {
        printf("%d, ", poly[i]);
    }
    printf("\n");
}

static inline __m256i linear_map( __m256i tab_l, __m256i tab_h, __m256i v, __m256i mask_f ) {
    return _mm256_shuffle_epi8(tab_l, v & mask_f)^_mm256_shuffle_epi8(tab_h, _mm256_srli_epi16(v, 4)&mask_f);
}

/**
 * @brief Encodes a message message of PARAM_K bits to a Reed-Solomon codeword codeword of PARAM_N1 bytes
 *
 * Following @cite lin1983error (Chapter 4 - Cyclic Codes),
 * We perform a systematic encoding using a linear (PARAM_N1 - PARAM_K)-stage shift register
 * with feedback connections based on the generator polynomial PARAM_RS_POLY of the Reed-Solomon code.
 *
 * @param[out] cdw Array of size VEC_N1_SIZE_64 receiving the encoded message
 * @param[in] msg Array of size VEC_K_SIZE_64 storing the message
 */
void reed_solomon_encode(uint64_t *cdw, const uint64_t *msg) {
    #if 24 != PARAM_K
    #error "Reed-Solomon encoding only implemented for hqc-3 k=24"
    #endif
    // hqc-3 k=24
    __m256i multab[24];
    gf256v_generate_multabs_avx2((uint8_t *)multab, (uint8_t *)msg, 24);
    __m256i mask = _mm256_set1_epi8(0xf);

    __m256i parity = _mm256_setzero_si256();
    __m256i* enc_mat_ptr = (__m256i *)gen_mat;
    for(int i=0;i<PARAM_K;i++) {
        __m256i m_tab = multab[i];
        __m256i ml = _mm256_permute2x128_si256( m_tab, m_tab, 0 );
        __m256i mh = _mm256_permute2x128_si256( m_tab, m_tab, 0x11 );

        parity ^= linear_map( ml , mh , enc_mat_ptr[i] , mask);
    }
    _mm256_storeu_si256((__m256i *)cdw, parity);

    memcpy(((uint8_t*)cdw) + PARAM_N1-PARAM_K, msg , PARAM_K);
}

/**
 * @brief Computes 2 * PARAM_DELTA syndromes
 *
 * @param[out] syndromes256 Array of size 2 * PARAM_DELTA receiving the computed syndromes
 * @param[in] cdw Array of size PARAM_N1 storing the received vector
 */
void compute_syndromes(__m256i *syndromes256, uint8_t *cdw) {
    syndromes256[0] = _mm256_set1_epi16(cdw[0]);
    syndromes256[1] = _mm256_set1_epi16(cdw[0]);

    __m256i multabs[PARAM_N1-1];
    gf256v_generate_multabs_avx2((uint8_t *)multabs, cdw+1, PARAM_N1-1);
    __m256i mask = _mm256_set1_epi8(0xf);
    for( int i = 0; i < PARAM_N1-1; i ++)
    {
        __m256i m_tab = multabs[i];
        __m256i ml = _mm256_permute2x128_si256( m_tab, m_tab, 0 );
        __m256i mh = _mm256_permute2x128_si256( m_tab, m_tab, 0x11 );
        syndromes256[0] ^= linearmap_8x8_ymm(alpha_ij256_1[i], ml, mh, mask);
        syndromes256[1] ^= linearmap_8x8_ymm(alpha_ij256_2[i], ml, mh, mask);
    }
}

/**
 * @brief Computes the error locator polynomial (ELP) sigma
 *
 * This is a constant time implementation of Berlekamp's algorithm (see @cite lin1983error (Chapter 6 - BCH Codes). <br>
 * We use the letter p for rho which is initialized at -1. <br>
 * The array X_sigma_p represents the polynomial X^(mu-rho)*sigma_p(X). <br>
 * Instead of maintaining a list of sigmas, we update in place both sigma and X_sigma_p. <br>
 * sigma_copy serves as a temporary save of sigma in case X_sigma_p needs to be updated. <br>
 * We can properly correct only if the degree of sigma does not exceed PARAM_DELTA.
 * This means only the first PARAM_DELTA + 1 coefficients of sigma are of value
 * and we only need to save its first PARAM_DELTA - 1 coefficients.
 *
 * @returns the degree of the ELP sigma
 * @param[out] sigma Array of size (at least) PARAM_DELTA receiving the ELP
 * @param[in] syndromes Array of size (at least) 2*PARAM_DELTA storing the syndromes
 */
static uint16_t compute_elp(uint16_t *sigma, const uint16_t *syndromes) {
    uint16_t deg_sigma = 0;
    uint16_t deg_sigma_p = 0;
    uint16_t deg_sigma_copy = 0;
    uint16_t sigma_copy[PARAM_DELTA + 1] = {0};
    uint16_t X_sigma_p[PARAM_DELTA + 1] = {0, 1};
    uint16_t pp = (uint16_t)-1;  // 2*rho
    uint16_t d_p = 1;
    uint16_t d = syndromes[0];

    uint16_t mask1, mask2, mask12;
    uint16_t deg_X, deg_X_sigma_p;
    uint16_t dd;
    uint16_t mu;

    uint16_t i;

    sigma[0] = 1;
    for (mu = 0; (mu < (2 * PARAM_DELTA)); ++mu) {
        // Save sigma in case we need it to update X_sigma_p
        memcpy(sigma_copy, sigma, 2 * (PARAM_DELTA));
        deg_sigma_copy = deg_sigma;

        dd = gf_mul(d, gf_inverse(d_p));

        for (i = 1; (i <= mu + 1) && (i <= PARAM_DELTA); ++i) {
            sigma[i] ^= gf_mul(dd, X_sigma_p[i]);
        }

        deg_X = mu - pp;
        deg_X_sigma_p = deg_X + deg_sigma_p;

        // mask1 = 0xffff if(d != 0) and 0 otherwise
        mask1 = -((uint16_t)-d >> 15);

        // mask2 = 0xffff if(deg_X_sigma_p > deg_sigma) and 0 otherwise
        mask2 = -((uint16_t)(deg_sigma - deg_X_sigma_p) >> 15);

        // mask12 = 0xffff if the deg_sigma increased and 0 otherwise
        volatile uint16_t mask12__ = mask1 & mask2;
        mask12 = mask12__;
        deg_sigma ^= mask12 & (deg_X_sigma_p ^ deg_sigma);

        if (mu == (2 * PARAM_DELTA - 1)) {
            break;
        }

        pp ^= mask12 & (mu ^ pp);
        d_p ^= mask12 & (d ^ d_p);
        for (i = PARAM_DELTA; i; --i) {
            X_sigma_p[i] = (mask12 & sigma_copy[i - 1]) ^ (~mask12 & X_sigma_p[i - 1]);
        }

        deg_sigma_p ^= mask12 & (deg_sigma_copy ^ deg_sigma_p);
        d = syndromes[mu + 1];

        for (i = 1; (i <= mu + 1) && (i <= PARAM_DELTA); ++i) {
            d ^= gf_mul(sigma[i], syndromes[mu + 1 - i]);
        }
    }

    return deg_sigma;
}

/**
 * @brief Computes the error polynomial error from the error locator polynomial sigma
 *
 * See function fft for more details.
 *
 * @param[out] error Array of 2^PARAM_M elements receiving the error polynomial
 * @param[in] sigma Array of 2^PARAM_FFT elements storing the error locator polynomial
 */
static void compute_roots(uint8_t *error, uint16_t *sigma) {
    uint16_t w[1 << PARAM_M] = {0};

    fft(w, sigma, PARAM_DELTA + 1);
    fft_retrieve_error_poly(error, w);
}

/**
 * @brief Computes the polynomial z(x)
 *
 * See @cite lin1983error (Chapter 6 - BCH Codes) for more details.
 *
 * @param[out] z Array of PARAM_DELTA + 1 elements receiving the polynomial z(x)
 * @param[in] sigma Array of 2^PARAM_FFT elements storing the error locator polynomial
 * @param[in] degree Integer that is the degree of polynomial sigma
 * @param[in] syndromes Array of 2 * PARAM_DELTA storing the syndromes
 */
static void compute_z_poly(uint16_t *z, const uint16_t *sigma, uint16_t degree, const uint16_t *syndromes) {
    size_t i, j;
    uint16_t mask;

    z[0] = 1;

    for (i = 1; i < PARAM_DELTA + 1; ++i) {
        mask = -((uint16_t)(i - degree - 1) >> 15);
        z[i] = mask & sigma[i];
    }

    z[1] ^= syndromes[0];

    for (i = 2; i <= PARAM_DELTA; ++i) {
        mask = -((uint16_t)(i - degree - 1) >> 15);
        z[i] ^= mask & syndromes[i - 1];

        for (j = 1; j < i; ++j) {
            z[i] ^= mask & gf_mul(sigma[j], syndromes[i - j - 1]);
        }
    }
}

/**
 * @brief Computes the error values
 *
 * See @cite lin1983error (Chapter 6 - BCH Codes) for more details.
 *
 * @param[out] error_values Array of PARAM_DELTA elements receiving the error values
 * @param[in] z Array of PARAM_DELTA + 1 elements storing the polynomial z(x)
 * @param[in] error Array of PARAM_DELTA elements storing the errors positions
 */
static void compute_error_values(uint16_t *error_values, const uint16_t *z, const uint8_t *error) {
    uint16_t beta_j[PARAM_DELTA] = {0};
    uint16_t e_j[PARAM_DELTA] = {0};

    uint16_t delta_counter;
    uint16_t delta_real_value;
    uint16_t found;
    uint16_t mask1;
    uint16_t mask2;
    uint16_t tmp1;
    uint16_t tmp2;
    uint16_t inverse;
    uint16_t inverse_power_j;

    // Compute the beta_{j_i} page 31 of the documentation
    delta_counter = 0;
    for (size_t i = 0; i < PARAM_N1; i++) {
        found = 0;
        mask1 = (uint16_t)(-((int32_t)error[i]) >> 31);  // error[i] != 0
        for (size_t j = 0; j < PARAM_DELTA; j++) {
            mask2 = ~((uint16_t)(-((int32_t)j ^ delta_counter) >> 31));  // j == delta_counter
            beta_j[j] += mask1 & mask2 & gf_exp[i];
            found += mask1 & mask2 & 1;
        }
        delta_counter += found;
    }
    delta_real_value = delta_counter;

    // Compute the e_{j_i} page 31 of the documentation
    for (size_t i = 0; i < PARAM_DELTA; ++i) {
        tmp1 = 1;
        tmp2 = 1;
        inverse = gf_inverse(beta_j[i]);
        inverse_power_j = 1;

        for (size_t j = 1; j <= PARAM_DELTA; ++j) {
            inverse_power_j = gf_mul(inverse_power_j, inverse);
            tmp1 ^= gf_mul(inverse_power_j, z[j]);
        }
        for (size_t k = 1; k < PARAM_DELTA; ++k) {
            tmp2 = gf_mul(tmp2, (1 ^ gf_mul(inverse, beta_j[(i + k) % PARAM_DELTA])));
        }
        mask1 = (uint16_t)(((int16_t)i - delta_real_value) >> 15);  // i < delta_real_value
        e_j[i] = mask1 & gf_mul(tmp1, gf_inverse(tmp2));
    }

    // Place the delta e_{j_i} values at the right coordinates of the output vector
    delta_counter = 0;
    for (size_t i = 0; i < PARAM_N1; ++i) {
        found = 0;
        mask1 = (uint16_t)(-((int32_t)error[i]) >> 31);  // error[i] != 0
        for (size_t j = 0; j < PARAM_DELTA; j++) {
            mask2 = ~((uint16_t)(-((int32_t)j ^ delta_counter) >> 31));  // j == delta_counter
            error_values[i] += mask1 & mask2 & e_j[j];
            found += mask1 & mask2 & 1;
        }
        delta_counter += found;
    }
}

/**
 * @brief Correct the errors
 *
 * @param[out] cdw Array of PARAM_N1 elements receiving the corrected vector
 * @param[in] error_values Array of PARAM_DELTA elements storing the error values
 */
static void correct_errors(uint8_t *cdw, const uint16_t *error_values) {
    for (size_t i = 0; i < PARAM_N1; ++i) {
        cdw[i] ^= error_values[i];
    }
}

/**
 * @brief Decodes the received word
 *
 * This function relies on six steps:
 * -# Compute the 2·PARAM_DELTA syndromes.
 * -# Compute the error-locator polynomial σ(x).
 * -# Use an additive FFT to find the roots of σ(x) (the error locations) and take their inverses.
 * -# Compute the error-evaluator polynomial z(x).
 * -# Compute the error values at each located position.
 * -# Correct the received polynomial by subtracting the error values.
 *
 * For a more complete picture on Reed-Solomon decoding, see Shu. Lin and Daniel J. Costello in Error Control Coding:
 * Fundamentals and Applications @cite lin1983error
 *
 * @param[out] msg Array of size VEC_K_SIZE_64 receiving the decoded message
 * @param[in] cdw Array of size VEC_N1_SIZE_64 storing the received word
 */
void reed_solomon_decode(uint64_t *msg, uint64_t *cdw) {
    uint8_t cdw_bytes[PARAM_N1] = {0};
    __m256i syndromes256[SYND_SIZE_256];
    uint16_t *syndromes = (uint16_t *)syndromes256;
    uint16_t sigma[1 << PARAM_FFT] = {0};
    uint8_t error[1 << PARAM_M] = {0};
    uint16_t z[PARAM_N1] = {0};
    uint16_t error_values[PARAM_N1] = {0};
    uint16_t deg;

    // Copy the vector in an array of bytes
    memcpy(cdw_bytes, cdw, PARAM_N1);

    // Calculate the 2*PARAM_DELTA syndromes
    compute_syndromes(syndromes256, cdw_bytes);

    // Compute the error locator polynomial sigma
    // Sigma's degree is at most PARAM_DELTA but the FFT requires the extra room
    deg = compute_elp(sigma, syndromes);

    // Compute the error polynomial error
    compute_roots(error, sigma);

    // Compute the polynomial z(x)
    compute_z_poly(z, sigma, deg, syndromes);

    // Compute the error values
    compute_error_values(error_values, z, error);

    // Correct the errors
    correct_errors(cdw_bytes, error_values);

    // Retrieve the message from the decoded codeword
    memcpy(msg, cdw_bytes + (PARAM_G - 1), PARAM_K);

#ifdef VERBOSE
    printf("\n\nThe syndromes: ");
    for (size_t i = 0; i < 2 * PARAM_DELTA; ++i) {
        printf("%u ", syndromes[i]);
    }
    printf("\n\nThe error locator polynomial: sigma(x) = ");
    bool first_coeff = true;
    if (sigma[0]) {
        printf("%u", sigma[0]);
        first_coeff = false;
    }
    for (size_t i = 1; i < (1 << PARAM_FFT); ++i) {
        if (sigma[i] == 0)
            continue;
        if (!first_coeff)
            printf(" + ");
        first_coeff = false;
        if (sigma[i] != 1)
            printf("%u ", sigma[i]);
        if (i == 1)
            printf("x");
        else
            printf("x^%zu", i);
    }
    if (first_coeff)
        printf("0");

    printf("\n\nThe polynomial: z(x) = ");
    bool first_coeff_1 = true;
    if (z[0]) {
        printf("%u", z[0]);
        first_coeff_1 = false;
    }
    for (size_t i = 1; i < (PARAM_DELTA + 1); ++i) {
        if (z[i] == 0)
            continue;
        if (!first_coeff_1)
            printf(" + ");
        first_coeff_1 = false;
        if (z[i] != 1)
            printf("%u ", z[i]);
        if (i == 1)
            printf("x");
        else
            printf("x^%zu", i);
    }
    if (first_coeff_1)
        printf("0");

    printf("\n\nThe pairs of (error locator numbers, error values): ");
    size_t j = 0;
    for (size_t i = 0; i < PARAM_N1; ++i) {
        if (error[i]) {
            printf("(%zu, %d) ", i, error_values[j]);
            j++;
        }
    }
    printf("\n");
#endif

    // Zeroize sensitive data
    memset_zero(cdw_bytes, sizeof cdw_bytes);
}
