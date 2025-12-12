#include "stdint.h"
#include "string.h"
#include "gf256.h"
#include "combine.h"
#include "arm_neon.h"
const uint16_t loading_index[108] = {
    127, 191, 247, 251, 255, 319, 367, 379, 382, 383, 431,
    439, 446, 447, 487, 491, 495, 499, 502, 503, 506, 507, 
    510, 511, 575, 607, 635, 637, 639, 671, 695, 701, 703, 
    727, 731, 735, 755, 757, 759, 761, 763, 765, 767, 799, 
    815, 829, 830, 831, 847, 859, 862, 863, 875, 877, 879, 
    889, 890, 891, 892, 893, 894, 895, 911, 919, 926, 927, 
    935, 941, 943, 949, 950, 951, 956, 957, 958, 959, 967, 
    971, 975, 979, 982, 983, 986, 987, 990, 991, 995, 997, 
    999, 1001, 1003, 1005, 1007, 1009, 1010, 1011, 1012, 1013, 
    1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023};
const uint16_t another_index[145] = {
    63, 111, 123, 126, 175, 183, 190, 231, 235, 239, 243, 246, 
    250, 254, 351, 367, 381, 382, 415, 431, 445, 446, 471, 475, 
    479, 487, 491, 495, 501, 502, 505, 506, 509, 510, 543, 559, 
    573, 574, 591, 603, 606, 619, 621, 623, 633, 634, 636, 638, 
    655, 663, 670, 679, 685, 687, 693, 694, 700, 702, 711, 715, 
    719, 723, 726, 730, 734, 739, 741, 743, 745, 747, 749, 751, 
    753, 754, 756, 758, 760, 762, 764, 766, 799, 815, 829, 830, 
    831, 847, 859, 862, 863, 875, 877, 879, 889, 890, 891, 892, 
    893, 894, 895, 911, 919, 926, 927, 935, 941, 943, 949, 950, 
    951, 956, 957, 958, 959, 967, 971, 975, 979, 982, 983, 986, 
    987, 990, 991, 995, 997, 999, 1001, 1003, 1005, 1007, 1009, 
    1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 
    1020, 1021, 1022, 1023
};
const uint16_t new_index[180] = {
    7, 11, 15, 19, 22, 26, 30, 143, 157, 158, 207, 221, 222, 263, 
    267, 271, 277, 278, 281, 282, 285, 286, 335, 349, 350, 383, 
    395, 397, 399, 409, 410, 412, 414, 447, 455, 461, 463, 469, 
    470, 476, 478, 503, 507, 511, 515, 517, 519, 521, 523, 525, 
    527, 529, 530, 532, 534, 536, 538, 540, 542, 591, 605, 606, 
    623, 637, 638, 651, 653, 655, 665, 666, 667, 668, 669, 670, 
    687, 701, 702, 711, 717, 719, 725, 726, 731, 732, 733, 734, 
    743, 747, 751, 757, 758, 761, 762, 765, 766, 771, 773, 775, 
    777, 779, 781, 783, 785, 786, 787, 788, 789, 790, 791, 792, 
    793, 794, 795, 796, 797, 798, 815, 829, 830, 831, 847, 859, 
    862, 863, 875, 877, 879, 889, 890, 891, 892, 893, 894, 895, 
    911, 919, 926, 927, 935, 941, 943, 949, 950, 951, 956, 957, 
    958, 959, 967, 971, 975, 979, 982, 983, 986, 987, 990, 991, 
    995, 997, 999, 1001, 1003, 1005, 1007, 1009, 1010, 1011, 1012, 
    1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023
};
const uint16_t new_index_2[141] = {
    141, 142, 205, 206, 261, 262, 265, 266, 269, 270, 333, 334, 
    381, 382, 393, 394, 396, 398, 445, 446, 453, 454, 460, 462, 
    501, 502, 505, 506, 509, 510, 513, 514, 516, 518, 520, 522, 
    524, 526, 589, 590, 621, 622, 649, 650, 651, 652, 653, 654, 
    685, 686, 709, 710, 715, 716, 717, 718, 741, 742, 745, 746, 
    749, 750, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 
    779, 780, 781, 782, 813, 814, 829, 830, 843, 846, 861, 862, 
    873, 874, 876, 878, 889, 890, 891, 892, 893, 894, 903, 910, 
    925, 926, 933, 934, 940, 942, 949, 950, 955, 956, 957, 958, 
    963, 966, 970, 974, 981, 982, 985, 986, 989, 990, 993, 994, 
    996, 998, 1000, 1002, 1004, 1006, 1009, 1010, 1011, 1012, 1013, 1014, 
    1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023
};
void combine_part_4479(uint8_t* dest_high, const uint8_t * pc_m384_high, const uint8_t *c1)
{
    dest_high[0] = c1[0];
    uint8x16_t xor_0_to_15 = veorq_u8(vld1q_u8(c1 ), vld1q_u8(pc_m384_high));
    
    const uint8x16_t xor_16_to_31 = veorq_u8(vld1q_u8(c1 + 16), vld1q_u8(pc_m384_high + 16));
    xor_0_to_15[0] = 0;
    const uint8x16_t xor_32_to_47 = veorq_u8(vld1q_u8(c1 + 32), vld1q_u8(pc_m384_high + 32));
    const uint8x16_t xor_48_to_63 = veorq_u8(vld1q_u8(c1 + 48), vld1q_u8(pc_m384_high + 48));
    const uint8x16_t xor_64_to_79 = veorq_u8(vld1q_u8(c1 + 64), vld1q_u8(pc_m384_high + 64));
    const uint8x16_t xor_80_to_95 = veorq_u8(vld1q_u8(c1 + 80), vld1q_u8(pc_m384_high + 80));
    const uint8x16_t xor_96_to_111 = veorq_u8(vld1q_u8(c1 + 96), vld1q_u8(pc_m384_high + 96));
    const uint8x16_t xor_112_to_127 = veorq_u8(vld1q_u8(c1 + 112), vld1q_u8(pc_m384_high + 112));
    const uint8x16_t xor_128_to_143 = veorq_u8(vld1q_u8(c1 + 128), vld1q_u8(pc_m384_high + 128));
    const uint8x16_t xor_144_to_159 = veorq_u8(vld1q_u8(c1 + 144), vld1q_u8(pc_m384_high + 144));
    const uint8x16_t xor_160_to_175 = veorq_u8(vld1q_u8(c1 + 160), vld1q_u8(pc_m384_high + 160));
    const uint8x16_t xor_176_to_191 = veorq_u8(vld1q_u8(c1 + 176), vld1q_u8(pc_m384_high + 176));
    const uint8x16_t xor_192_to_207 = veorq_u8(vld1q_u8(c1 + 192), vld1q_u8(pc_m384_high + 192));
    const uint8x16_t xor_208_to_223 = veorq_u8(vld1q_u8(c1 + 208), vld1q_u8(pc_m384_high + 208));
    const uint8x16_t xor_224_to_239 = veorq_u8(vld1q_u8(c1 + 224), vld1q_u8(pc_m384_high + 224));
    const uint8x16_t xor_240_to_255 = veorq_u8(vld1q_u8(c1 + 240), vld1q_u8(pc_m384_high + 240));
    const uint8x16_t xor_256_to_271 = veorq_u8(vld1q_u8(c1 + 256), vld1q_u8(pc_m384_high + 256));
    const uint8x16_t xor_272_to_287 = veorq_u8(vld1q_u8(c1 + 272), vld1q_u8(pc_m384_high + 272));
    const uint8x16_t xor_288_to_303 = veorq_u8(vld1q_u8(c1 + 288), vld1q_u8(pc_m384_high + 288));
    const uint8x16_t xor_304_to_319 = veorq_u8(vld1q_u8(c1 + 304), vld1q_u8(pc_m384_high + 304));
    const uint8x16_t xor_320_to_335 = veorq_u8(vld1q_u8(c1 + 320), vld1q_u8(pc_m384_high + 320));
    const uint8x16_t xor_336_to_351 = veorq_u8(vld1q_u8(c1 + 336), vld1q_u8(pc_m384_high + 336));
    const uint8x16_t xor_352_to_367 = veorq_u8(vld1q_u8(c1 + 352), vld1q_u8(pc_m384_high + 352));
    const uint8x16_t xor_368_to_383 = veorq_u8(vld1q_u8(c1 + 368), vld1q_u8(pc_m384_high + 368));

    const uint8x16_t zero = vdupq_n_u8(0);


    uint8x16_t temp1, temp2, temp3, temp4;
    // last 64 byte
    // 23, 53
    temp1 = vextq_u8(zero, xor_0_to_15, 6);
    temp2 = vextq_u8(xor_0_to_15, xor_16_to_31, 6);
    temp3 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 6), vextq_u8(zero, xor_0_to_15, 8));
    temp4 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 6), vextq_u8(xor_0_to_15, xor_16_to_31, 8));

    // 83, 113
    temp1 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 4), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 2), temp1));
    temp2 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 4), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 2), temp2));
    temp3 = veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 4), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 2), temp3));
    temp4 = veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 4), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 2), temp4));

    
    // 143, 158, 173, 188, 
    temp1 = veorq_u8(xor_80_to_95, veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 15), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 14), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 13), temp1))));
    temp2 = veorq_u8(xor_96_to_111, veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 15), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 14), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 13), temp2))));
    temp3 = veorq_u8(xor_112_to_127, veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 15), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 14), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 13), temp3))));
    temp4 = veorq_u8(xor_128_to_143, veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 15), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 14), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 13), temp4))));
    // 203, 218, 233, 248,
    temp1 = veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 12), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 11), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 10), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 9), temp1))));
    temp2 = veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 12), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 11), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 10), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 9), temp2))));
    temp3 = veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 12), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 11), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 10), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 9), temp3))));
    temp4 = veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 12), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 11), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 10), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 9), temp4))));
     // 263, 278, 293, 308,
    temp1 = veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 8), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 7), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 6), veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 5), temp1))));
    temp2 = veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 8), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 7), veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 6), veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 5), temp2))));
    temp3 = veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 8), veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 7), veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 6), veorq_u8(vextq_u8(xor_272_to_287, xor_288_to_303, 5), temp3))));
    temp4 = veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 8), veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 7), veorq_u8(vextq_u8(xor_272_to_287, xor_288_to_303, 6), veorq_u8(vextq_u8(xor_288_to_303, xor_304_to_319, 5), temp4))));
    // 323, 338, 353, 368, 383
    temp1 = veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 4), veorq_u8(vextq_u8(xor_272_to_287, xor_288_to_303, 3), veorq_u8(vextq_u8(xor_288_to_303, xor_304_to_319, 2), veorq_u8(vextq_u8(xor_304_to_319, xor_320_to_335, 1), veorq_u8(xor_320_to_335, temp1)))));
    temp2 = veorq_u8(vextq_u8(xor_272_to_287, xor_288_to_303, 4), veorq_u8(vextq_u8(xor_288_to_303, xor_304_to_319, 3), veorq_u8(vextq_u8(xor_304_to_319, xor_320_to_335, 2), veorq_u8(vextq_u8(xor_320_to_335, xor_336_to_351, 1), veorq_u8(xor_336_to_351, temp2)))));
    temp3 = veorq_u8(vextq_u8(xor_288_to_303, xor_304_to_319, 4), veorq_u8(vextq_u8(xor_304_to_319, xor_320_to_335, 3), veorq_u8(vextq_u8(xor_320_to_335, xor_336_to_351, 2), veorq_u8(vextq_u8(xor_336_to_351, xor_352_to_367, 1), veorq_u8(xor_352_to_367, temp3)))));
    temp4 = veorq_u8(vextq_u8(xor_304_to_319, xor_320_to_335, 4), veorq_u8(vextq_u8(xor_320_to_335, xor_336_to_351, 3), veorq_u8(vextq_u8(xor_336_to_351, xor_352_to_367, 2), veorq_u8(vextq_u8(xor_352_to_367, xor_368_to_383, 1), veorq_u8(xor_368_to_383, temp4)))));

    vst1q_u8(dest_high + 4415, temp1);
    vst1q_u8(dest_high + 575, veorq_u8(temp1, vld1q_u8(c1 + 575)));
    vst1q_u8(dest_high + 4431, temp2);
    vst1q_u8(dest_high + 591, veorq_u8(temp2, vld1q_u8(c1 + 591)));
    vst1q_u8(dest_high + 4447, temp3);
    vst1q_u8(dest_high + 607, veorq_u8(temp3, vld1q_u8(c1 + 607)));
    vst1q_u8(dest_high + 4463, temp4);
    vst1q_u8(dest_high + 623, veorq_u8(temp4, vld1q_u8(c1 + 623)));

    // last 65 to 128
    // 83 118
    temp1 = vextq_u8(zero, xor_0_to_15, 2);
    temp2 = vextq_u8(xor_0_to_15, xor_16_to_31, 2);
    temp3 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 2), vextq_u8(zero, xor_0_to_15, 4));
    temp4 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 2), vextq_u8(xor_0_to_15, xor_16_to_31, 4));

    // 143, 158, 173, 188,
    temp1 = veorq_u8(xor_16_to_31, veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 15), veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 14), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 13), temp1))));
    temp2 = veorq_u8(xor_32_to_47, veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 15), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 14), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 13), temp2))));
    temp3 = veorq_u8(xor_48_to_63, veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 15), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 14), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 13), temp3))));
    temp4 = veorq_u8(xor_64_to_79, veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 15), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 14), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 13), temp4))));
    // 203, 218, 233, 248,
    temp1 = veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 12), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 11), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 10), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 9), temp1))));
    temp2 = veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 12), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 11), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 10), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 9), temp2))));
    temp3 = veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 12), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 11), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 10), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 9), temp3))));
    temp4 = veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 12), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 11), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 10), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 9), temp4))));
    // 263, 278, 293, 308,
    temp1 = veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 8), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 7), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 6), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 5), temp1))));
    temp2 = veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 8), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 7), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 6), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 5), temp2))));
    temp3 = veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 8), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 7), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 6), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 5), temp3))));
    temp4 = veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 8), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 7), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 6), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 5), temp4))));
    // 323, 338, 353, 368, 383
    temp1 = veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 4), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 3), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 2), veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 1), veorq_u8(xor_256_to_271, temp1)))));
    temp2 = veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 4), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 3), veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 2), veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 1), veorq_u8(xor_272_to_287, temp2)))));
    temp3 = veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 4), veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 3), veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 2), veorq_u8(vextq_u8(xor_272_to_287, xor_288_to_303, 1), veorq_u8(xor_288_to_303, temp3)))));
    temp4 = veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 4), veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 3), veorq_u8(vextq_u8(xor_272_to_287, xor_288_to_303, 2), veorq_u8(vextq_u8(xor_288_to_303, xor_304_to_319, 1), veorq_u8(xor_304_to_319, temp4)))));

    vst1q_u8(dest_high + 4351, temp1);
    vst1q_u8(dest_high + 511, veorq_u8(temp1, vld1q_u8(c1 + 511)));
    vst1q_u8(dest_high + 4367, temp2);
    vst1q_u8(dest_high + 527, veorq_u8(temp2, vld1q_u8(c1 + 527)));
    vst1q_u8(dest_high + 4383, temp3);
    vst1q_u8(dest_high + 543, veorq_u8(temp3, vld1q_u8(c1 + 543)));
    vst1q_u8(dest_high + 4399, temp4);
    vst1q_u8(dest_high + 559, veorq_u8(temp4, vld1q_u8(c1 + 559)));

    // last 129 to 192
    // 143, 158, 173, 188
    temp1 = vextq_u8(zero, xor_0_to_15, 13);
    temp2 = veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 13), vextq_u8(zero, xor_0_to_15, 14));
    temp3 = veorq_u8(veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 13), vextq_u8(xor_0_to_15, xor_16_to_31, 14)), vextq_u8(zero, xor_0_to_15, 15));
    temp4 = veorq_u8(veorq_u8(veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 13), vextq_u8(xor_16_to_31, xor_32_to_47, 14)), vextq_u8(xor_0_to_15, xor_16_to_31, 15)), xor_0_to_15);

    // 203, 218, 233, 248
    temp1 = veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 12), veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 11), veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 10), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 9), temp1))));
    temp2 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 12), veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 11), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 10), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 9), temp2))));
    temp3 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 12), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 11), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 10), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 9), temp3))));
    temp4 = veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 12), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 11), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 10), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 9), temp4))));
    // 263, 278, 293, 308
    temp1 = veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 8), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 7), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 6), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 5), temp1))));
    temp2 = veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 8), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 7), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 6), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 5), temp2))));
    temp3 = veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 8), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 7), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 6), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 5), temp3))));
    temp4 = veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 8), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 7), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 6), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 5), temp4))));
    // 323, 338, 353, 368, 383
    temp1 = veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 4), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 3), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 2), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 1), veorq_u8(xor_192_to_207, temp1)))));
    temp2 = veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 4), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 3), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 2), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 1), veorq_u8(xor_208_to_223, temp2)))));
    temp3 = veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 4), veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 3), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 2), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 1), veorq_u8(xor_224_to_239, temp3)))));
    temp4 = veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 4), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 3), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 2), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 1), veorq_u8(xor_240_to_255, temp4)))));

    vst1q_u8(dest_high + 4287, temp1);
    vst1q_u8(dest_high + 447, veorq_u8(temp1, vld1q_u8(c1 + 447)));
    vst1q_u8(dest_high + 4303, temp2);
    vst1q_u8(dest_high + 463, veorq_u8(temp2, vld1q_u8(c1 + 463)));
    vst1q_u8(dest_high + 4319, temp3);
    vst1q_u8(dest_high + 479, veorq_u8(temp3, vld1q_u8(c1 + 479)));
    vst1q_u8(dest_high + 4335, temp4);
    vst1q_u8(dest_high + 495, veorq_u8(temp4, vld1q_u8(c1 + 495)));
    
    // last 193 to 256
    // 203, 218, 233, 248
    temp1 = vextq_u8(zero, xor_0_to_15, 9);
    temp2 = veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 9), vextq_u8(zero, xor_0_to_15, 10));
    temp3 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 9), veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 10), vextq_u8(zero, xor_0_to_15, 11)));
    temp4 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 9), veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 10), veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 11), vextq_u8(zero, xor_0_to_15, 12))));
    
    // 263, 278, 293, 308
    temp1 = veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 8), veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 7), veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 6), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 5), temp1))));
    temp2 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 8), veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 7), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 6), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 5), temp2))));
    temp3 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 8), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 7), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 6), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 5), temp3))));
    temp4 = veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 8), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 7), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 6), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 5), temp4))));

    // 323, 338, 353, 368, 383
    temp1 = veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 4), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 3), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 2), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 1), veorq_u8(xor_128_to_143, temp1)))));
    temp2 = veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 4), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 3), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 2), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 1), veorq_u8(xor_144_to_159, temp2)))));
    temp3 = veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 4), veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 3), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 2), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 1), veorq_u8(xor_160_to_175, temp3)))));
    temp4 = veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 4), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 3), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 2), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 1), veorq_u8(xor_176_to_191, temp4)))));

    vst1q_u8(dest_high + 4223, temp1);
    vst1q_u8(dest_high + 4239, temp2);
    vst1q_u8(dest_high + 399, veorq_u8(temp2, vld1q_u8(c1 + 399)));
    vst1q_u8(dest_high + 4255, temp3);
    vst1q_u8(dest_high + 415, veorq_u8(temp3, vld1q_u8(c1 + 415)));
    vst1q_u8(dest_high + 4271, temp4);
    vst1q_u8(dest_high + 431, veorq_u8(temp4, vld1q_u8(c1 + 431)));

    // last 257 to 320
    // 263, 278, 293, 308
    temp1 = vextq_u8(zero, xor_0_to_15, 5);
    temp2 = veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 5), vextq_u8(zero, xor_0_to_15, 6));
    temp3 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 5), veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 6), vextq_u8(zero, xor_0_to_15, 7)));
    temp4 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 5), veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 6), veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 7), vextq_u8(zero, xor_0_to_15, 8))));
    // 323, 338, 353, 368, 383
    temp1 = veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 4), veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 3), veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 2), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 1), veorq_u8(xor_64_to_79, temp1)))));
    temp2 = veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 4), veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 3), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 2), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 1), veorq_u8(xor_80_to_95, temp2)))));
    temp3 = veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 4), veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 3), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 2), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 1), veorq_u8(xor_96_to_111, temp3)))));
    temp4 = veorq_u8(vextq_u8(xor_48_to_63, xor_64_to_79, 4), veorq_u8(vextq_u8(xor_64_to_79, xor_80_to_95, 3), veorq_u8(vextq_u8(xor_80_to_95, xor_96_to_111, 2), veorq_u8(vextq_u8(xor_96_to_111, xor_112_to_127, 1), veorq_u8(xor_112_to_127, temp4)))));

    vst1q_u8(dest_high + 4159, temp1);
    vst1q_u8(dest_high + 4175, temp2);
    vst1q_u8(dest_high + 4191, temp3);
    vst1q_u8(dest_high + 4207, temp4);

    // last 321 to 384
    // 323, 338, 353, 368, 383
    temp1 = veorq_u8(xor_0_to_15, vextq_u8(zero, xor_0_to_15, 1));
    temp2 = veorq_u8(xor_16_to_31, veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 1), vextq_u8(zero, xor_0_to_15, 2)));
    temp3 = veorq_u8(xor_32_to_47, veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 1), veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 2), vextq_u8(zero, xor_0_to_15, 3))));
    temp4 = veorq_u8(xor_48_to_63, veorq_u8(vextq_u8(xor_32_to_47, xor_48_to_63, 1), veorq_u8(vextq_u8(xor_16_to_31, xor_32_to_47, 2), veorq_u8(vextq_u8(xor_0_to_15, xor_16_to_31, 3), vextq_u8(zero, xor_0_to_15, 4)))));

    vst1q_u8(dest_high + 4095, temp1);
    vst1q_u8(dest_high + 4111, temp2);
    vst1q_u8(dest_high + 4127, temp3);
    vst1q_u8(dest_high + 4143, temp4);

    // index 384 to 398
    // 8, 38, 68, 98, 128, 158, 173, 188, 203, 218, 233, 248, 263, 278, 293, 308, 323, 338, 353, 368, 383
    // 8, 38, 68, 98
    temp1 = veorq_u8(veorq_u8(veorq_u8(vextq_u8(zero, xor_0_to_15, 9), vextq_u8(xor_16_to_31, xor_32_to_47, 7)), vextq_u8(xor_48_to_63, xor_64_to_79, 5)), vextq_u8(xor_80_to_95, xor_96_to_111, 3));
    // 128, 158, 173, 188
    temp1 = veorq_u8(vextq_u8(xor_112_to_127, xor_128_to_143, 1), veorq_u8(vextq_u8(xor_128_to_143, xor_144_to_159, 15), veorq_u8(vextq_u8(xor_144_to_159, xor_160_to_175, 14), veorq_u8(vextq_u8(xor_160_to_175, xor_176_to_191, 13), temp1))));
    // 203, 218, 233, 248
    temp1 = veorq_u8(vextq_u8(xor_176_to_191, xor_192_to_207, 12), veorq_u8(vextq_u8(xor_192_to_207, xor_208_to_223, 11), veorq_u8(vextq_u8(xor_208_to_223, xor_224_to_239, 10), veorq_u8(vextq_u8(xor_224_to_239, xor_240_to_255, 9), temp1))));
    // 263, 278, 293, 308
    temp1 = veorq_u8(vextq_u8(xor_240_to_255, xor_256_to_271, 8), veorq_u8(vextq_u8(xor_256_to_271, xor_272_to_287, 7), veorq_u8(vextq_u8(xor_272_to_287, xor_288_to_303, 6), veorq_u8(vextq_u8(xor_288_to_303, xor_304_to_319, 5), temp1))));
    // 323, 338, 353, 368, 383
    temp1 = veorq_u8(vextq_u8(xor_304_to_319, xor_320_to_335, 4), veorq_u8(vextq_u8(xor_320_to_335, xor_336_to_351, 3), veorq_u8(vextq_u8(xor_336_to_351, xor_352_to_367, 2), veorq_u8(vextq_u8(xor_352_to_367, xor_368_to_383, 1), veorq_u8(xor_368_to_383, temp1)))));
    temp1 = veorq_u8(temp1, vld1q_u8(c1 + 383));
    vst1q_u8(dest_high + 383, temp1);

    memcpy(dest_high + 1, pc_m384_high + 1, 383 * sizeof(uint8_t));
    memcpy(dest_high + 639, c1 + 639, 3457 * sizeof(uint8_t));
}

void combine_part_9215(uint8_t* dest_high, const uint8_t * pc_m1024_high, const uint8_t *c1)
{
    dest_high[0] = c1[0];
    uint8_t temp_high[1024];
    for(int i = 0; i < 1024; i = i + 16)
    {
        uint8x16_t b1 = vld1q_u8(c1 + i);
        uint8x16_t b2 = vld1q_u8(pc_m1024_high + i);
        vst1q_u8(temp_high + i, veorq_u8(b1, b2));
    }

    uint8x16_t initial = vld1q_u8(temp_high+1);
    uint8x16_t temp1_1, temp1_2, temp1_3, temp1_4, temp1_5, temp1_6, temp1_7, temp1_8;
    uint8x16_t temp2_1, temp2_2, temp2_3, temp2_4, temp2_5, temp2_6, temp2_7, temp2_8;
    uint8x16_t temp3_1, temp3_2, temp3_3, temp3_4, temp3_5, temp3_6, temp3_7, temp3_8;
    uint8x16_t temp4_1, temp4_2, temp4_3, temp4_4, temp4_5, temp4_6, temp4_7, temp4_8;
    uint8x16_t temp5_1, temp5_2, temp5_3, temp5_4, temp5_5, temp5_6, temp5_7, temp5_8;
    uint8x16_t temp6_1, temp6_2, temp6_3, temp6_4, temp6_5, temp6_6, temp6_7, temp6_8;
    uint8x16_t temp7_1, temp7_2, temp7_3, temp7_4, temp7_5, temp7_6, temp7_7, temp7_8;
    uint8x16_t temp8_1, temp8_2, temp8_3, temp8_4, temp8_5, temp8_6, temp8_7, temp8_8;
    uint8x16_t temp9_1, temp9_2, temp9_3, temp9_4, temp9_5, temp9_6, temp9_7, temp9_8;
    uint8x16_t temp10_1, temp10_2, temp10_3, temp10_4, temp10_5, temp10_6, temp10_7, temp10_8;
    uint8x16_t temp11_1, temp11_2, temp11_3, temp11_4, temp11_5, temp11_6, temp11_7, temp11_8;
    uint8x16_t temp12_1, temp12_2, temp12_3, temp12_4, temp12_5, temp12_6, temp12_7, temp12_8;
    uint8x16_t temp13_1, temp13_2, temp13_3, temp13_4, temp13_5, temp13_6, temp13_7, temp13_8;
    uint8x16_t temp14_1, temp14_2, temp14_3, temp14_4, temp14_5, temp14_6, temp14_7, temp14_8;
    uint8x16_t temp15_1, temp15_2, temp15_3, temp15_4, temp15_5, temp15_6, temp15_7, temp15_8;
    uint8x16_t temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
    uint8x16_t zero = vdupq_n_u8(0);

    temp15_1 = vextq_u8(zero, initial, 15);
    temp15_2 = vld1q_u8(temp_high + 16);
    temp15_3 = vld1q_u8(temp_high + 32);
    temp15_4 = vld1q_u8(temp_high + 48);
    temp15_5 = vld1q_u8(temp_high + 64);
    temp15_6 = vld1q_u8(temp_high + 80);
    temp15_7 = vld1q_u8(temp_high + 96);
    temp15_8 = vld1q_u8(temp_high + 112);





    // last 128 bits
    temp1 = temp15_1;
    temp2 = temp15_2;
    temp3 = temp15_3;
    temp4 = temp15_4;
    temp5 = temp15_5;
    temp6 = temp15_6;   
    temp7 = temp15_7;
    temp8 = temp15_8;
    for(int i = 1; i < 108; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + loading_index[i] - 127));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + loading_index[i] - 111));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + loading_index[i] - 95));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + loading_index[i] - 79));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + loading_index[i] - 63));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + loading_index[i] - 47));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + loading_index[i] - 31));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + loading_index[i] - 15));
    }
    vst1q_u8(dest_high + 9087, temp1);
    vst1q_u8(dest_high + 4991, veorq_u8(temp1, vld1q_u8(c1 + 4991)));
    vst1q_u8(dest_high + 9103, temp2);
    vst1q_u8(dest_high + 5007, veorq_u8(temp2, vld1q_u8(c1 + 5007)));
    vst1q_u8(dest_high + 9119, temp3);
    vst1q_u8(dest_high + 5023, veorq_u8(temp3, vld1q_u8(c1 + 5023)));
    vst1q_u8(dest_high + 9135, temp4);
    vst1q_u8(dest_high + 5039, veorq_u8(temp4, vld1q_u8(c1 + 5039)));
    vst1q_u8(dest_high + 9151, temp5);
    vst1q_u8(dest_high + 5055, veorq_u8(temp5, vld1q_u8(c1 + 5055)));
    vst1q_u8(dest_high + 9167, temp6);
    vst1q_u8(dest_high + 5071, veorq_u8(temp6, vld1q_u8(c1 + 5071)));
    vst1q_u8(dest_high + 9183, temp7);
    vst1q_u8(dest_high + 5087, veorq_u8(temp7, vld1q_u8(c1 + 5087)));
    vst1q_u8(dest_high + 9199, temp8);
    vst1q_u8(dest_high + 5103, veorq_u8(temp8, vld1q_u8(c1 + 5103)));

    vst1q_u8(dest_high + 1407, veorq_u8(temp1, vld1q_u8(c1 + 1407)));
    vst1q_u8(dest_high + 1423, veorq_u8(temp2, vld1q_u8(c1 + 1423)));
    vst1q_u8(dest_high + 1439, veorq_u8(temp3, vld1q_u8(c1 + 1439)));
    vst1q_u8(dest_high + 1455, veorq_u8(temp4, vld1q_u8(c1 + 1455)));
    vst1q_u8(dest_high + 1471, veorq_u8(temp5, vld1q_u8(c1 + 1471)));
    vst1q_u8(dest_high + 1487, veorq_u8(temp6, vld1q_u8(c1 + 1487)));
    vst1q_u8(dest_high + 1503, veorq_u8(temp7, vld1q_u8(c1 + 1503)));
    vst1q_u8(dest_high + 1519, veorq_u8(temp8, vld1q_u8(c1 + 1519)));
    

    // last 256 to 129 bits
    temp11_1 = vextq_u8(zero, initial, 11);
    temp11_2 = vld1q_u8(temp_high + 12);
    temp11_3 = vld1q_u8(temp_high + 28);
    temp11_4 = vld1q_u8(temp_high + 44);
    temp11_5 = vld1q_u8(temp_high + 60);
    temp11_6 = vld1q_u8(temp_high + 76);
    temp11_7 = vld1q_u8(temp_high + 92);
    temp11_8 = vld1q_u8(temp_high + 108);

    temp7_1 = vextq_u8(zero, initial, 7);
    temp7_2 = vld1q_u8(temp_high + 8);
    temp7_3 = vld1q_u8(temp_high + 24);
    temp7_4 = vld1q_u8(temp_high + 40);
    temp7_5 = vld1q_u8(temp_high + 56);
    temp7_6 = vld1q_u8(temp_high + 72);
    temp7_7 = vld1q_u8(temp_high + 88);
    temp7_8 = vld1q_u8(temp_high + 104);

    temp1 = veorq_u8(temp15_1, veorq_u8(temp11_1, temp7_1));
    temp2 = veorq_u8(temp15_2, veorq_u8(temp11_2, temp7_2));
    temp3 = veorq_u8(temp15_3, veorq_u8(temp11_3, temp7_3));
    temp4 = veorq_u8(temp15_4, veorq_u8(temp11_4, temp7_4));
    temp5 = veorq_u8(temp15_1, veorq_u8(temp15_5, veorq_u8(temp11_5, temp7_5)));
    temp6 = veorq_u8(temp15_2, veorq_u8(temp15_6, veorq_u8(temp11_6, temp7_6)));
    temp7 = veorq_u8(temp15_3, veorq_u8(temp15_7, veorq_u8(temp11_7, temp7_7)));
    temp8 = veorq_u8(temp15_4, veorq_u8(temp15_8, veorq_u8(temp11_8, temp7_8)));
    for(int i = 5; i < 108; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + loading_index[i] - 255));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + loading_index[i] - 239));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + loading_index[i] - 223));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + loading_index[i] - 207));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + loading_index[i] - 191));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + loading_index[i] - 175));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + loading_index[i] - 159));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + loading_index[i] - 143));
    }
    vst1q_u8(dest_high + 8959, temp1);
    vst1q_u8(dest_high + 4863, veorq_u8(temp1, vld1q_u8(c1 + 4863)));
    vst1q_u8(dest_high + 8975, temp2);
    vst1q_u8(dest_high + 4879, veorq_u8(temp2, vld1q_u8(c1 + 4879)));
    vst1q_u8(dest_high + 8991, temp3);
    vst1q_u8(dest_high + 4895, veorq_u8(temp3, vld1q_u8(c1 + 4895)));
    vst1q_u8(dest_high + 9007, temp4);
    vst1q_u8(dest_high + 4911, veorq_u8(temp4, vld1q_u8(c1 + 4911)));
    vst1q_u8(dest_high + 9023, temp5);
    vst1q_u8(dest_high + 4927, veorq_u8(temp5, vld1q_u8(c1 + 4927)));
    vst1q_u8(dest_high + 9039, temp6);
    vst1q_u8(dest_high + 4943, veorq_u8(temp6, vld1q_u8(c1 + 4943)));
    vst1q_u8(dest_high + 9055, temp7);
    vst1q_u8(dest_high + 4959, veorq_u8(temp7, vld1q_u8(c1 + 4959)));
    vst1q_u8(dest_high + 9071, temp8);
    vst1q_u8(dest_high + 4975, veorq_u8(temp8, vld1q_u8(c1 + 4975)));

    vst1q_u8(dest_high + 1279, veorq_u8(temp1, vld1q_u8(c1 + 1279)));
    vst1q_u8(dest_high + 1295, veorq_u8(temp2, vld1q_u8(c1 + 1295)));
    vst1q_u8(dest_high + 1311, veorq_u8(temp3, vld1q_u8(c1 + 1311)));
    vst1q_u8(dest_high + 1327, veorq_u8(temp4, vld1q_u8(c1 + 1327)));
    vst1q_u8(dest_high + 1343, veorq_u8(temp5, vld1q_u8(c1 + 1343)));
    vst1q_u8(dest_high + 1359, veorq_u8(temp6, vld1q_u8(c1 + 1359)));
    vst1q_u8(dest_high + 1375, veorq_u8(temp7, vld1q_u8(c1 + 1375)));
    vst1q_u8(dest_high + 1391, veorq_u8(temp8, vld1q_u8(c1 + 1391)));

    // last 384 to 257 bits

    temp14_1 = vextq_u8(zero, initial, 14);
    temp14_2 = vld1q_u8(temp_high + 15);
    temp14_3 = vld1q_u8(temp_high + 31);
    temp14_4 = vld1q_u8(temp_high + 47);
    temp14_5 = vld1q_u8(temp_high + 63);
    temp14_6 = vld1q_u8(temp_high + 79);
    temp14_7 = vld1q_u8(temp_high + 95);
    temp14_8 = vld1q_u8(temp_high + 111);

    temp1 = veorq_u8(temp14_1, veorq_u8(temp15_1, temp11_1));
    temp2 = veorq_u8(temp15_1, veorq_u8(temp14_2, veorq_u8(temp15_2, temp11_2)));
    temp3 = veorq_u8(temp15_2, veorq_u8(temp14_3, veorq_u8(temp15_3, temp11_3)));
    temp4 = veorq_u8(temp15_3, veorq_u8(temp14_4, veorq_u8(temp15_4, temp11_4)));
    temp5 = veorq_u8(temp15_1, veorq_u8(temp15_4, veorq_u8(temp14_5, veorq_u8(temp15_5, temp11_5))));
    temp6 = veorq_u8(temp15_2, veorq_u8(temp15_5, veorq_u8(temp14_6, veorq_u8(temp15_6, temp11_6))));
    temp7 = veorq_u8(temp15_3, veorq_u8(temp15_6, veorq_u8(temp14_7, veorq_u8(temp15_7, temp11_7))));
    temp8 = veorq_u8(temp15_4, veorq_u8(temp15_7, veorq_u8(temp14_8, veorq_u8(temp15_8, temp11_8))));

    for(int i = 10; i < 108; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + loading_index[i] - 383));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + loading_index[i] - 367));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + loading_index[i] - 351));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + loading_index[i] - 335));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + loading_index[i] - 319));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + loading_index[i] - 303));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + loading_index[i] - 287));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + loading_index[i] - 271));
        
    }
    vst1q_u8(dest_high + 8831, temp1);
    vst1q_u8(dest_high + 4735, veorq_u8(temp1, vld1q_u8(c1 + 4735)));
    vst1q_u8(dest_high + 8847, temp2);
    vst1q_u8(dest_high + 4751, veorq_u8(temp2, vld1q_u8(c1 + 4751)));
    vst1q_u8(dest_high + 8863, temp3);
    vst1q_u8(dest_high + 4767, veorq_u8(temp3, vld1q_u8(c1 + 4767)));
    vst1q_u8(dest_high + 8879, temp4);
    vst1q_u8(dest_high + 4783, veorq_u8(temp4, vld1q_u8(c1 + 4783)));
    vst1q_u8(dest_high + 8895, temp5);
    vst1q_u8(dest_high + 4799, veorq_u8(temp5, vld1q_u8(c1 + 4799)));
    vst1q_u8(dest_high + 8911, temp6);
    vst1q_u8(dest_high + 4815, veorq_u8(temp6, vld1q_u8(c1 + 4815)));
    vst1q_u8(dest_high + 8927, temp7);
    vst1q_u8(dest_high + 4831, veorq_u8(temp7, vld1q_u8(c1 + 4831)));
    vst1q_u8(dest_high + 8943, temp8);
    vst1q_u8(dest_high + 4847, veorq_u8(temp8, vld1q_u8(c1 + 4847)));



    // last 512 to 387 bits
    temp10_1 = vextq_u8(zero, initial, 10);
    temp10_2 = vld1q_u8(temp_high + 11);
    temp10_3 = vld1q_u8(temp_high + 27);
    temp10_4 = vld1q_u8(temp_high + 43);
    temp10_5 = vld1q_u8(temp_high + 59);
    temp10_6 = vld1q_u8(temp_high + 75);
    temp10_7 = vld1q_u8(temp_high + 91);
    temp10_8 = vld1q_u8(temp_high + 107);

    temp6_1 = vextq_u8(zero, initial, 6);
    temp6_2 = vld1q_u8(temp_high + 7);
    temp6_3 = vld1q_u8(temp_high + 23);
    temp6_4 = vld1q_u8(temp_high + 39);
    temp6_5 = vld1q_u8(temp_high + 55);
    temp6_6 = vld1q_u8(temp_high + 71);
    temp6_7 = vld1q_u8(temp_high + 87);
    temp6_8 = vld1q_u8(temp_high + 103);

    temp3_1 = vextq_u8(zero, initial, 3);
    temp3_2 = vld1q_u8(temp_high + 4);
    temp3_3 = vld1q_u8(temp_high + 20);
    temp3_4 = vld1q_u8(temp_high + 36);
    temp3_5 = vld1q_u8(temp_high + 52);
    temp3_6 = vld1q_u8(temp_high + 68);
    temp3_7 = vld1q_u8(temp_high + 84);
    temp3_8 = vld1q_u8(temp_high + 100);

    temp1 = veorq_u8(temp3_1, veorq_u8(temp6_1, veorq_u8(temp7_1, veorq_u8(temp10_1, veorq_u8(temp11_1, veorq_u8(temp14_1, temp15_1))))));
    temp2 = veorq_u8(temp7_1, veorq_u8(temp11_1, veorq_u8(temp15_1, veorq_u8(temp3_2, veorq_u8(temp6_2, veorq_u8(temp7_2, veorq_u8(temp10_2, veorq_u8(temp11_2, veorq_u8(temp14_2, temp15_2)))))))));
    temp3 = veorq_u8(temp7_2, veorq_u8(temp11_2, veorq_u8(temp15_2, veorq_u8(temp3_3, veorq_u8(temp6_3, veorq_u8(temp7_3, veorq_u8(temp10_3, veorq_u8(temp11_3, veorq_u8(temp14_3, temp15_3)))))))));
    temp4 = veorq_u8(temp7_3, veorq_u8(temp11_3, veorq_u8(temp15_3, veorq_u8(temp3_4, veorq_u8(temp6_4, veorq_u8(temp7_4, veorq_u8(temp10_4, veorq_u8(temp11_4, veorq_u8(temp14_4, temp15_4)))))))));
    temp5 = veorq_u8(temp7_1, veorq_u8(temp14_1, veorq_u8(temp15_1, veorq_u8(temp7_4, veorq_u8(temp11_4, veorq_u8(temp15_4, veorq_u8(temp3_5, veorq_u8(temp6_5, veorq_u8(temp7_5, veorq_u8(temp10_5, veorq_u8(temp11_5, veorq_u8(temp14_5, temp15_5))))))))))));
    temp6 = veorq_u8(temp15_1, veorq_u8(temp7_2, veorq_u8(temp14_2, veorq_u8(temp15_2, veorq_u8(temp7_5, veorq_u8(temp11_5, veorq_u8(temp15_5, veorq_u8(temp3_6, veorq_u8(temp6_6, veorq_u8(temp7_6, veorq_u8(temp10_6, veorq_u8(temp11_6, veorq_u8(temp14_6, temp15_6)))))))))))));
    temp7 = veorq_u8(temp15_2, veorq_u8(temp7_3, veorq_u8(temp14_3, veorq_u8(temp15_3, veorq_u8(temp7_6, veorq_u8(temp11_6, veorq_u8(temp15_6, veorq_u8(temp3_7, veorq_u8(temp6_7, veorq_u8(temp7_7, veorq_u8(temp10_7, veorq_u8(temp11_7, veorq_u8(temp14_7, temp15_7)))))))))))));
    temp8 = veorq_u8(temp15_3, veorq_u8(temp7_4, veorq_u8(temp14_4, veorq_u8(temp15_4, veorq_u8(temp7_7, veorq_u8(temp11_7, veorq_u8(temp15_7, veorq_u8(temp3_8, veorq_u8(temp6_8, veorq_u8(temp7_8, veorq_u8(temp10_8, veorq_u8(temp11_8, veorq_u8(temp14_8, temp15_8)))))))))))));
    for(int i = 24; i < 108; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + loading_index[i] - 511));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + loading_index[i] - 495));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + loading_index[i] - 479));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + loading_index[i] - 463));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + loading_index[i] - 447));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + loading_index[i] - 431));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + loading_index[i] - 415));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + loading_index[i] - 399));
    }
    vst1q_u8(dest_high + 8703, temp1);
    vst1q_u8(dest_high + 4607, veorq_u8(temp1, vld1q_u8(c1 + 4607)));
    vst1q_u8(dest_high + 8719, temp2);
    vst1q_u8(dest_high + 4623, veorq_u8(temp2, vld1q_u8(c1 + 4623)));
    vst1q_u8(dest_high + 8735, temp3);
    vst1q_u8(dest_high + 4639, veorq_u8(temp3, vld1q_u8(c1 + 4639)));
    vst1q_u8(dest_high + 8751, temp4);
    vst1q_u8(dest_high + 4655, veorq_u8(temp4, vld1q_u8(c1 + 4655)));
    vst1q_u8(dest_high + 8767, temp5);
    vst1q_u8(dest_high + 4671, veorq_u8(temp5, vld1q_u8(c1 + 4671)));
    vst1q_u8(dest_high + 8783, temp6);
    vst1q_u8(dest_high + 4687, veorq_u8(temp6, vld1q_u8(c1 + 4687)));
    vst1q_u8(dest_high + 8799, temp7);
    vst1q_u8(dest_high + 4703, veorq_u8(temp7, vld1q_u8(c1 + 4703)));
    vst1q_u8(dest_high + 8815, temp8);
    vst1q_u8(dest_high + 4719, veorq_u8(temp8, vld1q_u8(c1 + 4719)));

    // last 640 to 513 bits

    temp13_1 = vextq_u8(zero, initial, 13);
    temp13_2 = vld1q_u8(temp_high + 14);
    temp13_3 = vld1q_u8(temp_high + 30);
    temp13_4 = vld1q_u8(temp_high + 46);
    temp13_5 = vld1q_u8(temp_high + 62);
    temp13_6 = vld1q_u8(temp_high + 78);
    temp13_7 = vld1q_u8(temp_high + 94);
    temp13_8 = vld1q_u8(temp_high + 110);

    temp1 = veorq_u8(temp11_1, veorq_u8(temp13_1, temp15_1));
    temp2 = veorq_u8(temp11_2, veorq_u8(temp13_2, temp15_2));
    temp3 = veorq_u8(temp15_1, veorq_u8(temp11_3, veorq_u8(temp13_3, temp15_3)));
    temp4 = veorq_u8(temp15_2, veorq_u8(temp11_4, veorq_u8(temp13_4, temp15_4)));
    temp5 = veorq_u8(temp15_1, veorq_u8(temp15_3, veorq_u8(temp11_5, veorq_u8(temp13_5, temp15_5))));
    temp6 = veorq_u8(temp15_2, veorq_u8(temp15_4, veorq_u8(temp11_6, veorq_u8(temp13_6, temp15_6))));
    temp7 = veorq_u8(temp15_3, veorq_u8(temp15_5, veorq_u8(temp11_7, veorq_u8(temp13_7, temp15_7))));
    temp8 = veorq_u8(temp15_4, veorq_u8(temp15_6, veorq_u8(temp11_8, veorq_u8(temp13_8, temp15_8))));

    for(int i = 29; i < 108; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + loading_index[i] - 639));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + loading_index[i] - 623));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + loading_index[i] - 607));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + loading_index[i] - 591));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + loading_index[i] - 575));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + loading_index[i] - 559));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + loading_index[i] - 543));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + loading_index[i] - 527));
    }
    vst1q_u8(dest_high + 8575, temp1);
    vst1q_u8(dest_high + 4479, veorq_u8(temp1, vld1q_u8(c1 + 4479)));
    vst1q_u8(dest_high + 8591, temp2);
    vst1q_u8(dest_high + 4495, veorq_u8(temp2, vld1q_u8(c1 + 4495)));
    vst1q_u8(dest_high + 8607, temp3);
    vst1q_u8(dest_high + 4511, veorq_u8(temp3, vld1q_u8(c1 + 4511)));
    vst1q_u8(dest_high + 8623, temp4);
    vst1q_u8(dest_high + 4527, veorq_u8(temp4, vld1q_u8(c1 + 4527)));
    vst1q_u8(dest_high + 8639, temp5);
    vst1q_u8(dest_high + 4543, veorq_u8(temp5, vld1q_u8(c1 + 4543)));
    vst1q_u8(dest_high + 8655, temp6);
    vst1q_u8(dest_high + 4559, veorq_u8(temp6, vld1q_u8(c1 + 4559)));
    vst1q_u8(dest_high + 8671, temp7);
    vst1q_u8(dest_high + 4575, veorq_u8(temp7, vld1q_u8(c1 + 4575)));
    vst1q_u8(dest_high + 8687, temp8);
    vst1q_u8(dest_high + 4591, veorq_u8(temp8, vld1q_u8(c1 + 4591)));

    // last 768 to 641 bits
    temp5_1 = vextq_u8(zero, initial, 5);
    temp5_2 = vld1q_u8(temp_high + 6);
    temp5_3 = vld1q_u8(temp_high + 22);
    temp5_4 = vld1q_u8(temp_high + 38);
    temp5_5 = vld1q_u8(temp_high + 54);
    temp5_6 = vld1q_u8(temp_high + 70);
    temp5_7 = vld1q_u8(temp_high + 86);
    temp5_8 = vld1q_u8(temp_high + 102);

    temp9_1 = vextq_u8(zero, initial, 9);
    temp9_2 = vld1q_u8(temp_high + 10);
    temp9_3 = vld1q_u8(temp_high + 26);
    temp9_4 = vld1q_u8(temp_high + 42);
    temp9_5 = vld1q_u8(temp_high + 58);
    temp9_6 = vld1q_u8(temp_high + 74);
    temp9_7 = vld1q_u8(temp_high + 90);
    temp9_8 = vld1q_u8(temp_high + 106);

    temp1 = veorq_u8(temp3_1, veorq_u8(temp5_1, veorq_u8(temp7_1, veorq_u8(temp9_1, veorq_u8(temp11_1, veorq_u8(temp13_1, temp15_1))))));
    temp2 = veorq_u8(temp3_2, veorq_u8(temp5_2, veorq_u8(temp7_2, veorq_u8(temp9_2, veorq_u8(temp11_2, veorq_u8(temp13_2, temp15_2))))));
    temp3 = veorq_u8(temp7_1, veorq_u8(temp11_1, veorq_u8(temp15_1, veorq_u8(temp3_3, veorq_u8(temp5_3, veorq_u8(temp7_3, veorq_u8(temp9_3, veorq_u8(temp11_3, veorq_u8(temp13_3, temp15_3)))))))));
    temp4 = veorq_u8(temp7_2, veorq_u8(temp11_2, veorq_u8(temp15_2, veorq_u8(temp3_4, veorq_u8(temp5_4, veorq_u8(temp7_4, veorq_u8(temp9_4, veorq_u8(temp11_4, veorq_u8(temp13_4, temp15_4)))))))));
    temp5 = veorq_u8(temp7_1, veorq_u8(temp13_1, veorq_u8(temp15_1, veorq_u8(temp7_3, veorq_u8(temp11_3, veorq_u8(temp15_3, veorq_u8(temp3_5, veorq_u8(temp5_5, veorq_u8(temp7_5, veorq_u8(temp9_5, veorq_u8(temp11_5, veorq_u8(temp13_5, temp15_5))))))))))));
    temp6 = veorq_u8(temp7_2, veorq_u8(temp13_2, veorq_u8(temp15_2, veorq_u8(temp7_4, veorq_u8(temp11_4, veorq_u8(temp15_4, veorq_u8(temp3_6, veorq_u8(temp5_6, veorq_u8(temp7_6, veorq_u8(temp9_6, veorq_u8(temp11_6, veorq_u8(temp13_6, temp15_6))))))))))));
    temp7 = veorq_u8(temp15_1, veorq_u8(temp7_3, veorq_u8(temp13_3, veorq_u8(temp15_3, veorq_u8(temp7_5, veorq_u8(temp11_5, veorq_u8(temp15_5, veorq_u8(temp3_7, veorq_u8(temp5_7, veorq_u8(temp7_7, veorq_u8(temp9_7, veorq_u8(temp11_7, veorq_u8(temp13_7, temp15_7)))))))))))));
    temp8 = veorq_u8(temp15_2, veorq_u8(temp7_4, veorq_u8(temp13_4, veorq_u8(temp15_4, veorq_u8(temp7_6, veorq_u8(temp11_6, veorq_u8(temp15_6, veorq_u8(temp3_8, veorq_u8(temp5_8, veorq_u8(temp7_8, veorq_u8(temp9_8, veorq_u8(temp11_8, veorq_u8(temp13_8, temp15_8)))))))))))));
    
    for(int i = 43; i < 108; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + loading_index[i] - 767));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + loading_index[i] - 751));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + loading_index[i] - 735));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + loading_index[i] - 719));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + loading_index[i] - 703));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + loading_index[i] - 687));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + loading_index[i] - 671));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + loading_index[i] - 655));
    }
    vst1q_u8(dest_high + 8447, temp1);
    vst1q_u8(dest_high + 4351, veorq_u8(temp1, vld1q_u8(c1 + 4351)));
    vst1q_u8(dest_high + 8463, temp2);
    vst1q_u8(dest_high + 4367, veorq_u8(temp2, vld1q_u8(c1 + 4367)));
    vst1q_u8(dest_high + 8479, temp3);
    vst1q_u8(dest_high + 4383, veorq_u8(temp3, vld1q_u8(c1 + 4383)));
    vst1q_u8(dest_high + 8495, temp4);
    vst1q_u8(dest_high + 4399, veorq_u8(temp4, vld1q_u8(c1 + 4399)));
    vst1q_u8(dest_high + 8511, temp5);
    vst1q_u8(dest_high + 4415, veorq_u8(temp5, vld1q_u8(c1 + 4415)));
    vst1q_u8(dest_high + 8527, temp6);
    vst1q_u8(dest_high + 4431, veorq_u8(temp6, vld1q_u8(c1 + 4431)));
    vst1q_u8(dest_high + 8543, temp7);
    vst1q_u8(dest_high + 4447, veorq_u8(temp7, vld1q_u8(c1 + 4447)));
    vst1q_u8(dest_high + 8559, temp8);
    vst1q_u8(dest_high + 4463, veorq_u8(temp8, vld1q_u8(c1 + 4463)));


    // last 896 to 769 bits
    temp12_1 = vextq_u8(zero, initial, 12);
    temp12_2 = vld1q_u8(temp_high + 13);
    temp12_3 = vld1q_u8(temp_high + 29);
    temp12_4 = vld1q_u8(temp_high + 45);
    temp12_5 = vld1q_u8(temp_high + 61);
    temp12_6 = vld1q_u8(temp_high + 77);
    temp12_7 = vld1q_u8(temp_high + 93);
    temp12_8 = vld1q_u8(temp_high + 109);

    // xor 9, 10, 11, 12, 13, 14, 15
    temp1 = veorq_u8(temp9_1, veorq_u8(temp10_1, veorq_u8(temp11_1, veorq_u8(temp12_1, veorq_u8(temp13_1, veorq_u8(temp14_1, temp15_1))))));
    temp2 = veorq_u8(temp11_1, veorq_u8(temp13_1, veorq_u8(temp15_1, veorq_u8(temp9_2, veorq_u8(temp10_2, veorq_u8(temp11_2, veorq_u8(temp12_2, veorq_u8(temp13_2, veorq_u8(temp14_2, temp15_2)))))))));
    temp3 = veorq_u8(temp11_1, veorq_u8(temp14_1, veorq_u8(temp15_1, veorq_u8(temp11_2, veorq_u8(temp13_2, veorq_u8(temp15_2, veorq_u8(temp9_3, veorq_u8(temp10_3, veorq_u8(temp11_3, veorq_u8(temp12_3, veorq_u8(temp13_3, veorq_u8(temp14_3, temp15_3))))))))))));
    temp4 = veorq_u8(temp15_1, veorq_u8(temp11_2, veorq_u8(temp14_2, veorq_u8(temp15_2, veorq_u8(temp11_3, veorq_u8(temp13_3, veorq_u8(temp15_3, veorq_u8(temp9_4, veorq_u8(temp10_4, veorq_u8(temp11_4, veorq_u8(temp12_4, veorq_u8(temp13_4, veorq_u8(temp14_4, temp15_4)))))))))))));
    temp5 = veorq_u8(temp13_1, veorq_u8(temp14_1, veorq_u8(temp15_1, veorq_u8(temp15_2, veorq_u8(temp11_3, veorq_u8(temp14_3, veorq_u8(temp15_3, veorq_u8(temp11_4, veorq_u8(temp13_4, veorq_u8(temp15_4, veorq_u8(temp9_5, veorq_u8(temp10_5, veorq_u8(temp11_5, veorq_u8(temp12_5, veorq_u8(temp13_5, veorq_u8(temp14_5, temp15_5))))))))))))))));
    temp6 = veorq_u8(temp15_1, veorq_u8(temp13_2, veorq_u8(temp14_2, veorq_u8(temp15_2, veorq_u8(temp15_3, veorq_u8(temp11_4, veorq_u8(temp14_4, veorq_u8(temp15_4, veorq_u8(temp11_5, veorq_u8(temp13_5, veorq_u8(temp15_5, veorq_u8(temp9_6, veorq_u8(temp10_6, veorq_u8(temp11_6, veorq_u8(temp12_6, veorq_u8(temp13_6, veorq_u8(temp14_6, temp15_6)))))))))))))))));
    temp7 = veorq_u8(temp15_1, veorq_u8(temp15_2, veorq_u8(temp13_3, veorq_u8(temp14_3, veorq_u8(temp15_3, veorq_u8(temp15_4, veorq_u8(temp11_5, veorq_u8(temp14_5, veorq_u8(temp15_5, veorq_u8(temp11_6, veorq_u8(temp13_6, veorq_u8(temp15_6, veorq_u8(temp9_7, veorq_u8(temp10_7, veorq_u8(temp11_7, veorq_u8(temp12_7, veorq_u8(temp13_7, veorq_u8(temp14_7, temp15_7))))))))))))))))));
    temp8 = veorq_u8(temp15_2, veorq_u8(temp15_3, veorq_u8(temp13_4, veorq_u8(temp14_4, veorq_u8(temp15_4, veorq_u8(temp15_5, veorq_u8(temp11_6, veorq_u8(temp14_6, veorq_u8(temp15_6, veorq_u8(temp11_7, veorq_u8(temp13_7, veorq_u8(temp15_7, veorq_u8(temp9_8, veorq_u8(temp10_8, veorq_u8(temp11_8, veorq_u8(temp12_8, veorq_u8(temp13_8, veorq_u8(temp14_8, temp15_8))))))))))))))))));

    for(int i = 62; i < 108; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + loading_index[i] - 895));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + loading_index[i] - 879));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + loading_index[i] - 863));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + loading_index[i] - 847));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + loading_index[i] - 831));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + loading_index[i] - 815));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + loading_index[i] - 799));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + loading_index[i] - 783));
    }
    vst1q_u8(dest_high + 8319, temp1);
    vst1q_u8(dest_high + 4223, veorq_u8(temp1, vld1q_u8(c1 + 4223)));
    vst1q_u8(dest_high + 8335, temp2);
    vst1q_u8(dest_high + 4239, veorq_u8(temp2, vld1q_u8(c1 + 4239)));
    vst1q_u8(dest_high + 8351, temp3);
    vst1q_u8(dest_high + 4255, veorq_u8(temp3, vld1q_u8(c1 + 4255)));
    vst1q_u8(dest_high + 8367, temp4);
    vst1q_u8(dest_high + 4271, veorq_u8(temp4, vld1q_u8(c1 + 4271)));
    vst1q_u8(dest_high + 8383, temp5);
    vst1q_u8(dest_high + 4287, veorq_u8(temp5, vld1q_u8(c1 + 4287)));
    vst1q_u8(dest_high + 8399, temp6);
    vst1q_u8(dest_high + 4303, veorq_u8(temp6, vld1q_u8(c1 + 4303)));
    vst1q_u8(dest_high + 8415, temp7);
    vst1q_u8(dest_high + 4319, veorq_u8(temp7, vld1q_u8(c1 + 4319)));
    vst1q_u8(dest_high + 8431, temp8);
    vst1q_u8(dest_high + 4335, veorq_u8(temp8, vld1q_u8(c1 + 4335)));
    // last 1024 to 897 bits
    temp8_1 = vextq_u8(zero, initial, 8);
    temp8_2 = vld1q_u8(temp_high + 9);
    temp8_3 = vld1q_u8(temp_high + 25);
    temp8_4 = vld1q_u8(temp_high + 41);
    temp8_5 = vld1q_u8(temp_high + 57);
    temp8_6 = vld1q_u8(temp_high + 73);
    temp8_7 = vld1q_u8(temp_high + 89);
    temp8_8 = vld1q_u8(temp_high + 105);

    temp4_1 = vextq_u8(zero, initial, 4);
    temp4_2 = vld1q_u8(temp_high + 5);
    temp4_3 = vld1q_u8(temp_high + 21);
    temp4_4 = vld1q_u8(temp_high + 37);
    temp4_5 = vld1q_u8(temp_high + 53);
    temp4_6 = vld1q_u8(temp_high + 69);
    temp4_7 = vld1q_u8(temp_high + 85);
    temp4_8 = vld1q_u8(temp_high + 101);

    temp2_1 = vextq_u8(zero, initial, 2);
    temp2_2 = vld1q_u8(temp_high + 3);
    temp2_3 = vld1q_u8(temp_high + 19);
    temp2_4 = vld1q_u8(temp_high + 35);
    temp2_5 = vld1q_u8(temp_high + 51);
    temp2_6 = vld1q_u8(temp_high + 67);
    temp2_7 = vld1q_u8(temp_high + 83);
    temp2_8 = vld1q_u8(temp_high + 99);

    temp1_1 = vextq_u8(zero, initial, 1);
    temp1_2 = vld1q_u8(temp_high + 2);
    temp1_3 = vld1q_u8(temp_high + 18);
    temp1_4 = vld1q_u8(temp_high + 34);
    temp1_5 = vld1q_u8(temp_high + 50);
    temp1_6 = vld1q_u8(temp_high + 66);
    temp1_7 = vld1q_u8(temp_high + 82);
    temp1_8 = vld1q_u8(temp_high + 98);

    // 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    temp1 = veorq_u8(temp1_1, veorq_u8(temp2_1, veorq_u8(temp3_1, veorq_u8(temp4_1, veorq_u8(temp5_1, veorq_u8(temp6_1, veorq_u8(temp7_1, veorq_u8(temp8_1, veorq_u8(temp9_1, veorq_u8(temp10_1, veorq_u8(temp11_1, veorq_u8(temp12_1, veorq_u8(temp13_1, veorq_u8(temp14_1, temp15_1))))))))))))));
    temp2 = veorq_u8(temp1_2, veorq_u8(temp2_2, veorq_u8(temp3_2, veorq_u8(temp4_2, veorq_u8(temp5_2, veorq_u8(temp6_2, veorq_u8(temp7_2, veorq_u8(temp8_2, veorq_u8(temp9_2, veorq_u8(temp10_2, veorq_u8(temp11_2, veorq_u8(temp12_2, veorq_u8(temp13_2, veorq_u8(temp14_2, temp15_2))))))))))))));
    temp3 = veorq_u8(temp1_3, veorq_u8(temp2_3, veorq_u8(temp3_3, veorq_u8(temp4_3, veorq_u8(temp5_3, veorq_u8(temp6_3, veorq_u8(temp7_3, veorq_u8(temp8_3, veorq_u8(temp9_3, veorq_u8(temp10_3, veorq_u8(temp11_3, veorq_u8(temp12_3, veorq_u8(temp13_3, veorq_u8(temp14_3, temp15_3))))))))))))));
    temp4 = veorq_u8(temp1_4, veorq_u8(temp2_4, veorq_u8(temp3_4, veorq_u8(temp4_4, veorq_u8(temp5_4, veorq_u8(temp6_4, veorq_u8(temp7_4, veorq_u8(temp8_4, veorq_u8(temp9_4, veorq_u8(temp10_4, veorq_u8(temp11_4, veorq_u8(temp12_4, veorq_u8(temp13_4, veorq_u8(temp14_4, temp15_4))))))))))))));
    temp5 = veorq_u8(temp1_5, veorq_u8(temp2_5, veorq_u8(temp3_5, veorq_u8(temp4_5, veorq_u8(temp5_5, veorq_u8(temp6_5, veorq_u8(temp7_5, veorq_u8(temp8_5, veorq_u8(temp9_5, veorq_u8(temp10_5, veorq_u8(temp11_5, veorq_u8(temp12_5, veorq_u8(temp13_5, veorq_u8(temp14_5, temp15_5))))))))))))));
    temp6 = veorq_u8(temp1_6, veorq_u8(temp2_6, veorq_u8(temp3_6, veorq_u8(temp4_6, veorq_u8(temp5_6, veorq_u8(temp6_6, veorq_u8(temp7_6, veorq_u8(temp8_6, veorq_u8(temp9_6, veorq_u8(temp10_6, veorq_u8(temp11_6, veorq_u8(temp12_6, veorq_u8(temp13_6, veorq_u8(temp14_6, temp15_6))))))))))))));
    temp7 = veorq_u8(temp1_7, veorq_u8(temp2_7, veorq_u8(temp3_7, veorq_u8(temp4_7, veorq_u8(temp5_7, veorq_u8(temp6_7, veorq_u8(temp7_7, veorq_u8(temp8_7, veorq_u8(temp9_7, veorq_u8(temp10_7, veorq_u8(temp11_7, veorq_u8(temp12_7, veorq_u8(temp13_7, veorq_u8(temp14_7, temp15_7))))))))))))));
    temp8 = veorq_u8(temp1_8, veorq_u8(temp2_8, veorq_u8(temp3_8, veorq_u8(temp4_8, veorq_u8(temp5_8, veorq_u8(temp6_8, veorq_u8(temp7_8, veorq_u8(temp8_8, veorq_u8(temp9_8, veorq_u8(temp10_8, veorq_u8(temp11_8, veorq_u8(temp12_8, veorq_u8(temp13_8, veorq_u8(temp14_8, temp15_8))))))))))))));
    // 99, 101, 103, 105, 107, 109, 111
    temp2 = veorq_u8(temp3_1, veorq_u8(temp5_1, veorq_u8(temp7_1, veorq_u8(temp9_1, veorq_u8(temp11_1, veorq_u8(temp13_1, veorq_u8(temp15_1, temp2)))))));
    temp3 = veorq_u8(temp3_2, veorq_u8(temp5_2, veorq_u8(temp7_2, veorq_u8(temp9_2, veorq_u8(temp11_2, veorq_u8(temp13_2, veorq_u8(temp15_2, temp3)))))));
    temp4 = veorq_u8(temp3_3, veorq_u8(temp5_3, veorq_u8(temp7_3, veorq_u8(temp9_3, veorq_u8(temp11_3, veorq_u8(temp13_3, veorq_u8(temp15_3, temp4)))))));
    temp5 = veorq_u8(temp3_4, veorq_u8(temp5_4, veorq_u8(temp7_4, veorq_u8(temp9_4, veorq_u8(temp11_4, veorq_u8(temp13_4, veorq_u8(temp15_4, temp5)))))));
    temp6 = veorq_u8(temp3_5, veorq_u8(temp5_5, veorq_u8(temp7_5, veorq_u8(temp9_5, veorq_u8(temp11_5, veorq_u8(temp13_5, veorq_u8(temp15_5, temp6)))))));
    temp7 = veorq_u8(temp3_6, veorq_u8(temp5_6, veorq_u8(temp7_6, veorq_u8(temp9_6, veorq_u8(temp11_6, veorq_u8(temp13_6, veorq_u8(temp15_6, temp7)))))));
    temp8 = veorq_u8(temp3_7, veorq_u8(temp5_7, veorq_u8(temp7_7, veorq_u8(temp9_7, veorq_u8(temp11_7, veorq_u8(temp13_7, veorq_u8(temp15_7, temp8)))))));
    // 83, 86, 87, 90, 91, 94, 95
    temp3 = veorq_u8(temp3_1, veorq_u8(temp6_1, veorq_u8(temp7_1, veorq_u8(temp10_1, veorq_u8(temp11_1, veorq_u8(temp14_1, veorq_u8(temp15_1, temp3)))))));
    temp4 = veorq_u8(temp3_2, veorq_u8(temp6_2, veorq_u8(temp7_2, veorq_u8(temp10_2, veorq_u8(temp11_2, veorq_u8(temp14_2, veorq_u8(temp15_2, temp4)))))));
    temp5 = veorq_u8(temp3_3, veorq_u8(temp6_3, veorq_u8(temp7_3, veorq_u8(temp10_3, veorq_u8(temp11_3, veorq_u8(temp14_3, veorq_u8(temp15_3, temp5)))))));
    temp6 = veorq_u8(temp3_4, veorq_u8(temp6_4, veorq_u8(temp7_4, veorq_u8(temp10_4, veorq_u8(temp11_4, veorq_u8(temp14_4, veorq_u8(temp15_4, temp6)))))));
    temp7 = veorq_u8(temp3_5, veorq_u8(temp6_5, veorq_u8(temp7_5, veorq_u8(temp10_5, veorq_u8(temp11_5, veorq_u8(temp14_5, veorq_u8(temp15_5, temp7)))))));
    temp8 = veorq_u8(temp3_6, veorq_u8(temp6_6, veorq_u8(temp7_6, veorq_u8(temp10_6, veorq_u8(temp11_6, veorq_u8(temp14_6, veorq_u8(temp15_6, temp8)))))));

    // 71, 75, 79
    temp4 = veorq_u8(temp7_1, veorq_u8(temp11_1, veorq_u8(temp15_1, temp4)));
    temp5 = veorq_u8(temp7_2, veorq_u8(temp11_2, veorq_u8(temp15_2, temp5)));
    temp6 = veorq_u8(temp7_3, veorq_u8(temp11_3, veorq_u8(temp15_3, temp6)));
    temp7 = veorq_u8(temp7_4, veorq_u8(temp11_4, veorq_u8(temp15_4, temp7)));
    temp8 = veorq_u8(temp7_5, veorq_u8(temp11_5, veorq_u8(temp15_5, temp8)));

    // 53, 54, 55, 60, 61, 62, 63
    temp5 = veorq_u8(temp5_1, veorq_u8(temp6_1, veorq_u8(temp7_1, veorq_u8(temp12_1, veorq_u8(temp13_1, veorq_u8(temp14_1, veorq_u8(temp15_1, temp5)))))));
    temp6 = veorq_u8(temp5_2, veorq_u8(temp6_2, veorq_u8(temp7_2, veorq_u8(temp12_2, veorq_u8(temp13_2, veorq_u8(temp14_2, veorq_u8(temp15_2, temp6)))))));
    temp7 = veorq_u8(temp5_3, veorq_u8(temp6_3, veorq_u8(temp7_3, veorq_u8(temp12_3, veorq_u8(temp13_3, veorq_u8(temp14_3, veorq_u8(temp15_3, temp7)))))));
    temp8 = veorq_u8(temp5_4, veorq_u8(temp6_4, veorq_u8(temp7_4, veorq_u8(temp12_4, veorq_u8(temp13_4, veorq_u8(temp14_4, veorq_u8(temp15_4, temp8)))))));

    // 39, 45, 47
    temp6 = veorq_u8(temp7_1, veorq_u8(temp13_1, veorq_u8(temp15_1, temp6)));
    temp7 = veorq_u8(temp7_2, veorq_u8(temp13_2, veorq_u8(temp15_2, temp7)));
    temp8 = veorq_u8(temp7_3, veorq_u8(temp13_3, veorq_u8(temp15_3, temp8)));
    // 23, 30, 31
    temp7 = veorq_u8(temp7_1, veorq_u8(temp14_1, veorq_u8(temp15_1, temp7)));
    temp8 = veorq_u8(temp7_2, veorq_u8(temp14_2, veorq_u8(temp15_2, temp8)));

    // 15
    temp8 = veorq_u8(temp15_1, temp8);

    vst1q_u8(dest_high + 8191, temp1);
    vst1q_u8(dest_high + 4095, veorq_u8(temp1, vld1q_u8(c1 + 4095)));
    vst1q_u8(dest_high + 8207, temp2);
    vst1q_u8(dest_high + 4111, veorq_u8(temp2, vld1q_u8(c1 + 4111)));
    vst1q_u8(dest_high + 8223, temp3);
    vst1q_u8(dest_high + 4127, veorq_u8(temp3, vld1q_u8(c1 + 4127)));
    vst1q_u8(dest_high + 8239, temp4);
    vst1q_u8(dest_high + 4143, veorq_u8(temp4, vld1q_u8(c1 + 4143)));
    vst1q_u8(dest_high + 8255, temp5);
    vst1q_u8(dest_high + 4159, veorq_u8(temp5, vld1q_u8(c1 + 4159)));
    vst1q_u8(dest_high + 8271, temp6);
    vst1q_u8(dest_high + 4175, veorq_u8(temp6, vld1q_u8(c1 + 4175)));
    vst1q_u8(dest_high + 8287, temp7);
    vst1q_u8(dest_high + 4191, veorq_u8(temp7, vld1q_u8(c1 + 4191)));
    vst1q_u8(dest_high + 8303, temp8);
    vst1q_u8(dest_high + 4207, veorq_u8(temp8, vld1q_u8(c1 + 4207)));

    // handle 1055 to 1278
    // 1151 to 1278 byte
    temp1 = veorq_u8(temp11_1, temp14_1);
    temp2 = veorq_u8(temp15_1, veorq_u8(temp11_2, temp14_2));
    temp3 = veorq_u8(temp15_2, veorq_u8(temp11_3, temp14_3));
    temp4 = veorq_u8(temp15_3, veorq_u8(temp11_4, temp14_4));
    temp5 = veorq_u8(temp15_1, veorq_u8(temp15_4, veorq_u8(temp11_5, temp14_5)));
    temp6 = veorq_u8(temp15_2, veorq_u8(temp15_5, veorq_u8(temp11_6, temp14_6)));
    temp7 = veorq_u8(temp15_3, veorq_u8(temp15_6, veorq_u8(temp11_7, temp14_7)));
    temp8 = veorq_u8(temp15_4, veorq_u8(temp15_7, veorq_u8(temp11_8, temp14_8)));

    for(int i = 4; i < 145; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + another_index[i] - 127));
        temp2 = veorq_u8(temp2, vld1q_u8(temp_high + another_index[i] - 111));
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + another_index[i] - 95));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + another_index[i] - 79));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + another_index[i] - 63));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + another_index[i] - 47));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + another_index[i] - 31));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + another_index[i] - 15));
    }
    temp1 = veorq_u8(temp1, vld1q_u8(c1 + 1151));
    temp2 = veorq_u8(temp2, vld1q_u8(c1 + 1167));
    temp3 = veorq_u8(temp3, vld1q_u8(c1 + 1183));
    temp4 = veorq_u8(temp4, vld1q_u8(c1 + 1199));
    temp5 = veorq_u8(temp5, vld1q_u8(c1 + 1215));
    temp6 = veorq_u8(temp6, vld1q_u8(c1 + 1231));
    temp7 = veorq_u8(temp7, vld1q_u8(c1 + 1247));
    temp8 = veorq_u8(temp8, vld1q_u8(c1 + 1263));

    vst1q_u8(dest_high + 1151, temp1);
    vst1q_u8(dest_high + 1167, temp2);
    vst1q_u8(dest_high + 1183, temp3);
    vst1q_u8(dest_high + 1199, temp4);
    vst1q_u8(dest_high + 1215, temp5);
    vst1q_u8(dest_high + 1231, temp6);
    vst1q_u8(dest_high + 1247, temp7);
    vst1q_u8(dest_high + 1263, temp8);
    // 1055 to 1150 byte
    temp3 = vdupq_n_u8(0);
    temp4 = vdupq_n_u8(0);
    temp5 = veorq_u8(temp7_1, temp14_1);
    temp6 = veorq_u8(temp15_1, veorq_u8(temp7_2, temp14_2));
    temp7 = veorq_u8(temp15_2, veorq_u8(temp7_3, temp14_3));
    temp8 = veorq_u8(temp15_3, veorq_u8(temp7_4, temp14_4));

    for(int i = 7; i < 145; i++)
    {
        temp3 = veorq_u8(temp3, vld1q_u8(temp_high + another_index[i] - 223));
        temp4 = veorq_u8(temp4, vld1q_u8(temp_high + another_index[i] - 207));
        temp5 = veorq_u8(temp5, vld1q_u8(temp_high + another_index[i] - 191));
        temp6 = veorq_u8(temp6, vld1q_u8(temp_high + another_index[i] - 175));
        temp7 = veorq_u8(temp7, vld1q_u8(temp_high + another_index[i] - 159));
        temp8 = veorq_u8(temp8, vld1q_u8(temp_high + another_index[i] - 143));
    }
    temp3 = veorq_u8(temp3, vld1q_u8(c1 + 1055));
    temp4 = veorq_u8(temp4, vld1q_u8(c1 + 1071));
    temp5 = veorq_u8(temp5, vld1q_u8(c1 + 1087));
    temp6 = veorq_u8(temp6, vld1q_u8(c1 + 1103));
    temp7 = veorq_u8(temp7, vld1q_u8(c1 + 1119));
    temp8 = veorq_u8(temp8, vld1q_u8(c1 + 1135));

    vst1q_u8(dest_high + 1055, temp3);
    vst1q_u8(dest_high + 1071, temp4);
    vst1q_u8(dest_high + 1087, temp5);
    vst1q_u8(dest_high + 1103, temp6);
    vst1q_u8(dest_high + 1119, temp7);
    vst1q_u8(dest_high + 1135, temp8);
    // left wit 1024 to 1054 byte
    // 1039 to 1054
    temp1 = veorq_u8(temp7_1, veorq_u8(temp11_1, temp15_1)); 
    for(int i = 3; i < 180; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + new_index[i] - 15));
    }
    temp1 = veorq_u8(temp1, vld1q_u8(c1 + 1039));
    vst1q_u8(dest_high + 1039, temp1);
    // 1024 to 1038
    temp1 = veorq_u8(temp3_1, veorq_u8(temp6_1, veorq_u8(temp10_1, temp14_1)));
    for(int i = 0; i < 141; i++)
    {
        temp1 = veorq_u8(temp1, vld1q_u8(temp_high + new_index_2[i]-15));
    }
    temp1 = veorq_u8(temp1, vld1q_u8(c1 + 1023));
    vst1q_u8(dest_high + 1023, temp1);
    
    memcpy(dest_high + 1, pc_m1024_high + 1, 1023 * sizeof(uint8_t));
    memcpy(dest_high + 1535, c1 + 1535, 2560 * sizeof(uint8_t));
    memcpy(dest_high + 5119, c1 + 5119, 3073 * sizeof(uint8_t));

}


void ringmul_combine_4479( uint8_t * out , const uint8_t * pc_m384_low , const uint8_t * pc_m384_high, const uint8_t *c0, const uint8_t *c1)
{
    uint8_t dest_low[4480] = {0};
    uint8_t dest_high[4479] = {0};

    combine_part_4479(dest_high, pc_m384_high, c1);
    combine_part_4479(dest_low, pc_m384_low, c0);
    out[0] = dest_low[0];
    gf256v_add( out+1 , dest_low+1 , dest_high , 4479 );
}
void ringmul_combine_9215( uint8_t * out , const uint8_t * pc_m1024_low , const uint8_t * pc_m1024_high, 
    const uint8_t *c0, const uint8_t *c1)
{

    uint8_t dest_low[9216];
    uint8_t dest_high[9215];

    
    combine_part_9215(dest_high, pc_m1024_high, c1);
    combine_part_9215(dest_low, pc_m1024_low, c0);
    out[0] = dest_low[0];
    gf256v_add( out+1 , dest_low+1 , dest_high , 9215 );
}
