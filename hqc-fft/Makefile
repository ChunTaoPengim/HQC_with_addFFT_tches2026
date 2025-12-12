ifeq ($(ROOT),)
ROOT:=.
endif

CC := gcc

CFLAGS:=-O3 -Wall -Wextra

SHA3_SRC:=$(ROOT)/lib/fips202/fips202.c
SHA3_INCLUDE:=-I $(ROOT)/lib/fips202

POLYMUL_DIR  := $(ROOT)/lib/bitpolymul_ref
POLYMUL_SRCS := $(wildcard $(POLYMUL_DIR)/*.c)
POLYMUL_OBJS := $(notdir $(POLYMUL_SRCS:.c=.o))
POLYMUL_INCLUDE := -I$(POLYMUL_DIR)

SRC      := $(ROOT)/src
INCLUDE  := $(SHA3_INCLUDE) -I$(ROOT)/benchmark
LIB      :=

PROJ ?= hqc-1
#PROJ ?= hqc-3
#PROJ ?= hqc-5

#TARGET ?= ref
#TARGET ?= x86_hqcteam
TARGET ?= fft_ref
#TARGET ?= fft_x86
#TARGET ?= fft_gfni
#TARGET ?= m1_hqcport
#TARGET ?= fft_m1
#TARGET ?= a72_hqcport
#TARGET ?= fft_a72

# correct some wrong input
ifeq ($(TARGET), x86_ref)
	TARG := x86_hqcteam
else ifeq ($(TARGET), m1_ref)
	TARG := m1_hqcport
else ifeq ($(TARGET), a72_ref)
	TARG := a72_hqcport
else ifeq ($(TARGET), x86_fft)
	TARG := fft_x86
else ifeq ($(TARGET), gfni_fft)
	TARG := fft_gfni
else ifeq ($(TARGET), a72_fft)
	TARG := fft_a72
else ifeq ($(TARGET), m1_fft)
	TARG := fft_m1
else
	TARG = $(TARGET)
endif

ifeq ($(TARG), ref)

	INCLUDE += -I$(SRC)/ref -I$(SRC)/common 
	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/ref/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/ref/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/ref/hqc-5 -I$(SRC)/common/hqc-5
	endif

else ifeq ($(TARG), x86_hqcteam)
	CFLAGS+= -std=c99 -funroll-all-loops -flto -mavx -mavx2 -mbmi -mpclmul
	INCLUDE  += -I$(SRC)/x86_official/avx256 -I$(SRC)/x86_official/common -I$(SRC)/common 

	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/x86_official/avx256/hqc-1 -I$(SRC)/x86_official/common/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/x86_official/avx256/hqc-3 -I$(SRC)/x86_official/common/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/x86_official/avx256/hqc-5 -I$(SRC)/x86_official/common/hqc-5 -I$(SRC)/common/hqc-5
	endif
else ifeq ($(TARG), fft_ref)
	
	CFLAGS+= -DTEST_CACHEKEY
	INCLUDE += $(POLYMUL_INCLUDE) -I$(SRC)/cachekey -I$(SRC)/ref -I$(SRC)/common 
	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/ref/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/ref/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/ref/hqc-5 -I$(SRC)/common/hqc-5
	endif
else ifeq ($(TARG), fft_x86)
	
	POLYMUL_DIR  := $(ROOT)/lib/bitpolymul_x86
	POLYMUL_SRCS := $(wildcard $(POLYMUL_DIR)/*.c)
	POLYMUL_OBJS := $(notdir $(POLYMUL_SRCS:.c=.o))
	POLYMUL_INCLUDE := -I$(POLYMUL_DIR)

	CFLAGS+= -std=c99 -funroll-all-loops -flto -mavx -mavx2 -mbmi -mpclmul -DTEST_CACHEKEY
	INCLUDE  += $(POLYMUL_INCLUDE) -I$(SRC)/code_x86 -I$(SRC)/cachekey -I$(SRC)/x86_official/avx256 -I$(SRC)/x86_official/common -I$(SRC)/common 

	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/code_x86/avx2/hqc-1 -I$(SRC)/x86_official/avx256/hqc-1 -I$(SRC)/x86_official/common/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/code_x86/avx2/hqc-3 -I$(SRC)/x86_official/avx256/hqc-3 -I$(SRC)/x86_official/common/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/code_x86/avx2/hqc-5 -I$(SRC)/x86_official/avx256/hqc-5 -I$(SRC)/x86_official/common/hqc-5 -I$(SRC)/common/hqc-5
	endif
else ifeq ($(TARG), fft_gfni)

	POLYMUL_DIR  := $(ROOT)/lib/bitpolymul_gfni
	POLYMUL_SRCS := $(wildcard $(POLYMUL_DIR)/*.c)
	POLYMUL_OBJS := $(notdir $(POLYMUL_SRCS:.c=.o))
	POLYMUL_INCLUDE := -I$(POLYMUL_DIR)

	CFLAGS+= -std=c99 -funroll-all-loops -flto -mgfni -mavx -mavx2 -mbmi -mpclmul -DTEST_CACHEKEY
	INCLUDE  += $(POLYMUL_INCLUDE) -I$(SRC)/code_x86 -I$(SRC)/cachekey -I$(SRC)/x86_official/avx256 -I$(SRC)/x86_official/common -I$(SRC)/common 

	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/code_x86/avx2gfni/hqc-1 -I$(SRC)/x86_official/avx256/hqc-1 -I$(SRC)/x86_official/common/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/code_x86/avx2gfni/hqc-3 -I$(SRC)/x86_official/avx256/hqc-3 -I$(SRC)/x86_official/common/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/code_x86/avx2gfni/hqc-5 -I$(SRC)/x86_official/avx256/hqc-5 -I$(SRC)/x86_official/common/hqc-5 -I$(SRC)/common/hqc-5
	endif

else ifeq ($(TARG), m1_hqcport)
	CFLAGS+= -std=c99 -mcpu=apple-m1 -mtune=apple-m1 -D_APPLE_SILICON_ -pedantic
	INCLUDE += -I$(SRC)/armv8 -I$(SRC)/common

	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/armv8/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/armv8/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/armv8/hqc-5 -I$(SRC)/common/hqc-5
	endif

else ifeq ($(TARG), fft_m1)

	POLYMUL_DIR  := $(ROOT)/lib/bitpolymul_m1
	POLYMUL_SRCS := $(wildcard $(POLYMUL_DIR)/*.c)
	POLYMUL_OBJS := $(notdir $(POLYMUL_SRCS:.c=.o))
	POLYMUL_INCLUDE := -I$(POLYMUL_DIR)
	CFLAGS+= -std=c99 -mcpu=apple-m1 -mtune=apple-m1 -DTEST_CACHEKEY -D_APPLE_SILICON_ -DNEW_CODE -pedantic
	INCLUDE += $(POLYMUL_INCLUDE) -I$(SRC)/cachekey -I$(SRC)/armv8 -I$(SRC)/common

	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/armv8/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/armv8/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/armv8/hqc-5 -I$(SRC)/common/hqc-5
	endif

else ifeq ($(TARG), a72_hqcport)

	CFLAGS += -g -std=c99 -funroll-all-loops -Wpedantic -fno-omit-frame-pointer -flax-vector-conversions -march=native -DPERF_CYCLES
	INCLUDE += -I$(SRC)/a72 -I$(SRC)/armv8 -I$(SRC)/common

	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/armv8/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/armv8/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/armv8/hqc-5 -I$(SRC)/common/hqc-5
	endif
else ifeq ($(TARG), fft_a72)

	POLYMUL_DIR  := $(ROOT)/lib/bitpolymul_a72
	POLYMUL_SRCS := $(wildcard $(POLYMUL_DIR)/*.c)
	POLYMUL_OBJS := $(notdir $(POLYMUL_SRCS:.c=.o))
	POLYMUL_INCLUDE := -I$(POLYMUL_DIR)
	CFLAGS += -g -std=c99 -funroll-all-loops -Wpedantic -fno-omit-frame-pointer -flax-vector-conversions -march=native -DPERF_CYCLES -DTEST_CACHEKEY -DNEW_CODE
	INCLUDE += $(POLYMUL_INCLUDE) -I$(SRC)/cachekey -I$(SRC)/a72 -I$(SRC)/armv8 -I$(SRC)/common

	ifeq ($(PROJ),hqc-1)
	INCLUDE      += -I$(SRC)/armv8/hqc-1 -I$(SRC)/common/hqc-1
	else ifeq ($(PROJ),hqc-3)
	INCLUDE      += -I$(SRC)/armv8/hqc-3 -I$(SRC)/common/hqc-3
	else ifeq ($(PROJ),hqc-5)
	INCLUDE      += -I$(SRC)/armv8/hqc-5 -I$(SRC)/common/hqc-5
	endif
endif



#MAIN_HQC:=$(ROOT)/src/main_hqc.c
#MAIN_KAT:=$(ROOT)/src/main_kat.c

HQC_OBJS:= code.o crypto_memset.o fft.o kem.o symmetric.o gf.o gf2x.o hqc.o parsing.o reed_muller.o reed_solomon.o vector.o 
ifeq ($(filter $(TARG), fft_ref fft_x86 fft_gfni fft_m1 fft_a72),$(TARG))
HQC_OBJS+= cachekey.o
endif
#HQC_OBJS_VERBOSE:=vector.o reed_muller.o reed_solomon-verbose.o fft.o gf.o gf2x.o code-verbose.o parsing.o hqc-verbose.o kem-verbose.o shake_ds.o shake_prng.o api_cachekey.o
LIB_OBJS:= fips202.o
ifeq ($(filter $(TARG), fft_ref fft_x86 fft_gfni fft_m1 fft_a72),$(TARG))
LIB_OBJS+= $(POLYMUL_OBJS)
endif

TEST_DIR:=$(ROOT)/tests
bench_INCLUDE:=
OS := $(shell uname -s)
ARCH := $(shell uname -m)
ifeq  ($(OS), Darwin)
ifeq  ($(ARCH), arm64)
	LIB_OBJS    += m1cycles.o
	bench_src   := $(ROOT)/benchmark/m1cycles.c
	bench_INCLUDE:=-I $(ROOT)/benchmark
#TEST_DIR	:= $(ROOT)/tests_m1
endif
endif

ifeq ($(OS), Linux)
  ifeq ($(ARCH), aarch64)
	LIB_OBJS    += hal.o
    bench_INCLUDE := -I $(ROOT)/benchmark
	bench_src   := $(ROOT)/benchmark/hal.c
    TEST_DIR := $(ROOT)/tests_a72
  endif
endif

BIN:=bin
BUILD:=bin/build

all: code-test hqc-test gf2x-test

folders:
	@echo -e "\n### Creating folders\n"
	mkdir -p $(BUILD)

fips202.o: | folders
	@/bin/echo -e "\n### Compiling fips202"
	$(CC) $(CFLAGS) -c $(SHA3_SRC) $(SHA3_INCLUDE) -o $(BUILD)/$@

#ifeq ($(filter $(TARG), m1_hqcport fft_m1),$(TARG))
m1cycles.o: | folders
	@/bin/echo -e "\n### Compiling m1cycles"
	$(CC) $(CFLAGS) -c $(bench_src) $(bench_INCLUDE) -o $(BUILD)/$@
#endif

#ifeq ($(filter $(TARG), a72_hqcport fft_a72),$(TARG))
hal.o: | folders
	@/bin/echo -e "\n### Compiling hal.o"
	$(CC) $(CFLAGS) -c $(bench_src) $(bench_INCLUDE) -o $(BUILD)/$@
#endif

ifeq ($(TARG), fft_x86)
%.o: $(SRC)/code_x86/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o: $(SRC)/code_x86/avx2/$(PROJ)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

else ifeq ($(TARG), fft_gfni)
%.o: $(SRC)/code_x86/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o: $(SRC)/code_x86/avx2gfni/$(PROJ)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@
endif

ifeq ($(filter $(TARG), fft_ref fft_x86 fft_gfni fft_m1 fft_a72),$(TARG))
%.o: $(SRC)/cachekey/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@
endif

ifeq ($(filter $(TARG), ref fft_ref),$(TARG))
%.o: $(SRC)/ref/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@
endif

%.o: $(SRC)/common/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

ifeq ($(filter $(TARG), fft_ref fft_x86 fft_gfni fft_m1 fft_a72),$(TARG))
%.o: $(POLYMUL_DIR)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@
endif

ifeq ($(filter $(TARG), a72_hqcport fft_a72),$(TARG))
%.o: $(SRC)/a72/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o : $(SRC)/a72/$(PROJ)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@
endif

ifeq ($(filter $(TARG), m1_hqcport fft_m1 a72_hqcport fft_a72),$(TARG))
%.o: $(SRC)/armv8/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o : $(SRC)/armv8/$(PROJ)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@
endif

%.o: $(ROOT)/benchmark/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o: $(TEST_DIR)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

ifeq ($(filter $(TARG), x86_hqcteam fft_x86 fft_gfni),$(TARG))
%.o: $(SRC)/x86_official/avx256/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o : $(SRC)/x86_official/common/$(PROJ)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o: $(SRC)/x86_official/avx256/$(PROJ)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o: $(SRC)/x86_official/common/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@
endif

%-verbose.o: $(SRC)/%.c | folders
	@echo -e "\n### Compiling $@ (verbose mode)\n"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -D VERBOSE -o $(BUILD)/$@

hqc-128: $(HQC_OBJS) $(LIB_OBJS) | folders
	@echo -e "\n### Compiling hqc-256\n"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-128-kat: $(HQC_OBJS) $(LIB_OBJS) | folders
	@echo -e "\n### Compiling hqc-256 KAT\n"
	$(CC) $(CFLAGS) $(MAIN_KAT) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-128-verbose: $(HQC_OBJS_VERBOSE) $(LIB_OBJS) | folders
	@echo -e "\n### Compiling hqc-256 (verbose mode)\n"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -D VERBOSE -o $(BIN)/$@

gf2x-test: gf2x-test.o $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling gf2x-test.c"
	$(CC) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-test: hqc-test.o $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-test.c"
	$(CC) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

code-test: code-test.o $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling code-test.c"
	$(CC) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

%-test: %-test.o $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-test.c"
	$(CC) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@


clean:
	rm -f PQCkemKAT_*
	rm -f vgcore.*
	rm -rf ./bin
