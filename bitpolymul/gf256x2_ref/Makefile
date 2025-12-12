

CC = clang
LD = clang


CFLAGS   := -O3 -std=c11 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer -g #-Werror
INCPATH  := -I/usr/local/include -I/opt/local/include -I/usr/include -I./include -I./src/common
LDFLAGS  := $(LDFLAGS)
LIBS     := -lcrypto

LIBNAME = fftpolymulgf256x2

PROJ ?= ref
#PROJ ?= gfni
#PROJ ?= neon

OS := $(shell uname -s)
ARCH := $(shell uname -m)
ifneq (,$(filter $(ARCH),arm64 aarch64))
PROJ    = neon
endif
ifeq  ($(ARCH), aarch64)
CFLAGS    +=  -flax-vector-conversions -march=native
endif

EXT_SRC_DIRS  =

ifeq ($(PROJ),ref)

EXT_SRC_DIRS  += ./ref
INCPATH      += -I./ref

else ifeq ($(PROJ),gfni)

EXT_SRC_DIRS  += ./avx2gfni
### XXX: avx2gfni/polymul.h is different from include/polymul.h
INCPATH      := $(filter-out -I./include , $(INCPATH)) -I./avx2gfni -I./include
### XXX
CFLAGS       += -mavx2 -mgfni -mpclmul -mbmi

else ifeq ($(PROJ),neon)

EXT_SRC_DIRS  += ./neon
INCPATH      += -I./neon

endif


ifdef ASAN
CFLAGS  += -fsanitize=address -fno-sanitize-recover=all
LDFLAGS += -fsanitize=address -fno-sanitize-recover=all
endif


TESTINCPATH   := $(INCPATH) -I./benchmark -I./tests

#############################

CSRC        := $(wildcard src/common/*.c) $(wildcard $(EXT_SRC_DIRS)/*.c)
SRC_O       := $(CSRC:.c=.o)
SRC_O_NODIR := $(notdir $(SRC_O))
OBJS         = $(SRC_O_NODIR)

#############################

OS := $(shell uname -s)
ARCH := $(shell uname -m)
ifeq  ($(OS), Darwin)
ifeq  ($(ARCH), arm64)
CFLAGS    += -D_APPLE_SILICON_
OBJS      += m1cycles.o
endif
endif


#############################

.INTERMEDIATE:  $(OBJS)

.PHONY: all clean


all: polymul-test polymul-knsub-addfft-test polymul-addfft-benchmark polymulcrt-test # ringmul-test ringmul-384-test

%-test: $(OBJS) %-test.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^  $(LIBS)

%-benchmark: $(OBJS) %-benchmark.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^  $(LIBS)

lib: $(OBJS)
	ar rcs lib$(LIBNAME).a $^

%.o: tests/%.c
	$(CC) $(CFLAGS) $(INCPATH) $(TESTINCPATH) -o $@ -c $<

%.o: benchmark/%.c
	$(CC) $(CFLAGS) $(INCPATH) $(TESTINCPATH) -o $@ -c $<

#m1cycles.o: benchmark/m1cycles.c
#	$(CC) $(CFLAGS) $(TESTINCPATH) -o $@ -c $<

define GEN_O
%.o: $(1)/%.c
	$(CC) $(CFLAGS) $(INCPATH) -o $$@ -c $$<
endef
$(foreach dir, $(EXT_SRC_DIRS), $(eval $(call GEN_O,$(dir))))

%.o: src/common/%.c
	$(CC) $(CFLAGS) $(INCPATH) -o $@ -c $<

%.o: src/common/%.S
	$(CC) $(CFLAGS) $(INCPATH) -o $@ -c $<

%.S: %.c
	$(CC) $(CFLAGS) -S -c  -o$@ $^


#############################

clean:
	-rm *.o *.s *.q *.a test speed *-test *-benchmark

