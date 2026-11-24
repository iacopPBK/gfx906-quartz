# quartz: inference engine for two MI50 (gfx906) cards
BIN       ?= build/bin
ROCM_PATH ?= /opt/rocm
HIPCC     ?= $(shell command -v hipcc 2>/dev/null || echo $(ROCM_PATH)/bin/hipcc)
UNI       := third_party/llama-unicode

ARCH    := --offload-arch=gfx906
CXXFLAGS := -O2 -std=c++17 $(ARCH)
INC     := -I quartz -I $(UNI) -I third_party
ENGINE  := quartz/loader.cpp $(wildcard quartz/core/*.cpp) $(wildcard quartz/vision/*.cpp) $(wildcard quartz/arch/*/*.cpp)
UNICODE := $(UNI)/unicode.cpp $(UNI)/unicode-data.cpp
HEADERS := $(wildcard quartz/*.h quartz/*.hip quartz/core/*.h quartz/kernels/*.h quartz/vision/*.h quartz/arch/*/*.h quartz/arch/*/*.hip) $(UNI)/unicode.h third_party/stb/stb_image.h

.PHONY: all lib clean
all: lib
lib: $(BIN)/libquartz.so

$(BIN):
	mkdir -p $@

# libquartz: decode, sampling, C API, tokenizer, unicode tables, image encoder (rocBLAS)
$(BIN)/libquartz.so: quartz/capi.cpp quartz/tokapi.cpp $(ENGINE) $(HEADERS) | $(BIN)
	$(HIPCC) $(CXXFLAGS) -fPIC -shared $(INC) quartz/capi.cpp quartz/tokapi.cpp $(ENGINE) $(UNICODE) -lrocblas -o $@

clean:
	rm -rf build
