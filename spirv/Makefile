SOURCES := $(wildcard *.slang)
TARGETS := $(SOURCES:.slang=.spirv)

all: $(TARGETS)

%.spirv: %.slang
	slangc $< -profile glsl_450 -target spirv -o $@ -entry main

.PHONY: clean
clean:
	rm -f $(TARGETS)
