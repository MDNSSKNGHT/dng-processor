SOURCES := $(wildcard *.slang)
TARGETS := $(SOURCES:.slang=.glsl)

all: $(TARGETS)

%.glsl: %.slang
	slangc $< -profile glsl_460 -target glsl -o $@ -entry main

.PHONY: clean
clean:
	rm -f $(TARGETS)
