SOURCE_FILES=main.c graph.c impl/cpu.c impl/gpu.cu
INCLUDE_FILES=graph.h impl.h cuda_helpers.h

all: connected-components

connected-components: $(addprefix src/,$(SOURCE_FILES)) $(addprefix include/,$(INCLUDE_FILES))
	nvcc -g -G $(addprefix src/,$(SOURCE_FILES)) -I ./include -o ./connected-components

test: connected-components
	./connected-components bench 10 100 1

.PHONY: test
