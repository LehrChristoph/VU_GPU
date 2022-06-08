SOURCE_FILES=main.c io.c graph.c generator.c impl/cpu.c impl/gpu_thread_per_cc.cu impl/gpu_thread_per_cc_vector.cu impl/gpu_pinned.cu impl/gpu_vector_pinned.cu
INCLUDE_FILES=generator.h graph.h impl.h io.h cuda_helpers.h

all: connected-components

connected-components: $(addprefix src/,$(SOURCE_FILES)) $(addprefix include/,$(INCLUDE_FILES))
	nvcc -g -G $(addprefix src/,$(SOURCE_FILES)) -I ./include -o ./connected-components

test: connected-components
	./connected-components bench 10 100 1

.PHONY: test
