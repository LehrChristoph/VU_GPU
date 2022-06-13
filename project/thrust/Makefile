SOURCE_FILES=main.cpp graph.cpp impl/cpu.cpp impl/gpu.cu
INCLUDE_FILES=graph.h impl.h cuda_helpers.h

all: connected-components

connected-components: $(addprefix src/,$(SOURCE_FILES)) $(addprefix include/,$(INCLUDE_FILES))
	nvcc -arch=sm_61 -rdc=true -g -G $(addprefix src/,$(SOURCE_FILES)) -I ./include -o ./connected-components --std=c++11

test: connected-components
		./connected-components bench 10 10000 1

evaluate.csv: connected-components $(wildcard ../evaluate/*) ../evaluate/
	./connected-components evaluate ../evaluate/ > evaluate.csv

clean:
	rm -f connected-components evaluate.csv

.PHONY: test

