
obj_dir		 = ./obj
src_dir   	 = ./src

LIB_OBJ_PATH	 = $(obj_dir)/$(LIBNAME)

srcs		 = $(wildcard $(src_dir)/*.cu)
srcs	        += $(wildcard $(src_dir)/*.cpp) 
incs 		 = $(wildcard $(src_dir)/*.h)

local_objects_tmp	= $(srcs:$(src_dir)%.cu=$(obj_dir)%.o)
local_objects		= $(local_objects_tmp:$(src_dir)%.cpp=$(obj_dir)%.o)

CXX := nvcc

CXXFLAGS := -W -Wall -Wextra -ansi -pedantic -O3 

all: hist

%.o: %.cpp
	@mkdir -p `dirname $@`
	$(CXX) -I $(src_dir) $(CXXFLAGS) -c $< -o $@
%.o: %.cu
	$(CXX) -I $(src_dir) $(CXXFLAGS) -c $< -o $@ 	

hist : $(local_objects)
	$(CXX) -o $(local_objects) -o hist

clean:
	rm -f unittest benchmark pngdetail showpng lodepng_unittest.o lodepng_benchmark.o lodepng.o lodepng_util.o pngdetail.o examples/example_sdl.o
