FILENAME	 = hist

obj_dir		 = ./obj
src_dir   	 = ./src
inc_dir		 = ./inc

LIB_OBJ_PATH	 = $(obj_dir)/$(LIBNAME)

srcs		 = $(wildcard $(src_dir)/*.cu)
srcs	    += $(wildcard $(src_dir)/*.cpp) 
incs 		 = $(wildcard $(inc_dir)/*.h)

local_objects_tmp	= $(srcs:$(src_dir)%.cu=$(obj_dir)%.o)
local_objects		= $(local_objects_tmp:$(src_dir)%.cpp=$(obj_dir)%.o)

CXX 	= /usr/local/cuda-11.6/bin/nvcc
NVCXX 	= /usr/local/cuda-11.6/bin/nvcc

NVCXXFLAGS	= 
CXXFLAGS	= -ccbin g++ -Xcompiler "-W -Wall -Wextra -ansi -pedantic -O3 -std=c++11 " -Lcuda -Lcudart
LINKERFLAGS 	= in g++ -Xcompiler -lcuda -lcudart

all: $(obj_dir) hist

$(obj_dir)/%.o: $(src_dir)/%.cpp
	$(CXX) -I $(inc_dir) $(CXXFLGAS) -c $< -o $@
$(obj_dir)/%.o: $(src_dir)/%.cu
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

hist : $(local_objects)
	$(CXX) $(CXXFLAGS) $(local_objects) -o $(FILENAME)

$(obj_dir):
	mkdir -p $(obj_dir)

clean:
	rm -rf $(obj_dir) $(FILENAME)
