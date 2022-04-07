# Exercise 1
Group 3 / Christoph Lehr, Jerome Hue, Lukas Rysavy, Manuel Reinsperger 

## Building / Executing

The application can be built using `make` and executed with `./hist
<implementation (0-2)> <repetitions> <image>`. The first parameter specifies
the implementation to use (see below), the second one how often to execute the
algorithm (to calculate total and average execution times), and the last one
the path to an image file (png).

## Algorithm

The algorithm creates a histogram per channel (RGBA) and outputs it to
stdout. The reason why we decided against having a histogram over channels,
ie. for 'real colors', is on one hand the size of the output array (over 4
billion entries for 4 channels), and on the other hand the fact that this way,
collisions are more likely (since only one channel has to match between two
threads, instead of all 4), which makes for a worse scenario and therefore
more room for improvements.

## Implementations

We implemented three versions of the algorithm to be able to compare their
performance.

### 0: CPU

The first version is a basic loop over the image, running entirely on the
CPU. This can act as a basis for all GPU algorithms to be compared to.

### 1: Naive GPU

This is the naive approach similar to the one presented in the lecture: Each
thread is assigned one pixel, and the histogram is updated using atomic add
operations. This is between 25 and 35 times faster than the CPU version.

### 2: Better GPU

Here, a histogram per thread block into a local array of buckets is
performed. These buckets are then added together atomically to one global
array. This is about 10 times faster than the naive approach with only global
buckets.

### Further improvements

Another idea we had was to take properties of images into account: We assumed
that, especially for pictures not created artificially, for example pictures
taken by a camera, similar (equivalent) colors might occur more frequently in
adjacent pixels. Therefore, we tried spreading out the pixels handled by
threads belonging to the same block over the whole image. However, in our
tests this did not lead to any noticable performance improvements.
