CC = icc
CFLAGS = -diag-disable=10441 -qopenmp 

all: mandelbrot_omp_mk

mandelbrot_omp_mk: main.cpp
    $(CC) $(CFLAGS) -o mandelbrot_omp_mk mandelbrot_makefile.cpp

clean:
    rm -f mandelbrot_omp_mk


run_4000_24:
    @echo "Compiling program with 4000 resolution and number of thread 32..."
    ./mandelbrot_omp_mk 4000 32 output.txt
    @echo "/--------------------------------------------------------------/"

run_2000_32:
    @echo "Compiling program with 2000 resolution and number of thread 32..."
    ./mandelbrot_omp_mk 2000 32 output.txt
    @echo "/--------------------------------------------------------------/"

run_1000_32:
    @echo "Compiling program with 1000 resolution and number of thread 32..."
    ./mandelbrot_omp_mk 1000 32 output.txt
    @echo "/--------------------------------------------------------------/"

run_4000_24:
    @echo "Compiling program with 4000 resolution and number of thread 24..."
    ./mandelbrot_omp_mk 4000 24 output.txt
    @echo "/--------------------------------------------------------------/"

run_2000_24:
    @echo "Compiling program with 2000 resolution and number of thread 24..."
    ./mandelbrot_omp_mk 2000 24 output.txt
    @echo "/--------------------------------------------------------------/"

run_1000_24:
    @echo "Compiling program with 1000 resolution and number of thread 24..."
    ./mandelbrot_omp_mk 1000 24 output.txt
    @echo "/--------------------------------------------------------------/"

run_4000_16:
    @echo "Compiling program with 4000 resolution and number of thread 16..."
    ./mandelbrot_omp_mk 4000 16 output.txt
    @echo "/--------------------------------------------------------------/"

run_2000_16:
    @echo "Compiling program with 2000 resolution and number of thread 16..."
    ./mandelbrot_omp_mk 2000 16 output.txt
    @echo "/--------------------------------------------------------------/"

run_1000_16:
    @echo "Compiling program with 1000 resolution and number of thread 16..."
    ./mandelbrot_omp_mk 1000 16 output.txt
    @echo "/--------------------------------------------------------------/"

run_4000_8:
    @echo "Compiling program with 4000 resolution and number of thread 8..."
    ./mandelbrot_omp_mk 4000 8 output.txt
    @echo "/--------------------------------------------------------------/"

run_2000_8:
    @echo "Compiling program with 2000 resolution and number of thread 8..."
    ./mandelbrot_omp_mk 2000 8 output.txt
    @echo "/--------------------------------------------------------------/"

run_1000_8:
    @echo "Compiling program with 1000 resolution and number of thread 8..."
    ./mandelbrot_omp_mk 1000 8 output.txt
    @echo "/--------------------------------------------------------------/"

run_4000_4:
    @echo "Compiling program with 4000 resolution and number of thread 4..."
    ./mandelbrot_omp_mk 4000 4 output.txt
    @echo "/--------------------------------------------------------------/"

run_2000_4:
    @echo "Compiling program with 2000 resolution and number of thread 4..."
    ./mandelbrot_omp_mk 2000 4 output.txt
    @echo "/--------------------------------------------------------------/"

run_1000_4:
    @echo "Compiling program with 1000 resolution and number of thread 4..."
    ./mandelbrot_omp_mk 1000 4 output_custom_res.txt
    @echo "/--------------------------------------------------------------/"
