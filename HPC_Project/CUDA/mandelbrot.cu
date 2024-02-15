#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <cmath>
//#include <cuda.h>
#include "cuda_runtime.h"

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;



__global__ void mandelbrotGPUfunction(int *image, double step, double minX, double minY, int width, int height, int iterations)
{

    //int pos = blockIdx.x * blockDim.x + threadIdx.x;
    // int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = row * WIDTH + col;

    if (pos < width * height) {
        image[pos] = 0;

        //const int row = pos / width;
        //const int col = pos % width;

        //const int row = (pos / width) + blockIdx.y * blockDim.y + threadIdx.y;
        //const int col = (pos % width) + blockIdx.x * blockDim.x + threadIdx.x;

        //const int row = blockIdx.y * blockDim.y + threadIdx.y;
        //const int col = blockIdx.x * blockDim.x + threadIdx.x;
        //int pos = row * WIDTH + col;

        const complex<double> c(col * step + minX, row * step + minY);

        complex<double> z(0, 0);

        for (int i = 1; i <= iterations; i++){
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                image[pos] = i;
                break;
            }
            if(i == iterations){
                image[pos]= 0;
            }
        }
    }
}



int main(int argc, char **argv)
{
    int *const image = new int[HEIGHT * WIDTH];

    // Timer
    cudaEvent_t  start, stop;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    // Allocation
    int *d_image;
    cudaMalloc(&d_image, sizeof(int) * WIDTH * HEIGHT);
    // cudaMemcpy(d_image, image, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    // Threads
    // dim3 block(512); // threads per block 
    dim3 block(16, 16);
    //dim3 grid((WIDTH * HEIGHT + block.x - 1) / block.x); // num blocks 
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    // Start timer
    cudaEventRecord(start);

    mandelbrotGPUfunction<<<grid, block>>>(d_image, STEP, MIN_X, MIN_Y, WIDTH, HEIGHT, ITERATIONS);
    cudaDeviceSynchronize();
    cudaMemcpy(image, d_image, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time = 0.0;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cout << "Time elapsed: " <<  elapsed_time << " milliseconds." << endl;

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    cudaFree(d_image);
    return 0;
}


/*  ORIGINAL cpp 
int main(int argc, char **argv)
{  

    int *const image = new int[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();
    for (int pos = 0; pos < HEIGHT * WIDTH; pos++)
    {
        image[pos] = 0;

        const int row = pos / WIDTH;
        const int col = pos % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                image[pos] = i;
                break;
            }
        }
    }

    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::seconds>(end - start).count()
         << " seconds." << endl;

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    return 0;
}
*/
