#include <iostream>
#include <fstream>
#include <complex>
#include <cuda_runtime.h>

#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1
#define RESOLUTION 1000
#define WIDTH (RESOLUTION * (MAX_X - MIN_X))
#define HEIGHT (RESOLUTION * (MAX_Y - MIN_Y))
#define STEP ((double)(MAX_X - MIN_X) / WIDTH)
#define ITERATIONS 1000

using namespace std;

__device__ int computePixel(double x, double y) {
    complex<double> c(x, y);
    complex<double> z(0, 0);
    for (int i = 0; i < ITERATIONS; ++i) {
        z = z * z + c;
        if (abs(z) >= 2.0) return i;
    }
    return 0;
}

__global__ void mandelbrotKernel(int *image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= WIDTH || idy >= HEIGHT) return;

    double x = MIN_X + idx * STEP;
    double y = MIN_Y + idy * STEP;

    image[idy * WIDTH + idx] = computePixel(x, y);
}

int main(int argc, char **argv) {
    int *image, *d_image;

    image = new int[WIDTH * HEIGHT];
    cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(int));

    dim3 blockSize(512);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    mandelbrotKernel<<<gridSize, blockSize>>>(d_image);

    cudaMemcpy(image, d_image, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    ofstream outFile(argv[1]);
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            outFile << image[i * WIDTH + j] << (j < WIDTH - 1 ? "," : "");
        }
        outFile << endl;
    }
    outFile.close();

    cudaFree(d_image);
    delete[] image;

    return 0;
}