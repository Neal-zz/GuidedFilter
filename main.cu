#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "guidedFilter.cuh"
#include "config.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>



/*
Class to calculate time taken by functions in seconds.
* Creating an object of the class in a function, calls the constructor which starts the timer.
* At the end of the function, the destructor is called which stops the timer and calculates the duration.
* We can get the duration manually using the getElapsedTime method.
*/
class Timer {
private:
	std::chrono::time_point<std::chrono::steady_clock> m_Start, m_End;
	std::chrono::duration<float> m_Duration;

public:
	Timer() {
		m_Start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		std::cout << "Done (" << m_Duration.count() << " s)" << std::endl;
	}

	float getElapsedTime() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		return m_Duration.count();
	}
};


// Display GPU info
// https://stackoverflow.com/a/5689133
void DisplayHeader() {
	const int kb = 1024;
	const int mb = kb * kb;
	std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

	std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	std::cout << "CUDA Devices: " << std::endl << std::endl;

	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
		std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
		std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
		std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
		std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

		std::cout << "  Warp size:         " << props.warpSize << std::endl;
		std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
		std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
		std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << std::endl;
		std::cout << std::endl;
	}
}

std::vector<unsigned char> loadImage(const char* filename, unsigned& width, unsigned& height) {
	Timer timer;

	std::vector<unsigned char> pixels;

	unsigned error = lodepng::decode(pixels, width, height, filename);
	if (error) {
		std::cout << "Failed to load image: " << lodepng_error_text(error) << std::endl;
		std::cin.get();
		exit(-1);
	}

	return pixels;
}

void CudaCall(const cudaError_t& status) {
	if (status != cudaSuccess) {
		std::cout << "Error [" << status << "]: " << cudaGetErrorString(status) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
	}
}

std::vector<unsigned char> showResult(const std::vector<float>& in,
	const unsigned width, const unsigned height) {
	std::vector<unsigned char> result(width * height * 4);

	for (int i = 0; i < width * height * 4; i += 4) {
		float temp = in[i / 4];

		result[i] = result[i + 1] = result[i + 2] = static_cast<unsigned char>(temp);
		result[i + 3] = 255;
	}

	return result;
}

constexpr int scaleFactor = 1;
constexpr float epsilon = 10 * 10;

int main()
{
	Timer timer;

	DisplayHeader();

	// Host variables
	std::vector<unsigned char> pixels;  // 1 byte: 0-255
	unsigned width, height;

	std::cout << "Reading Filtering Input...";
	pixels = loadImage("filteringInput.png", width, height);

	unsigned imSize = width * height;
	std::vector<float> output(imSize);

	// Device variabels
	unsigned char* d_orig, * d_filteringP, * d_guidedI;
	float* d_ab, * d_outputQ;

	CudaCall(cudaMalloc((void**)&d_orig, sizeof(unsigned char) * imSize * 4));  // 4: rgbd.
	CudaCall(cudaMalloc((void**)&d_filteringP, sizeof(unsigned char) * imSize));
	CudaCall(cudaMalloc((void**)&d_guidedI, sizeof(unsigned char) * imSize));
	CudaCall(cudaMalloc((void**)&d_ab, sizeof(float) * imSize * 2));  // 2: ab.
	CudaCall(cudaMalloc((void**)&d_outputQ, sizeof(float) * imSize));

	// Copy Data from host to device
	CudaCall(cudaMemcpy(d_orig, pixels.data(), sizeof(pixels[0]) * pixels.size(), cudaMemcpyHostToDevice));

	// Profiling
	float elapsed = 0;
	cudaEvent_t start, stop;

	CudaCall(cudaEventCreate(&start));
	CudaCall(cudaEventCreate(&stop));

	// Scale and gray filtering input.
	std::cout << "Converting Filtering Input to Grayscale...";
	CudaCall(cudaEventRecord(start));

	ScaleAndGray<<<height, width>>>(d_orig, d_filteringP, width, height, scaleFactor);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Scale and gray guided I.
	std::cout << "Converting Guided I to Grayscale...";
	CudaCall(cudaEventRecord(start));

	ScaleAndGray<<<height, width>>>(d_orig, d_guidedI, width, height, scaleFactor);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// block and thread allocation.
	dim3 blocks((width + tileWidth - 1) / tileWidth, (height + tileHeight - 1) / tileHeight);
	dim3 threads(aproneWidth, aproneHeight);

	// calculate ak and bk for each window wk.
	std::cout << "Calculating ak and bk...";
	CudaCall(cudaEventRecord(start));

	linearPara<<<blocks, threads>>>(d_filteringP, d_guidedI, d_ab, width, height, epsilon);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	//// image filtering
	std::cout << "Image Filtering...";
	CudaCall(cudaEventRecord(start));

	doFiltering_new<<<blocks, threads >>>(d_ab, d_guidedI, d_outputQ, width, height);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Copy data from device to host
	CudaCall(cudaMemcpy(&output[0], d_outputQ, sizeof(output[0]) * imSize, cudaMemcpyDeviceToHost));

	lodepng::encode("output.png", showResult(output, width, height), width, height);  // draw right image on the left image.

	std::cout << "The program took " << timer.getElapsedTime() << " s" << std::endl;

	cudaFree(d_orig);
	cudaFree(d_filteringP);
	cudaFree(d_guidedI);
	cudaFree(d_outputQ);
	cudaFree(d_ab);

	std::cin.get();

    return 0;
}


