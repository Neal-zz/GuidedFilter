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

// test function.
void darkChannel_cpu(std::vector<unsigned char> pixels, std::vector<unsigned char> darkImage, unsigned width, unsigned height) {
	int loss = 0;
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int value = 255;
			for (int dx = -windowR; dx <= windowR; dx++) {
				for (int dy = -windowR; dy <= windowR; dy++) {
					for (int ch = 0; ch < 3; ch++) {
						int tempx = x + dx;
						int tempy = y + dy;
						tempx = tempx > 0 ? tempx : 0;
						tempy = tempy > 0 ? tempy : 0;
						tempx = tempx < (width - 1) ? tempx : (width - 1);
						tempy = tempy < (height - 1) ? tempy : (height - 1);

						int tempv = static_cast<int>(pixels[tempy * width * 4 + tempx * 4 + ch]);
						if (tempv < value) {
							value = tempv;
						}
					}
				}
			}
			loss += abs(value - static_cast<int>(darkImage[y * width + x]));
		}
	}
	std::cout << "loss: " << loss << std::endl;

	return;
}

// test function.
void getHist_cpu(std::vector<unsigned char> darkImage, std::vector<unsigned> hist, unsigned width, unsigned height) {
	std::vector<unsigned> local_hist(nbins);
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			local_hist[darkImage[y * width + x]]++;
		}
	}

	int loss = 0;
	for (int bin = 0; bin < nbins; bin++) {
		loss += abs(static_cast<int>(local_hist[bin]) - static_cast<int>(hist[bin]));
	}
	std::cout << "loss: " << loss << std::endl;

	return;
}

// test funciton.
void getAc_cpu(std::vector<unsigned char> pixels, std::vector<unsigned char> Idark, std::vector<unsigned char> Ac, unsigned char colorThresh, unsigned width, unsigned height) {
	std::vector<unsigned char> local_Ac(3);
	local_Ac[0] = 0; local_Ac[1] = 0; local_Ac[2] = 0;
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			if (Idark[y * width + x] >= colorThresh && pixels[y * width * 4 + x * 4] > local_Ac[0]) {
				local_Ac[0] = pixels[y * width * 4 + x * 4];
			}
			if (Idark[y * width + x] >= colorThresh && pixels[y * width * 4 + x * 4 + 1] > local_Ac[1]) {
				local_Ac[1] = pixels[y * width * 4 + x * 4 + 1];
			}
			if (Idark[y * width + x] >= colorThresh && pixels[y * width * 4 + x * 4 + 2] > local_Ac[2]) {
				local_Ac[2] = pixels[y * width * 4 + x * 4 + 2];
			}
		}
	}

	int loss = 0;
	for (int bin = 0; bin < 3; bin++) {
		loss += abs(static_cast<int>(local_Ac[bin]) - static_cast<int>(Ac[bin]));
	}
	std::cout << "loss: " << loss << std::endl;

	return;
}

// test function.
void getttilde_cpu(std::vector<unsigned char> pixels, std::vector<float> ttilde_test, std::vector<unsigned char> Ac_test, unsigned width, unsigned height) {
	float loss = 0.0;
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			float value = 5.0;
			for (int dx = -windowR; dx <= windowR; dx++) {
				for (int dy = -windowR; dy <= windowR; dy++) {
					for (int ch = 0; ch < 3; ch++) {
						int tempx = x + dx;
						int tempy = y + dy;
						tempx = tempx > 0 ? tempx : 0;
						tempy = tempy > 0 ? tempy : 0;
						tempx = tempx < (width - 1) ? tempx : (width - 1);
						tempy = tempy < (height - 1) ? tempy : (height - 1);

						float tempv = static_cast<float>(pixels[tempy * width * 4 + tempx * 4 + ch]) / static_cast<float>(Ac_test[ch]);
						if (tempv < value) {
							value = tempv;
						}
					}
				}
			}
			float temp2 = 1.0 - static_cast<float>(ttilde_w) * value;
			loss += fabs(temp2 - ttilde_test[y * width + x]);
		}
	}
	std::cout << "loss: " << loss << std::endl;

	return;
}

// test function.
void gray_cpu(std::vector<unsigned char> pixels, std::vector<unsigned char> gray_test, unsigned char width, unsigned char height) {
	int loss = 0;
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			float temp = 0.299 * static_cast<float>(pixels[y * width * 4 + x * 4]) + 0.587 * static_cast<float>(pixels[y * width * 4 + x * 4 + 1]) + 0.114 * static_cast<float>(pixels[y * width * 4 + x * 4 + 2]);
			loss += abs(static_cast<int>(gray_test[y * width + x]) - static_cast<int>(temp));
		}
	}
	std::cout << "loss: " << loss << std::endl;

	return;
}

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

std::vector<unsigned char> showResult(const std::vector<unsigned char>& in,
	const unsigned width, const unsigned height) {
	std::vector<unsigned char> result(width * height * 4);

	for (int i = 0; i < width * height * 4; i += 4) {
		result[i] = static_cast<unsigned char>(in[i/4*3]);
		result[i + 1] = static_cast<unsigned char>(in[i / 4 * 3 + 1]);
		result[i + 2] = static_cast<unsigned char>(in[i / 4 * 3 + 2]);
		result[i + 3] = 255;
	}

	return result;
}

int main()
{
	Timer timer;

	DisplayHeader();

	// read image
	std::vector<unsigned char> pixels;  // 1 byte: 0-255
	unsigned width, height;
	std::cout << "Reading Fog Image...";
	pixels = loadImage("fog0.png", width, height);
	unsigned imSize = width * height;

	// block and thread allocation.
	dim3 blocks((width + tileWidth - 1) / tileWidth, (height + tileHeight - 1) / tileHeight);
	dim3 threads_withAprone(aproneWidth, aproneHeight);  // waste threads.
	dim3 threads_noAprone(tileWidth, tileHeight);

	dim3 threads_subHist(512);
	dim3 blocks_subHist((width + threads_subHist.x - 1) / threads_subHist.x);
	//unsigned nsubHist = blocks.x * blocks.y;  // for subHist.
	unsigned nsubHist = blocks_subHist.x;  // for subHist_2.
	unsigned subHistSize = nsubHist * nbins;

	dim3 threads_sumHist(nbins);
	dim3 blocks_sumHist(1);

	dim3 threads_getAc(3);
	dim3 blocks_getAc(1);

	std::vector<unsigned> h_hist(nbins);

	// Device variabels
	unsigned char* d_orig, * d_Idark, * d_AcRow, * d_Ac, * d_guidedI, * d_outputQ;
	unsigned* d_subHist, *d_hist;
	float* d_ttilde, * d_ttilde2, * d_ab;

	CudaCall(cudaMalloc((void**)&d_orig, sizeof(unsigned char) * imSize * 4));  // 4: rgbd.
	CudaCall(cudaMalloc((void**)&d_Idark, sizeof(unsigned char) * imSize));
	CudaCall(cudaMalloc((void**)&d_AcRow, sizeof(unsigned char) * width * 3));
	CudaCall(cudaMalloc((void**)&d_Ac, sizeof(unsigned char) * 3));  // 3: rgb.
	CudaCall(cudaMalloc((void**)&d_guidedI, sizeof(unsigned char) * imSize));
	CudaCall(cudaMalloc((void**)&d_outputQ, sizeof(unsigned char) * imSize * 3));  // 3: rgb.
	CudaCall(cudaMalloc((void**)&d_subHist, sizeof(unsigned) * subHistSize));
	CudaCall(cudaMalloc((void**)&d_hist, sizeof(unsigned) * nbins));
	CudaCall(cudaMalloc((void**)&d_ttilde, sizeof(float) * imSize));
	CudaCall(cudaMalloc((void**)&d_ttilde2, sizeof(float) * imSize));
	CudaCall(cudaMalloc((void**)&d_ab, sizeof(float) * imSize * 2));  // 2: ab.

	// Copy Data from host to device
	CudaCall(cudaMemcpy(d_orig, pixels.data(), sizeof(pixels[0]) * pixels.size(), cudaMemcpyHostToDevice));
	// Profiling
	float elapsed = 0;
	cudaEvent_t start, stop;

	CudaCall(cudaEventCreate(&start));
	CudaCall(cudaEventCreate(&stop));
 
	// I dark.
	std::cout << "Get the Dark Image of I...";
	CudaCall(cudaEventRecord(start));

	darkChannel<<<blocks, threads_noAprone>>>(d_orig, d_Idark, width, height);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// test.
	//std::vector<unsigned char> darkImage_test(imSize);
	//CudaCall(cudaMemcpy(&darkImage_test[0], d_Idark, sizeof(darkImage_test[0]) * imSize, cudaMemcpyDeviceToHost));
	//darkChannel_cpu(pixels, darkImage_test, width, height);

	// subhistogram.
	std::cout << "Calculate the Subhistogram...";
	CudaCall(cudaEventRecord(start));

	//subHist<<<blocks, threads_noAprone>>>(d_Idark, d_subHist, width, height);  // slower.
	subHist_2<<<blocks_subHist, threads_subHist>>>(d_Idark, d_subHist, width, height);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// sum subhistograms.
	std::cout << "Sum Subhistograms...";
	CudaCall(cudaEventRecord(start));

	sumHist<<<blocks_sumHist, threads_sumHist>>>(d_subHist, d_hist, subHistSize);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// test.
	//std::vector<unsigned char> darkImage_test(imSize);
	//std::vector<unsigned> hist_test(nbins);
	//CudaCall(cudaMemcpy(&darkImage_test[0], d_Idark, sizeof(darkImage_test[0])* imSize, cudaMemcpyDeviceToHost));
	//CudaCall(cudaMemcpy(&hist_test[0], d_hist, sizeof(hist_test[0])* nbins, cudaMemcpyDeviceToHost));
	//getHist_cpu(darkImage_test, hist_test, width, height);

	// get AcRow.
	std::cout << "Calculate AcRow...";
	CudaCall(cudaEventRecord(start));

	unsigned char colorThresh = 0;
	unsigned totalNum = 0;
	CudaCall(cudaMemcpy(&h_hist[0], d_hist, sizeof(h_hist[0]) * nbins, cudaMemcpyDeviceToHost));
	for (int bin = (nbins - 1); bin >= 0; bin--) {
		totalNum += h_hist[bin];
		if (totalNum >= (imSize * 0.001)) {
			colorThresh = static_cast<unsigned char>(bin);
			break;
		}
	}
	getAcRow<<<blocks_subHist, threads_subHist>>>(d_orig, d_Idark, d_AcRow, colorThresh, width, height);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// get Ac.
	std::cout << "Calculate Ac...";
	CudaCall(cudaEventRecord(start));

	getAc<<<blocks_getAc, threads_getAc>>>(d_AcRow, d_Ac, width);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// test.
	//std::vector<unsigned char> Idark_test(imSize);
	//std::vector<unsigned char> Ac_test(3);
	//CudaCall(cudaMemcpy(&Idark_test[0], d_Idark, sizeof(Idark_test[0]) * imSize, cudaMemcpyDeviceToHost));
	//CudaCall(cudaMemcpy(&Ac_test[0], d_Ac, sizeof(Ac_test[0]) * 3, cudaMemcpyDeviceToHost));
	//getAc_cpu(pixels, Idark_test, Ac_test, colorThresh, width, height);

	// get t_tilde.
	std::cout << "Get the t_tilde(x)...";
	CudaCall(cudaEventRecord(start));

	getttilde<<<blocks, threads_noAprone>>>(d_orig, d_ttilde, d_Ac, width, height, ttilde_w);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// test.
	//std::vector<float> ttilde_test(imSize);
	//std::vector<unsigned char> Ac_test(3);
	//CudaCall(cudaMemcpy(&ttilde_test[0], d_ttilde, sizeof(ttilde_test[0]) * imSize, cudaMemcpyDeviceToHost));
	//CudaCall(cudaMemcpy(&Ac_test[0], d_Ac, sizeof(Ac_test[0]) * 3, cudaMemcpyDeviceToHost));
	//getttilde_cpu(pixels, ttilde_test, Ac_test, width, height);

	// Scale and gray guided I.
	std::cout << "Converting Guided I to Grayscale...";
	CudaCall(cudaEventRecord(start));

	gray<<<height, width>>>(d_orig, d_guidedI, width, height);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// test.
	//std::vector<unsigned char> gray_test(imSize);
	//CudaCall(cudaMemcpy(&gray_test[0], d_guidedI, sizeof(gray_test[0]) * imSize, cudaMemcpyDeviceToHost));
	//gray_cpu(pixels, gray_test, width, height);

	// calculate ak and bk for each window wk.
	std::cout << "Calculating ak and bk...";
	CudaCall(cudaEventRecord(start));

	linearPara<<<blocks, threads_noAprone>>>(d_ttilde, d_guidedI, d_ab, width, height, guided_epsilon);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// image filtering
	std::cout << "Image Filtering...";
	CudaCall(cudaEventRecord(start));

	doFiltering_new<<<blocks, threads_noAprone>>>(d_ab, d_guidedI, d_ttilde2, width, height);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// calculate J
	std::cout << "Calculating J...";
	CudaCall(cudaEventRecord(start));

	get_J<<<blocks, threads_noAprone>>>(d_orig, d_ttilde2, d_outputQ, d_Ac, width, height, J_t0);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Copy data from device to host
	std::vector<unsigned char> output(imSize * 3);
	CudaCall(cudaMemcpy(&output[0], d_outputQ, sizeof(output[0]) * imSize * 3, cudaMemcpyDeviceToHost));
	
	lodepng::encode("output.png", showResult(output, width, height), width, height);  // draw right image on the left image.
	
	//unsigned num = 0;
	//for (int i = 0; i < nbins; i++) {
	//	num += h_hist[i];
	//}
	//std::cout << num << std::endl;

	//std::cout << static_cast<int>(output[0]) << " " << static_cast<int>(output[1]) << " " << static_cast<int>(output[2]) << " " << std::endl;

	//std::cout << "The program took " << timer.getElapsedTime() << " s" << std::endl;

	cudaFree(d_orig);
	cudaFree(d_Idark);
	cudaFree(d_AcRow);
	cudaFree(d_Ac);
	cudaFree(d_ttilde);
	cudaFree(d_ttilde2);
	cudaFree(d_guidedI);
	cudaFree(d_subHist);
	cudaFree(d_hist);
	cudaFree(d_ab);
	cudaFree(d_outputQ);

	std::cin.get();

    return 0;
}


