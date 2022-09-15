#include "guidedFilter.cuh"

__global__ void ScaleAndGray(unsigned char* orig, unsigned* gray, unsigned width, unsigned height, int scaleFactor) {
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (i >= height || j >= width)
		return;

	int newWidth = width / scaleFactor;

	int x = (scaleFactor * i - 1 * (i > 0));
	int y = (scaleFactor * j - 1 * (j > 0));

	gray[i * newWidth + j] =
		0.3 * orig[x * (4 * width) + 4 * y] +
		0.59 * orig[x * (4 * width) + 4 * y + 1] +
		0.11 * orig[x * (4 * width) + 4 * y + 2];
	
	return;
}

__global__ void linearPara(unsigned* filteringP, unsigned* guidedI, float* ab, int width, int height, int windowWidth, int windowHeight, float epsilon) {
	
	float w_num = 0.0, miu = 0.0, pk = 0.0, Ip = 0.0, miu2 = 0.0;

	int i = blockIdx.x;  // height
	int j = threadIdx.x;  // width

	if (i >= height || j >= width)
		return;

	// (use only other pixel value.)
	for (int x = -(windowHeight - 1) / 2; x <= (windowHeight - 1) / 2; x++) {
		for (int y = -(windowWidth - 1) / 2; y <= (windowWidth - 1) / 2; y++) {
			// Check for image borders
			if (
				!(i + x >= 0) ||
				!(i + x < height) ||
				!(j + y >= 0) ||
				!(j + y < width)
				) {
				continue;
			}
			float temp1 = static_cast<float>(guidedI[(i + x) * width + (j + y)]);
			float temp2 = static_cast<float>(filteringP[(i + x) * width + (j + y)]);
			w_num += 1.0;
			miu += temp1;
			pk += temp2;
			Ip += temp1 * temp2;
			miu2 += temp1 * temp1;
		}
	}
	float ak = (Ip / w_num - (miu / w_num) * (pk / w_num)) / (miu2 / w_num - (miu / w_num) * (miu / w_num) + epsilon);
	float bk = (pk / w_num) - ak * (miu / w_num);
	ab[i * width * 2 + j * 2] = ak;
	ab[i * width * 2 + j * 2 + 1] = bk;

	return;
}

__global__ void doFiltering(float* ab, unsigned* guidedI, float* outputQ, int width, int height, int windowWidth, int windowHeight) {

	int i = blockIdx.x;  // height
	int j = threadIdx.x;  // width

	if (i >= height || j >= width)
		return;

	/*gIs_size needs to be adjusted, every time image size changes.*/
	const unsigned gIs_size = 960 * 9;  // width*windowHeight
	__shared__ unsigned gIs[gIs_size];
	int sm_index = 0;
	for (int sm_i = i - (windowHeight - 1) / 2; sm_i <= i + (windowHeight - 1) / 2; sm_i++) {
		if ((sm_i < 0) || (sm_i >= height)) {
			gIs[sm_index * width + j] = 0;
		}
		else {
			gIs[sm_index * width + j] = guidedI[sm_i * width + j];
		}
		sm_index++;
	}
	__syncthreads();

	/* start filtering.*/
	float ak = ab[i * width * 2 + j * 2];
	float bk = ab[i * width * 2 + j * 2 + 1];
	sm_index = 0;
	// (use only other pixel value.)
	for (int x = -(windowHeight - 1) / 2; x <= (windowHeight - 1) / 2; x++) {
		for (int y = -(windowWidth - 1) / 2; y <= (windowWidth - 1) / 2; y++) {
			// Check for image borders
			if (
				!(i + x >= 0) ||
				!(i + x < height) ||
				!(j + y >= 0) ||
				!(j + y < width)
				) {
				continue;
			}

			outputQ[(i + x) * width + (j + y)] += (ak * static_cast<float>(gIs[sm_index * width + (j + y)]) + bk);
			__syncthreads();
		}
		sm_index++;
	}

	return;
}

__global__ void doFiltering_new(float* ab, unsigned* guidedI, float* outputQ, int width, int height, int windowWidth, int windowHeight) {

	int i = blockIdx.x;  // height
	int j = threadIdx.x;  // width

	if (i >= height || j >= width)
		return;

	/* start filtering.*/
	float Iij = static_cast<float>(guidedI[i * width + j]);
	float result = 0.0;
	// (use only other pixel value.)
	for (int x = -(windowHeight - 1) / 2; x <= (windowHeight - 1) / 2; x++) {
		for (int y = -(windowWidth - 1) / 2; y <= (windowWidth - 1) / 2; y++) {
			// Check for image borders
			if (
				!(i + x >= 0) ||
				!(i + x < height) ||
				!(j + y >= 0) ||
				!(j + y < width)
				) {
				continue;
			}
			result += ab[(i + x) * width * 2 + (j + y) * 2] * Iij + ab[(i + x) * width * 2 + (j + y) * 2 + 1];
		}
	}

	// normalize.
	int width_max = (width-1) < (j + (windowWidth - 1) / 2) ? (width-1) : (j + (windowWidth - 1) / 2);
	int width_min = 0 > (j - (windowWidth - 1) / 2) ? 0 : (j - (windowWidth - 1) / 2);
	int height_max = (height-1) < (i + (windowHeight - 1) / 2) ? (height-1) : (i + (windowHeight - 1) / 2);
	int height_min = 0 > (i - (windowHeight - 1) / 2) ? 0 : (i - (windowHeight - 1) / 2);
	int w_num = (width_max - width_min + 1) * (height_max - height_min + 1);
	outputQ[i * width + j] = result / static_cast<float>(w_num);
	return;
}

__global__ void imNormalizing(float* outputQ, int width, int height, int windowWidth, int windowHeight) {
	
	int i = blockIdx.x;  // height
	int j = threadIdx.x;  // width

	if (i >= height || j >= width)
		return;

	// normalize.
	int width_max = width < (j + (windowWidth - 1) / 2) ? width : (j + (windowWidth - 1) / 2);
	int width_min = 0 > (j - (windowWidth - 1) / 2) ? 0 : (j - (windowWidth - 1) / 2);
	int height_max = height < (i + (windowHeight - 1) / 2) ? height : (i + (windowHeight - 1) / 2);
	int height_min = 0 > (i - (windowHeight - 1) / 2) ? 0 : (i - (windowHeight - 1) / 2);
	int w_num = (width_max - width_min + 1) * (height_max - height_min + 1);
	outputQ[i * width + j] = outputQ[i * width + j] / static_cast<float>(w_num);
}