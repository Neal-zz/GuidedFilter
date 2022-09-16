#include "guidedFilter.cuh"

__global__ void ScaleAndGray(unsigned char* orig, unsigned char* gray, unsigned width, unsigned height, int scaleFactor) {
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (i >= height || j >= width)
		return;

	int newWidth = width / scaleFactor;

	int x = (scaleFactor * i - 1 * (i > 0));
	int y = (scaleFactor * j - 1 * (j > 0));

	float temp = 0.3 * static_cast<float>(orig[x * (4 * width) + 4 * y]) +
		0.59 * static_cast<float>(orig[x * (4 * width) + 4 * y + 1]) +
		0.11 * static_cast<float>(orig[x * (4 * width) + 4 * y + 2]);
	gray[i * newWidth + j] = static_cast<unsigned char>(temp);
		
	return;
}

__global__ void linearPara(unsigned char* filteringP, unsigned char* guidedI, float* ab, int width, int height, float epsilon) {
	
	// shared memory.
	__shared__ unsigned char fPsm[aproneWidth * aproneHeight];
	__shared__ unsigned char gIsm[aproneWidth * aproneHeight];

	// image coordinates in fP and gI.
	int x = blockIdx.x * tileWidth + threadIdx.x - windowR;  // width.
	int y = blockIdx.y * tileHeight + threadIdx.y - windowR;  // height.
	x = x > 0 ? x : 0;
	x = x < (width - 1) ? x : (width - 1);
	y = y > 0 ? y : 0;
	y = y < (height - 1) ? y : (height - 1);
	
	// index.
	unsigned index = y * width + x;  // image index.
	unsigned smindex = threadIdx.y * blockDim.x + threadIdx.x;  // shared memory index.

	// data copy.
	fPsm[smindex] = filteringP[index];
	gIsm[smindex] = guidedI[index];
	__syncthreads();

	// aprone area checking
	if (threadIdx.x < windowR || threadIdx.x >= (aproneWidth - windowR) ||
		threadIdx.y < windowR || threadIdx.y >= (aproneHeight - windowR) ||
		(blockIdx.x * tileWidth + threadIdx.x - windowR) >= width ||
		(blockIdx.y * tileHeight + threadIdx.y - windowR) >= height)
		return;

	// (use only other pixel value.)
	float miu = 0.0, pk = 0.0, Ip = 0.0, miu2 = 0.0;
	for (int dy = -windowR; dy <= windowR; dy++) {
		for (int dx = -windowR; dx <= windowR; dx++) {
			float temp1 = static_cast<float>(gIsm[smindex + (dy * blockDim.x) + dx]);
			float temp2 = static_cast<float>(fPsm[smindex + (dy * blockDim.x) + dx]);
			miu += temp1;
			pk += temp2;
			Ip += temp1 * temp2;
			miu2 += temp1 * temp1;
		}
	}
	float num = static_cast<float>(windowSize);
	float ak = (Ip / num - (miu / num) * (pk / num)) / (miu2 / num - (miu / num) * (miu / num) + epsilon);
	float bk = (pk / num) - ak * (miu / num);
	ab[index * 2] = ak;
	ab[index * 2 + 1] = bk;

	return;
}

//__global__ void doFiltering(float* ab, unsigned* guidedI, float* outputQ, int width, int height) {
//
//	int i = blockIdx.x;  // height
//	int j = threadIdx.x;  // width
//
//	if (i >= height || j >= width)
//		return;
//
//	/*gIs_size needs to be adjusted, every time image size changes.*/
//	const unsigned gIs_size = 960 * 9;  // width*windowHeight
//	__shared__ unsigned gIs[gIs_size];
//	int sm_index = 0;
//	for (int sm_i = i - (windowHeight - 1) / 2; sm_i <= i + (windowHeight - 1) / 2; sm_i++) {
//		if ((sm_i < 0) || (sm_i >= height)) {
//			gIs[sm_index * width + j] = 0;
//		}
//		else {
//			gIs[sm_index * width + j] = guidedI[sm_i * width + j];
//		}
//		sm_index++;
//	}
//	__syncthreads();
//
//	/* start filtering.*/
//	float ak = ab[i * width * 2 + j * 2];
//	float bk = ab[i * width * 2 + j * 2 + 1];
//	sm_index = 0;
//	// (use only other pixel value.)
//	for (int x = -(windowHeight - 1) / 2; x <= (windowHeight - 1) / 2; x++) {
//		for (int y = -(windowWidth - 1) / 2; y <= (windowWidth - 1) / 2; y++) {
//			// Check for image borders
//			if (
//				!(i + x >= 0) ||
//				!(i + x < height) ||
//				!(j + y >= 0) ||
//				!(j + y < width)
//				) {
//				continue;
//			}
//
//			outputQ[(i + x) * width + (j + y)] += (ak * static_cast<float>(gIs[sm_index * width + (j + y)]) + bk);
//			__syncthreads();
//		}
//		sm_index++;
//	}
//
//	return;
//}
//
//__global__ void imNormalizing(float* outputQ, int width, int height) {
//
//	int i = blockIdx.x;  // height
//	int j = threadIdx.x;  // width
//
//	if (i >= height || j >= width)
//		return;
//
//	// normalize.
//	int width_max = width < (j + (windowWidth - 1) / 2) ? width : (j + (windowWidth - 1) / 2);
//	int width_min = 0 > (j - (windowWidth - 1) / 2) ? 0 : (j - (windowWidth - 1) / 2);
//	int height_max = height < (i + (windowHeight - 1) / 2) ? height : (i + (windowHeight - 1) / 2);
//	int height_min = 0 > (i - (windowHeight - 1) / 2) ? 0 : (i - (windowHeight - 1) / 2);
//	int w_num = (width_max - width_min + 1) * (height_max - height_min + 1);
//	outputQ[i * width + j] = outputQ[i * width + j] / static_cast<float>(w_num);
//}

__global__ void doFiltering_new(float* ab, unsigned char* guidedI, float* outputQ, int width, int height) {

	// shared memory.
	__shared__ float aksm[aproneWidth * aproneHeight];
	__shared__ float bksm[aproneWidth * aproneHeight];

	// image coordinates in fP and gI.
	int x = blockIdx.x * tileWidth + threadIdx.x - windowR;  // width.
	int y = blockIdx.y * tileHeight + threadIdx.y - windowR;  // height.
	x = x > 0 ? x : 0;
	x = x < (width - 1) ? x : (width - 1);
	y = y > 0 ? y : 0;
	y = y < (height - 1) ? y : (height - 1);

	// index.
	unsigned index = y * width + x;  // image index.
	unsigned smindex = threadIdx.y * blockDim.x + threadIdx.x;  // shared memory index.

	// data copy.
	aksm[smindex] = ab[index * 2];
	bksm[smindex] = ab[index * 2 + 1];
	__syncthreads();

	// aprone area checking
	if (threadIdx.x < windowR || threadIdx.x >= (aproneWidth - windowR) ||
		threadIdx.y < windowR || threadIdx.y >= (aproneHeight - windowR) ||
		(blockIdx.x * tileWidth + threadIdx.x - windowR) >= width ||
		(blockIdx.y * tileHeight + threadIdx.y - windowR) >= height)
		return;

	/* start filtering.*/
	float Iij = static_cast<float>(guidedI[index]);
	float result = 0.0;
	// (use only other pixel value.)
	for (int dy = -windowR; dy <= windowR; dy++) {
		for (int dx = -windowR; dx <= windowR; dx++) {
			result += aksm[smindex + (dy * blockDim.x) + dx] * Iij + bksm[smindex + (dy * blockDim.x) + dx];
		}
	}

	// normalize.
	//int width_max = (width-1) < (j + (windowWidth - 1) / 2) ? (width-1) : (j + (windowWidth - 1) / 2);
	//int width_min = 0 > (j - (windowWidth - 1) / 2) ? 0 : (j - (windowWidth - 1) / 2);
	//int height_max = (height-1) < (i + (windowHeight - 1) / 2) ? (height-1) : (i + (windowHeight - 1) / 2);
	//int height_min = 0 > (i - (windowHeight - 1) / 2) ? 0 : (i - (windowHeight - 1) / 2);
	//int w_num = (width_max - width_min + 1) * (height_max - height_min + 1);
	outputQ[index] = result / static_cast<float>(windowSize);
	return;
}

