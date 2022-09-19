#include "guidedFilter.cuh"

__global__ void darkChannel(unsigned char* orig, unsigned char* dark, unsigned width, unsigned height) {
	// shared memory.
	__shared__ unsigned char smdark[aproneSize];

	// image coordinates in orig.
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // width.
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // height.
	// shared memory start image coordinate in orig. (may be out of image boundary.)
	int smx_start = blockIdx.x * blockDim.x - windowR;
	int smy_start = blockIdx.y * blockDim.y - windowR;

	// index.
	int thindex = threadIdx.y * blockDim.x + threadIdx.x;  // thread index.

	// data copy.
	for (int smindex = thindex; smindex < aproneSize; smindex += (blockDim.x * blockDim.y)) {
		// shared memory image coordinate in orig.
		int smx = smx_start + (smindex % aproneWidth);
		int smy = smy_start + (smindex / aproneWidth);
		smx = smx > 0 ? smx : 0;
		smy = smy > 0 ? smy : 0;
		smx = smx < (width - 1) ? smx : (width - 1);
		smy = smy < (height - 1) ? smy : (height - 1);

		int smindex4 = smy * (width * 4) + smx * 4;  // shared memory index for 4 channels image.
		unsigned char temp = 255;
		for (int ci = 0; ci < 3; ci++) {
			unsigned char temp2 = orig[smindex4 + ci];
			if (temp2 < temp) {
				temp = temp2;
			}
		}
		smdark[smindex] = temp;
	}
	__syncthreads();

	// boundary checking.
	if (x >= width || y >= height)
		return;

	// miniumn filter.
	unsigned char temp = 255;
	for (int dy = -windowR; dy <= windowR; dy++) {
		for (int dx = -windowR; dx <= windowR; dx++) {
			unsigned char temp2 = smdark[(windowR + threadIdx.y + dy) * aproneWidth + (windowR + threadIdx.x + dx)];
			if (temp2 < temp) {
				temp = temp2;
			}
		}
	}
	dark[y * width + x] = temp;

	return;
}

__global__ void subHist(unsigned char* dark, unsigned* subHist, unsigned width, unsigned height) {
	// shared memory.
	__shared__ unsigned local_subHist[nbins];

	// index.
	unsigned index = threadIdx.y * blockDim.x + threadIdx.x;  // thread index.
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;  // width.
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;  // height.
	unsigned imIndex = y * width + x;

	// data initialization.
	for (unsigned bin = index; bin < nbins; bin += (blockDim.x * blockDim.y)) {
		local_subHist[bin] = 0;
	}
	__syncthreads();

	// boundary checking.
	if (x >= width || y >= height)
		return;

	// calculate subHist.
	unsigned char pv = dark[imIndex];
	atomicAdd(&local_subHist[pv], 1);
	__syncthreads();

	// data copy.
	unsigned threadsRemain_w = (blockIdx.x + 1) * blockDim.x - width;
	threadsRemain_w = threadsRemain_w > 0 ? (blockDim.x - threadsRemain_w) : blockDim.x;
	unsigned threadsRemain_h = (blockIdx.y + 1) * blockDim.y - height;
	threadsRemain_h = threadsRemain_h > 0 ? (blockDim.y - threadsRemain_h) : blockDim.y;
	for (unsigned bin = index; bin < nbins; bin += (threadsRemain_w * threadsRemain_h)) {
		subHist[(blockIdx.y * gridDim.x + blockIdx.x) * nbins + bin] = local_subHist[bin];
	}
	
	return;
}

__global__ void subHist_2(unsigned char* dark, unsigned* subHist, unsigned width, unsigned height) {
	// shared memory.
	__shared__ unsigned local_subHist[nbins];

	// index.
	unsigned index = threadIdx.x;  // thread index.
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;  // width.

	// data initialization.
	for (unsigned bin = index; bin < nbins; bin += blockDim.x) {
		local_subHist[bin] = 0;
	}
	__syncthreads();

	// boundary checking.
	if (x >= width)
		return;

	// calculate subHist.
	unsigned char pv = 0;
	for (int i = 0; i < height; i++) {
		pv = dark[width * i + x];
		atomicAdd(&local_subHist[pv], 1);
	}
	__syncthreads();

	// data copy. (some threads has been return!)
	unsigned threadsRemain = (blockIdx.x + 1) * blockDim.x - width;
	threadsRemain = threadsRemain > 0 ? (blockDim.x - threadsRemain) : blockDim.x;
	for (unsigned bin = index; bin < nbins; bin += threadsRemain) {
		subHist[blockIdx.x * nbins + bin] = local_subHist[bin];
	}

	return;
}

__global__ void sumHist(unsigned* subHist, unsigned* hist, unsigned subHistSize) {
	// index.
	unsigned index = threadIdx.x;  // hist index.

	// boundary checking.
	if ((index > nbins))
		return;

	// sum.
	for (int i = index; i < nbins; i += blockDim.x) {
		unsigned sum = 0;
		for (int j = i; j < subHistSize; j += nbins) {
			sum += subHist[j];
		}
		hist[i] = sum;
	}

	return;
}

__global__ void getAcRow(unsigned char* orig, unsigned char* Idark, unsigned char* AcRow, unsigned char colorThresh, unsigned width, unsigned height) {
	// index.
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;  // width.

	// boundary checking.
	if (x >= width)
		return;

	// find AcRow for each column.
	unsigned char local_Ac[3] = {0,0,0};
	for (int y = 0; y < height; y++) {
		if (Idark[y * width + x] >= colorThresh) {
			for (int c = 0; c < 3; c++) {
				unsigned char temp = orig[y * width * 4 + x * 4 + c];
				if ( temp > local_Ac[c]) {
					local_Ac[c] = temp;
				}
			}
		}
	}
	AcRow[x * 3] = local_Ac[0];
	AcRow[x * 3 + 1] = local_Ac[1];
	AcRow[x * 3 + 2] = local_Ac[2];

	return;
}

__global__ void getAc(unsigned char* AcRow, unsigned char* Ac, unsigned width) {
	// index.
	unsigned c = blockIdx.x * blockDim.x + threadIdx.x;  // width.

	// boundary checking.
	if (c >= 3)
		return;

	// find AcRow for each column.
	unsigned char local_Ac = 0;
	for (int x = 0; x < width; x++) {
		unsigned char temp = AcRow[x * 3 + c];
		if (temp > local_Ac) {
			local_Ac = temp;
		}

	}
	Ac[c] = local_Ac;

	return;
}

__global__ void getttilde(unsigned char* orig, float* ttilde, unsigned char* Ac, unsigned width, unsigned height, float w) {
	// shared memory.
	__shared__ float smt[aproneWidth * aproneHeight];

	// image coordinates in orig.
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // width.
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // height.
	// shared memory start image coordinate in orig. (may be out of image boundary.)
	int smx_start = blockIdx.x * blockDim.x - windowR;
	int smy_start = blockIdx.y * blockDim.y - windowR;

	// index.
	int thindex = threadIdx.y * blockDim.x + threadIdx.x;  // thread index.

	// data copy.
	float local_Ac[3];
	local_Ac[0] = static_cast<float>(Ac[0]); static_cast<float>(local_Ac[1] = Ac[1]); static_cast<float>(local_Ac[2] = Ac[2]);
	for (int smindex = thindex; smindex < aproneSize; smindex += (blockDim.x * blockDim.y)) {
		// shared memory image coordinate in orig.
		int smx = smx_start + (smindex % aproneWidth);
		int smy = smy_start + (smindex / aproneWidth);
		smx = smx > 0 ? smx : 0;
		smy = smy > 0 ? smy : 0;
		smx = smx < (width - 1) ? smx : (width - 1);
		smy = smy < (height - 1) ? smy : (height - 1);

		int smindex4 = smy * (width * 4) + smx * 4;  // shared memory index for 4 channels image.
		float temp = 5.0;
		for (int ci = 0; ci < 3; ci++) {
			float temp2 = orig[smindex4 + ci] / local_Ac[ci];
			if (temp2 < temp) {
				temp = temp2;
			}
		}
		smt[smindex] = temp;
	}
	__syncthreads();

	// boundary checking.
	if (x >= width || y >= height)
		return;

	// miniumn filter.
	float temp = 5.0;
	for (int dy = -windowR; dy <= windowR; dy++) {
		for (int dx = -windowR; dx <= windowR; dx++) {
			float temp2 = smt[(windowR + threadIdx.y + dy) * aproneWidth + (windowR + threadIdx.x + dx)];
			if (temp2 < temp) {
				temp = temp2;
			}
		}
	}
	ttilde[y * width + x] = 1.0 - static_cast<float>(w) * temp;

	return;
}

__global__ void gray(unsigned char* orig, unsigned char* gray, unsigned width, unsigned height) {
	int i = blockIdx.x;  // height
	int j = threadIdx.x;  // width

	if (i >= height || j >= width)
		return;

	float temp = 0.299 * static_cast<float>(orig[i * (4 * width) + 4 * j]) +
		0.587 * static_cast<float>(orig[i * (4 * width) + 4 * j + 1]) +
		0.114 * static_cast<float>(orig[i * (4 * width) + 4 * j + 2]);
	gray[i * width + j] = static_cast<unsigned char>(temp);
	
	return;
}

__global__ void linearPara(float* filteringP, unsigned char* guidedI, float* ab, int width, int height, float epsilon) {
	
	// shared memory.
	__shared__ float fPsm[guided_aproneSize];
	__shared__ unsigned char gIsm[guided_aproneSize];

	// image coordinates in fP and gI.
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // width.
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // height.
	// shared memory start image coordinate in orig. (may be out of image boundary.)
	int smx_start = blockIdx.x * blockDim.x - guided_windowR;
	int smy_start = blockIdx.y * blockDim.y - guided_windowR;
	
	// index.
	int thindex = threadIdx.y * blockDim.x + threadIdx.x;  // thread index.
	int outindex = y * width + x;  // image index.

	// data copy.
	for (int smindex = thindex; smindex < guided_aproneSize; smindex += (blockDim.x * blockDim.y)) {
		// shared memory image coordinate in orig.
		int smx = smx_start + (smindex % guided_aproneWidth);
		int smy = smy_start + (smindex / guided_aproneWidth);
		smx = smx > 0 ? smx : 0;
		smy = smy > 0 ? smy : 0;
		smx = smx < (width - 1) ? smx : (width - 1);
		smy = smy < (height - 1) ? smy : (height - 1);

		int imindex = smy * width + smx;  // shared memory index for 1 channels image.
		fPsm[smindex] = filteringP[imindex];
		gIsm[smindex] = guidedI[imindex];
	}
	__syncthreads();

	// boundary checking.
	if (x >= width || y >= height)
		return;

	// (use only other pixel value.)
	float miu = 0.0, pk = 0.0, Ip = 0.0, miu2 = 0.0;
	for (int dy = -guided_windowR; dy <= guided_windowR; dy++) {
		for (int dx = -guided_windowR; dx <= guided_windowR; dx++) {
			float temp1 = static_cast<float>(gIsm[(guided_windowR + threadIdx.y + dy) * guided_aproneWidth + (guided_windowR + threadIdx.x + dx)]);
			float temp2 = fPsm[(guided_windowR + threadIdx.y + dy) * guided_aproneWidth + (guided_windowR + threadIdx.x + dx)];
			miu += temp1;
			pk += temp2;
			Ip += temp1 * temp2;
			miu2 += temp1 * temp1;
		}
	}
	float num = static_cast<float>(guided_windowSize);
	float ak = (Ip / num - (miu / num) * (pk / num)) / (miu2 / num - (miu / num) * (miu / num) + epsilon);
	float bk = (pk / num) - ak * (miu / num);
	ab[outindex * 2] = ak;
	ab[outindex * 2 + 1] = bk;

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
	__shared__ float aksm[guided_aproneSize];
	__shared__ float bksm[guided_aproneSize];

	// image coordinates in fP and gI.
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // width.
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // height.
	// shared memory start image coordinate in orig. (may be out of image boundary.)
	int smx_start = blockIdx.x * blockDim.x - guided_windowR;
	int smy_start = blockIdx.y * blockDim.y - guided_windowR;

	// index.
	int thindex = threadIdx.y * blockDim.x + threadIdx.x;  // thread index.
	int outindex = y * width + x;  // image index.

	// data copy.
	for (int smindex = thindex; smindex < guided_aproneSize; smindex += (blockDim.x * blockDim.y)) {
		// shared memory image coordinate in orig.
		int smx = smx_start + (smindex % guided_aproneWidth);
		int smy = smy_start + (smindex / guided_aproneWidth);
		smx = smx > 0 ? smx : 0;
		smy = smy > 0 ? smy : 0;
		smx = smx < (width - 1) ? smx : (width - 1);
		smy = smy < (height - 1) ? smy : (height - 1);

		int imindex = smy * width * 2 + smx * 2;  // shared memory index for ab.
		aksm[smindex] = ab[imindex];
		bksm[smindex] = ab[imindex + 1];
	}
	__syncthreads();

	// boundary checking.
	if (x >= width || y >= height)
		return;

	/* start filtering.*/
	float Iij = static_cast<float>(guidedI[outindex]);
	float result = 0.0;
	// (use only other pixel value.)
	for (int dy = -guided_windowR; dy <= guided_windowR; dy++) {
		for (int dx = -guided_windowR; dx <= guided_windowR; dx++) {
			result += (aksm[(guided_windowR + threadIdx.y + dy) * guided_aproneWidth + (guided_windowR + threadIdx.x + dx)] * Iij +
				bksm[(guided_windowR + threadIdx.y + dy) * guided_aproneWidth + (guided_windowR + threadIdx.x + dx)]);
		}
	}
	outputQ[outindex] = result / static_cast<float>(guided_windowSize);

	return;
}

__global__ void get_J(unsigned char* orig, float* ttilde2, unsigned char* output, unsigned char* Ac, unsigned width, unsigned height, float t0) {
	// image coordinates.
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // width.
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // height.
													
	// index.
	int thindex = y * width + x;  // image index.

	// boundary checking.
	if (x >= width || y >= height)
		return;

	float local_t = ttilde2[thindex] > t0 ? ttilde2[thindex] : t0;
	for (int ch = 0; ch < 3; ch++) {
		float temp = static_cast<float>(Ac[ch]);
		float temp2 = (static_cast<float>(orig[thindex * 4 + ch]) - temp) / local_t + temp;
		output[thindex * 3 + ch] = static_cast<unsigned char>(temp2);
	}

	return;
}




