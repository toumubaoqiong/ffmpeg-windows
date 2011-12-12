#include "wavelets.h"

extern void cuda_iwt_13_5(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iiwt_13_5(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iwt_5_3(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iiwt_5_3(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iwt_9_3(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iiwt_9_3(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iwt_9_7(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iiwt_9_7(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iwt_fidelity(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iiwt_fidelity(int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iwt_haar(int shift, int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);
extern void cuda_iiwt_haar(int shift, int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream);

extern "C" {

void cuda_wavelet_inverse_transform_2d (int filter, int16_t *data, int stride, int width, int height, cudaStream_t stream)
{
  stride /= 2;
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      cuda_iiwt_9_3 (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_5_3:
      cuda_iiwt_5_3 (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_13_5:
      cuda_iiwt_13_5 (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_HAAR_0:
      cuda_iiwt_haar (0, data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_HAAR_1:
      cuda_iiwt_haar (1, data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_HAAR_2:
      cuda_iiwt_haar (2, data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_FIDELITY:
      cuda_iiwt_fidelity (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      cuda_iiwt_9_7(data, width, height, stride, stream);
      break;
  }
}

void cuda_wavelet_transform_2d (int filter, int16_t *data, int stride, int width, int height, cudaStream_t stream)
{
  stride /= 2;
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      cuda_iwt_9_3 (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_5_3:
      cuda_iwt_5_3 (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_13_5:
      cuda_iwt_13_5 (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_HAAR_0:
      cuda_iwt_haar (0, data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_HAAR_1:
      cuda_iwt_haar (1, data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_HAAR_2:
      cuda_iwt_haar (2, data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_FIDELITY:
      cuda_iwt_fidelity (data, width, height, stride, stream);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      cuda_iwt_9_7(data, width, height, stride, stream);
      break;
  }
}


}
