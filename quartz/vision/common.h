// What both image encoders share, included inside each encoder's namespace right
// after its API header (the including file has the HIP, rocBLAS, stb and C
// headers): the error checks, the f16 GEMM, the bias add, and the encoded-image
// entry point over the encoder's qz_vision_encode_rgb.
#pragma once

#define VCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "quartz_vision: HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define VBLAS(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { \
    fprintf(stderr, "quartz_vision: rocBLAS error %s at %s:%d\n", rocblas_status_to_string(s_), __FILE__, __LINE__); return -1; } } while (0)

// Y[n][m] (+)= X[n][k] . W[m][k]^T, W f16 [m][k], X f16, Y f32 or f16
static int vgemm(rocblas_handle blas, const __half * W, const __half * X, void * Y, bool y16, int m, int n, int k, float beta = 0.f) {
    const float alpha = 1.f;
    const rocblas_datatype yt = y16 ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
    VBLAS(rocblas_gemm_ex(blas, rocblas_operation_transpose, rocblas_operation_none, m, n, k, &alpha,
                          W, rocblas_datatype_f16_r, k, X, rocblas_datatype_f16_r, k, &beta,
                          Y, yt, m, Y, yt, m, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    return 0;
}

// x[row][i] += b[i]; grid (d / 256, rows)
__global__ void __launch_bounds__(256) v_bias(float * __restrict__ x, const float * __restrict__ b, int d) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < d) x[(size_t) blockIdx.y * d + i] += b[i];
}

// an encoded image (any format stb_image reads) -> RGB -> qz_vision_encode_rgb
int qz_vision_encode(qz_vision * v, const void * bytes, size_t n, int * nx, int * ny) {
    int w = 0, h = 0, c = 0;
    unsigned char * px = stbi_load_from_memory((const stbi_uc *) bytes, (int) n, &w, &h, &c, 3);
    if (!px) { fprintf(stderr, "quartz_vision: cannot decode the image (%s)\n", stbi_failure_reason()); return -1; }
    const int r = qz_vision_encode_rgb(v, px, w, h, nx, ny);
    stbi_image_free(px);
    return r;
}
