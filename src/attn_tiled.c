/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Q3b: cache tiling + 4x16 AVX2/FMA register-tiled micro-kernels (packed K and V panels).
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here. */
/* ------------------------------------------------------------------------- */
#include <immintrin.h>

#define NR 16 /* register tile: 4 rows x 16 columns (2 YMM per row, 8 accumulators) */

static void *amalloc(size_t bytes)
{
    void *p;
    if (posix_memalign(&p, 64, bytes))
    {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static void check_dims(int N, int d)
{
    if (N % 32 || d % 16)
    {
        fprintf(stderr, "this variant needs N %% 32 == 0 and d %% 16 == 0\n");
        exit(1);
    }
}

/* Pack K into panels of 16 keys: Kp[jb][k][c] = K[jb*16 + c][k]  (d x 16 each, contiguous). */
static inline void pack_K_blk(const float *K, float *Kp, int jb, int d)
{
    for (int k = 0; k < d; k++)
        for (int c = 0; c < NR; c++)
            Kp[((size_t)jb * d + k) * NR + c] = K[((size_t)jb * NR + c) * d + k];
}

/* Pack V into panels of 16 columns: Vp[kb][j][c] = V[j][kb*16 + c]  (N x 16 each, contiguous). */
static inline void pack_V_row(const float *V, float *Vp, int j, int N, int d)
{
    for (int kb = 0; kb < d / NR; kb++)
        for (int c = 0; c < NR; c++)
            Vp[((size_t)kb * N + j) * NR + c] = V[(size_t)j * d + kb * NR + c];
}

#define ROW(R, p, i)                           \
    a = _mm256_broadcast_ss((p) + (i));        \
    c##R##0 = _mm256_fmadd_ps(a, b0, c##R##0); \
    c##R##1 = _mm256_fmadd_ps(a, b1, c##R##1);

#define ZERO_ACC                                                       \
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, \
           c20 = c00, c21 = c00, c30 = c00, c31 = c00, a, b0, b1;

/* S[4 x 16] = scale * Q[4 x d] . Kpanel  (Kp is the packed d x 16 panel). */
static inline void qk_kernel(const float *Q, const float *Kp, float *S, int d, int ldS, float scale)
{
    ZERO_ACC
    const float *q0 = Q, *q1 = Q + d, *q2 = Q + 2 * (size_t)d, *q3 = Q + 3 * (size_t)d;
    for (int k = 0; k < d; k++)
    {
        b0 = _mm256_loadu_ps(Kp + (size_t)k * NR);
        b1 = _mm256_loadu_ps(Kp + (size_t)k * NR + 8);
        ROW(0, q0, k)
        ROW(1, q1, k) ROW(2, q2, k) ROW(3, q3, k)
    }
    __m256 sc = _mm256_set1_ps(scale);
    float *s1 = S + ldS, *s2 = S + 2 * (size_t)ldS, *s3 = S + 3 * (size_t)ldS;
    _mm256_storeu_ps(S, _mm256_mul_ps(c00, sc));
    _mm256_storeu_ps(S + 8, _mm256_mul_ps(c01, sc));
    _mm256_storeu_ps(s1, _mm256_mul_ps(c10, sc));
    _mm256_storeu_ps(s1 + 8, _mm256_mul_ps(c11, sc));
    _mm256_storeu_ps(s2, _mm256_mul_ps(c20, sc));
    _mm256_storeu_ps(s2 + 8, _mm256_mul_ps(c21, sc));
    _mm256_storeu_ps(s3, _mm256_mul_ps(c30, sc));
    _mm256_storeu_ps(s3 + 8, _mm256_mul_ps(c31, sc));
}

/* O[4 x 16] = P[4 x N] . Vpanel  (Vp is the packed N x 16 panel). */
static inline void pv_kernel(const float *P, int ldP, const float *Vp, float *O, int ldO, int N)
{
    ZERO_ACC
    const float *p0 = P, *p1 = P + ldP, *p2 = P + 2 * (size_t)ldP, *p3 = P + 3 * (size_t)ldP;
    for (int j = 0; j < N; j++)
    {
        b0 = _mm256_loadu_ps(Vp + (size_t)j * NR);
        b1 = _mm256_loadu_ps(Vp + (size_t)j * NR + 8);
        ROW(0, p0, j)
        ROW(1, p1, j) ROW(2, p2, j) ROW(3, p3, j)
    }
    float *o1 = O + ldO, *o2 = O + 2 * (size_t)ldO, *o3 = O + 3 * (size_t)ldO;
    _mm256_storeu_ps(O, c00);
    _mm256_storeu_ps(O + 8, c01);
    _mm256_storeu_ps(o1, c10);
    _mm256_storeu_ps(o1 + 8, c11);
    _mm256_storeu_ps(o2, c20);
    _mm256_storeu_ps(o2 + 8, c21);
    _mm256_storeu_ps(o3, c30);
    _mm256_storeu_ps(o3 + 8, c31);
}

/* Scalar softmax of one row, exactly as in the naive code. */
static void softmax_row_scalar(float *row, int N)
{
    float max = row[0];
    for (int j = 1; j < N; j++)
        if (row[j] > max)
            max = row[j];
    float sum = 0.0f;
    for (int j = 0; j < N; j++)
    {
        row[j] = expf(row[j] - max);
        sum += row[j];
    }
    for (int j = 0; j < N; j++)
        row[j] = row[j] / sum;
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    check_dims(N, d);
    const float scale = 1.0f / sqrtf((float)d);
    float *S = amalloc((size_t)N * N * sizeof(float));
    float *Kp = amalloc((size_t)N * d * sizeof(float));
    float *Vp = amalloc((size_t)N * d * sizeof(float));
    for (int jb = 0; jb < N / NR; jb++)
        pack_K_blk(K, Kp, jb, d);
    for (int j = 0; j < N; j++)
        pack_V_row(V, Vp, j, N, d);

    /* 1. S = Q K^T * scale.  Cache tiling: a block of IB query rows stays in L1 while
     *    every 16-key panel of K is streamed through it; register tiling: the 4x16
     *    micro-kernel keeps 8 YMM accumulators and reuses each load 4 (B) or 2 (A) times. */
    enum
    {
        IB = 32
    };
    for (int ib = 0; ib < N; ib += IB)
        for (int jp = 0; jp < N / NR; jp++)
            for (int ir = ib; ir < ib + IB; ir += 4)
                qk_kernel(Q + (size_t)ir * d, Kp + (size_t)jp * d * NR,
                          S + (size_t)ir * N + (size_t)jp * NR, d, N, scale);

    /* 2. softmax of each row (scalar, as in naive). */
    for (int i = 0; i < N; i++)
        softmax_row_scalar(S + (size_t)i * N, N);

    /* 3. O = S V.  IB2 rows of S are re-used against each packed 16-column panel of V. */
    enum
    {
        IB2 = 16
    };
    for (int ib = 0; ib < N; ib += IB2)
        for (int kb = 0; kb < d / NR; kb++)
            for (int ir = ib; ir < ib + IB2; ir += 4)
                pv_kernel(S + (size_t)ir * N, N, Vp + (size_t)kb * N * NR,
                          O + (size_t)ir * d + (size_t)kb * NR, d, N);

    free(Vp);
    free(Kp);
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in)
    {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1)
    {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n)
    {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++)
    {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n)
    {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
