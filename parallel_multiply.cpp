// =============================================================================
//  Parallel Matrix Multiplication in C++
//  Application: Image Convolution Filter (Edge Detection using Sobel Operator)
//
//  Equivalent C++ implementation for direct comparison with the Julia version.
//  Uses OpenMP for multi-threading parallelism.
//
//  Compile:
//    g++ -O3 -fopenmp -std=c++17 -o parallel_multiply parallel_multiply.cpp
//    (MSVC) cl /O2 /openmp /std:c++17 /EHsc parallel_multiply.cpp
// =============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <functional>
#include <cassert>

#ifdef _OPENMP
    #include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Matrix Class — Simple 2D Dense Matrix
// ─────────────────────────────────────────────────────────────────────────────
class Matrix {
public:
    int rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    inline double& operator()(int i, int j) {
        return data[i * cols + j];
    }

    inline double operator()(int i, int j) const {
        return data[i * cols + j];
    }

    // Fill with random values [0, 1)
    void randomize(unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& v : data) v = dist(gen);
    }

    // Max absolute difference from another matrix
    double max_diff(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        double max_err = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            max_err = std::max(max_err, std::abs(data[i] - other.data[i]));
        }
        return max_err;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  1. NAIVE (SEQUENTIAL) MATRIX MULTIPLICATION
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Standard O(n³) sequential matrix multiplication.
 * Used as a baseline for performance comparison.
 */
Matrix naive_multiply(const Matrix& A, const Matrix& B) {
    int m = A.rows, k = A.cols, n = B.cols;
    Matrix C(m, n);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < k; ++p) {
                s += A(i, p) * B(p, j);
            }
            C(i, j) = s;
        }
    }
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. PARALLEL MATRIX MULTIPLICATION (OpenMP)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Parallelized matrix multiplication using OpenMP.
 * Distributes row computation across available CPU threads.
 */
Matrix parallel_multiply_openmp(const Matrix& A, const Matrix& B) {
    int m = A.rows, k = A.cols, n = B.cols;
    Matrix C(m, n);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < k; ++p) {
                s += A(i, p) * B(p, j);
            }
            C(i, j) = s;
        }
    }
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. BLOCK-PARALLEL MATRIX MULTIPLICATION (Cache-friendly + OpenMP)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Cache-optimized blocked matrix multiplication with OpenMP.
 * Divides the matrices into sub-blocks for better cache locality.
 */
Matrix parallel_multiply_blocked(const Matrix& A, const Matrix& B, int block_size = 64) {
    int m = A.rows, k = A.cols, n = B.cols;
    Matrix C(m, n);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < m; bi += block_size) {
        for (int bj = 0; bj < n; bj += block_size) {
            int i_end = std::min(bi + block_size, m);
            int j_end = std::min(bj + block_size, n);

            for (int bk = 0; bk < k; bk += block_size) {
                int k_end = std::min(bk + block_size, k);

                for (int i = bi; i < i_end; ++i) {
                    for (int p = bk; p < k_end; ++p) {
                        double a_ip = A(i, p);
                        for (int j = bj; j < j_end; ++j) {
                            C(i, j) += a_ip * B(p, j);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
//  4. APPLICATION: IMAGE CONVOLUTION (Sobel Edge Detection)
// ─────────────────────────────────────────────────────────────────────────────

// Sobel kernels
constexpr double SOBEL_X[3][3] = {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
};

constexpr double SOBEL_Y[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

/**
 * Generate a synthetic grayscale image with geometric patterns
 * that produce detectable edges.
 */
Matrix generate_synthetic_image(int height, int width) {
    Matrix img(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (i > 0.25 * height && i < 0.75 * height &&
                j > 0.25 * width  && j < 0.75 * width) {
                img(i, j) = 200.0;
            } else if (i > 0.1 * height && i < 0.9 * height &&
                       j > 0.1 * width  && j < 0.9 * width) {
                img(i, j) = 100.0;
            } else {
                img(i, j) = static_cast<double>(i + j)
                           / static_cast<double>(height + width) * 50.0;
            }
        }
    }
    return img;
}

/**
 * Apply a 3x3 convolution kernel to an image — SEQUENTIAL.
 */
Matrix apply_convolution_sequential(const Matrix& image, const double kernel[3][3]) {
    int h = image.rows, w = image.cols;
    Matrix output(h, w);

    for (int i = 1; i < h - 1; ++i) {
        for (int j = 1; j < w - 1; ++j) {
            double s = 0.0;
            for (int ki = 0; ki < 3; ++ki) {
                for (int kj = 0; kj < 3; ++kj) {
                    s += image(i - 1 + ki, j - 1 + kj) * kernel[ki][kj];
                }
            }
            output(i, j) = s;
        }
    }
    return output;
}

/**
 * Apply a 3x3 convolution kernel to an image — PARALLEL (OpenMP).
 */
Matrix apply_convolution_parallel(const Matrix& image, const double kernel[3][3]) {
    int h = image.rows, w = image.cols;
    Matrix output(h, w);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < h - 1; ++i) {
        for (int j = 1; j < w - 1; ++j) {
            double s = 0.0;
            for (int ki = 0; ki < 3; ++ki) {
                for (int kj = 0; kj < 3; ++kj) {
                    s += image(i - 1 + ki, j - 1 + kj) * kernel[ki][kj];
                }
            }
            output(i, j) = s;
        }
    }
    return output;
}

/**
 * Full Sobel edge detection: apply Sobel-X and Sobel-Y, compute gradient magnitude.
 */
Matrix sobel_edge_detect(const Matrix& image, bool parallel = true) {
    Matrix gx, gy;
    if (parallel) {
        gx = apply_convolution_parallel(image, SOBEL_X);
        gy = apply_convolution_parallel(image, SOBEL_Y);
    } else {
        gx = apply_convolution_sequential(image, SOBEL_X);
        gy = apply_convolution_sequential(image, SOBEL_Y);
    }

    Matrix edges(image.rows, image.cols);
    int total = image.rows * image.cols;
    for (int i = 0; i < total; ++i) {
        edges.data[i] = std::sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
    }
    return edges;
}

// ─────────────────────────────────────────────────────────────────────────────
//  5. BENCHMARKING UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

/**
 * Benchmark a function with warmup, returning median elapsed time in seconds.
 */
double benchmark(std::function<void()> func, int warmup = 1, int runs = 3) {
    for (int i = 0; i < warmup; ++i) func();

    std::vector<double> times;
    for (int i = 0; i < runs; ++i) {
        auto t0 = Clock::now();
        func();
        auto t1 = Clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(elapsed);
    }
    std::sort(times.begin(), times.end());
    return times[runs / 2]; // median
}

/**
 * Print a formatted line with method name, time, and GFLOP/s.
 */
void print_result(const std::string& name, double time_s, int n) {
    double gflops = (2.0 * n * n * n) / time_s / 1e9;
    std::cout << "    " << std::left << std::setw(25) << name
              << "  " << std::fixed << std::setprecision(4) << std::setw(8) << time_s << " s"
              << "   " << std::setprecision(2) << std::setw(6) << gflops << " GFLOP/s"
              << std::endl;
}

/**
 * Verify correctness of all methods against blocked parallel (as reference).
 */
void verify_correctness(const Matrix& A, const Matrix& B) {
    Matrix ref = parallel_multiply_blocked(A, B);

    struct Method {
        std::string name;
        std::function<Matrix()> func;
    };

    std::vector<Method> methods = {
        {"Naive Sequential",   [&]() { return naive_multiply(A, B); }},
        {"OpenMP Parallel",    [&]() { return parallel_multiply_openmp(A, B); }},
    };

    std::cout << "  Correctness Verification (vs Blocked Parallel):" << std::endl;
    for (auto& m : methods) {
        Matrix result = m.func();
        double max_err = result.max_diff(ref);
        const char* status = (max_err < 1e-10) ? "PASS" : "FAIL";
        std::cout << "    " << std::left << std::setw(25) << m.name
                  << "  max error = " << std::scientific << std::setprecision(2) << max_err
                  << "  " << status << std::endl;
    }
    std::cout << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
//  6. MAIN DRIVER
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  PARALLEL MATRIX MULTIPLICATION IN C++" << std::endl;
    std::cout << "  Application: Image Processing - Sobel Edge Detection" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  C++ Standard  : C++17" << std::endl;
    #ifdef _OPENMP
        std::cout << "  OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
        std::cout << "  OpenMP        : NOT AVAILABLE (single-threaded fallback)" << std::endl;
    #endif
    std::cout << std::string(72, '=') << std::endl;

    // ── Part A: Matrix Multiplication Benchmark ─────────────────────────
    std::cout << "\n+-------------------------------------------------------------+" << std::endl;
    std::cout << "|  PART A: Dense Matrix Multiplication Benchmark              |" << std::endl;
    std::cout << "+-------------------------------------------------------------+\n" << std::endl;

    std::vector<int> sizes = {256, 512, 1024};

    for (int n : sizes) {
        std::cout << "  Matrix Size: " << n << " x " << n << std::endl;
        std::cout << "  " << std::string(55, '-') << std::endl;

        Matrix A(n, n), B(n, n);
        A.randomize(42);
        B.randomize(123);

        if (n <= 512) {
            verify_correctness(A, B);
        }

        std::cout << "  Benchmark Results:" << std::endl;

        if (n <= 512) {
            double t_naive = benchmark([&]() { naive_multiply(A, B); });
            print_result("Naive Sequential", t_naive, n);
        }

        double t_omp = benchmark([&]() { parallel_multiply_openmp(A, B); });
        print_result("OpenMP Parallel", t_omp, n);

        double t_blocked = benchmark([&]() { parallel_multiply_blocked(A, B); });
        print_result("Blocked Parallel", t_blocked, n);

        std::cout << std::endl;
    }

    // ── Part B: Image Convolution Application ───────────────────────────
    std::cout << "\n+-------------------------------------------------------------+" << std::endl;
    std::cout << "|  PART B: Image Convolution - Sobel Edge Detection           |" << std::endl;
    std::cout << "+-------------------------------------------------------------+\n" << std::endl;

    std::vector<std::pair<int,int>> img_sizes = {{512, 512}, {1024, 1024}, {2048, 2048}};

    for (auto& [h, w] : img_sizes) {
        std::cout << "  Image Size: " << h << " x " << w << " pixels" << std::endl;
        std::cout << "  " << std::string(45, '-') << std::endl;

        Matrix image = generate_synthetic_image(h, w);

        double t_seq = benchmark([&]() { sobel_edge_detect(image, false); });
        double t_par = benchmark([&]() { sobel_edge_detect(image, true); });

        double speedup = t_seq / t_par;

        Matrix edges = sobel_edge_detect(image, true);
        int edge_pixels = 0;
        for (auto& v : edges.data) {
            if (v > 50.0) ++edge_pixels;
        }

        std::cout << std::fixed;
        std::cout << "    Sequential         : " << std::setprecision(4) << std::setw(8) << t_seq << " s" << std::endl;
        #ifdef _OPENMP
            std::cout << "    Parallel (" << omp_get_max_threads() << " thds) : "
                      << std::setprecision(4) << std::setw(8) << t_par << " s" << std::endl;
        #else
            std::cout << "    Parallel (1 thd)   : " << std::setprecision(4) << std::setw(8) << t_par << " s" << std::endl;
        #endif
        std::cout << "    Speedup            : " << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "    Edges detected     : " << edge_pixels << " pixels ("
                  << std::setprecision(1) << (100.0 * edge_pixels / (h * w)) << "%)" << std::endl;
        std::cout << std::endl;
    }

    // ── Summary ─────────────────────────────────────────────────────────
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  SUMMARY" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  C++ requires more boilerplate than Julia for parallel computing:" << std::endl;
    std::cout << "    - Manual memory management with Matrix class" << std::endl;
    std::cout << "    - Explicit OpenMP pragmas for parallelism" << std::endl;
    std::cout << "    - Separate compilation flags needed (-fopenmp)" << std::endl;
    std::cout << "    - No built-in BLAS (requires external library like MKL/OpenBLAS)" << std::endl;
    std::cout << "    - More verbose syntax for the same operations" << std::endl;
    std::cout << std::string(72, '=') << std::endl;

    return 0;
}
