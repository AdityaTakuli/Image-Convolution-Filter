# ⚡ Parallel Matrix Multiplication — Julia vs C++

> **Application:** Image Processing — Sobel Edge Detection  
> A head-to-head comparison of Julia and C++ for parallel matrix multiplication, applied to a real-world image convolution pipeline.

---

## 📌 Overview

This project implements **parallel dense matrix multiplication** in both **Julia** and **C++**, applied to **Sobel edge detection** on synthetic images. It demonstrates:

- Sequential (naïve) matrix multiplication
- Multi-threaded parallel multiplication
- Cache-optimized blocked parallel multiplication
- BLAS-accelerated multiplication (Julia only)
- Full Sobel edge detection pipeline as a practical application

The goal is to compare **developer productivity**, **performance**, and **ease of parallelism** between the two languages.

---

## 📂 Project Structure

```
JuliaMultiplier/
├── parallel_multiply.jl     # Julia implementation
├── parallel_multiply.cpp    # C++ implementation
└── README.md                # This file
```

---

## 🚀 Quick Start

### Julia

```bash
# Run with multiple threads (e.g., 8 threads)
julia --threads=8 parallel_multiply.jl

# Or set the environment variable
export JULIA_NUM_THREADS=8
julia parallel_multiply.jl
```

**Requirements:** Julia 1.6+ (no external packages needed — uses only the standard library)

### C++ (with OpenMP)

```bash
# Linux / macOS (GCC)
g++ -O3 -fopenmp -std=c++17 -o parallel_multiply parallel_multiply.cpp
./parallel_multiply

# macOS (Clang + libomp via Homebrew)
clang++ -O3 -Xpreprocessor -fopenmp -lomp -std=c++17 -o parallel_multiply parallel_multiply.cpp

# Windows (MSVC)
cl /O2 /openmp /std:c++17 /EHsc parallel_multiply.cpp
parallel_multiply.exe
```

**Requirements:** C++17 compiler with OpenMP support

---

## 🧪 What It Does

### Part A — Matrix Multiplication Benchmark

Benchmarks four multiplication strategies across matrix sizes (**256×256**, **512×512**, **1024×1024**):

| Method | Julia | C++ |
|---|---|---|
| **Naïve Sequential** | Triple nested loop | Triple nested loop |
| **Threaded Parallel** | `@threads` macro | `#pragma omp parallel for` |
| **Blocked Parallel** | Block tiling + `@threads` | Block tiling + `#pragma omp` |
| **BLAS Accelerated** | Built-in `A * B` | ❌ Requires external library |

Each run reports:
- Execution time (seconds)
- GFLOP/s throughput
- Speedup ratios
- Correctness verification

### Part B — Sobel Edge Detection (Application)

Applies the matrix multiplication concepts to a practical image processing task:

1. **Generates** a synthetic grayscale image with geometric patterns
2. **Applies** Sobel-X and Sobel-Y convolution kernels (3×3 matrix multiplications)
3. **Computes** gradient magnitude: `√(Gx² + Gy²)`
4. **Compares** sequential vs. parallel convolution performance

Tested on image sizes: **512×512**, **1024×1024**, **2048×2048**

---

## 📊 Julia vs C++ Comparison

### Code Complexity

| Metric | Julia | C++ |
|---|---|---|
| **Lines of Code** | ~250 | ~310 |
| **Parallelism Syntax** | `@threads for` | `#pragma omp parallel for` + compile flags |
| **Matrix Class** | Built-in `Matrix{Float64}` | Custom class required (~40 lines) |
| **BLAS Support** | Built-in (`LinearAlgebra`) | External library (MKL/OpenBLAS) |
| **Random Init** | `rand(Float64, n, n)` | 5+ lines with `<random>` |
| **Element-wise Ops** | `sqrt.(gx .^ 2 .+ gy .^ 2)` | Manual loop |

### Developer Experience

| Aspect | Julia 🟢 | C++ 🔴 |
|---|---|---|
| **Time to first prototype** | Minutes | Hours |
| **Parallelism boilerplate** | 1 macro | Pragma + compiler flags + header |
| **Memory management** | Automatic (GC) | Manual / RAII |
| **Compilation model** | JIT (first-run cost) | AOT (separate compile step) |
| **Type safety** | Optional annotations | Strict static typing |
| **Debugging** | Interactive REPL | GDB/LLDB |

### Performance Characteristics

| Aspect | Julia | C++ |
|---|---|---|
| **Peak throughput** | Near-C via LLVM JIT | Native machine code |
| **BLAS performance** | OpenBLAS/MKL built-in | Requires linking |
| **First-run latency** | ~2-5s (JIT compilation) | None (pre-compiled) |
| **Subsequent runs** | Comparable to C++ | Baseline |
| **Memory overhead** | Higher (GC + runtime) | Lower (no runtime) |

---

## 🔬 Key Code Comparisons

### Parallel Matrix Multiply

**Julia** — 1 line change from sequential:
```julia
@threads for i in 1:m
    for j in 1:n
        s = 0.0
        for p in 1:k
            s += A[i, p] * B[p, j]
        end
        C[i, j] = s
    end
end
```

**C++** — requires OpenMP pragma + compilation flags:
```cpp
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
```

### Edge Detection in One Line

**Julia:**
```julia
edges = sqrt.(gx .^ 2 .+ gy .^ 2)
```

**C++:**
```cpp
for (int i = 0; i < total; ++i) {
    edges.data[i] = std::sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
}
```

---

## 📈 Expected Output

```
========================================================================
  PARALLEL MATRIX MULTIPLICATION IN JULIA
  Application: Image Processing — Sobel Edge Detection
========================================================================
  Julia version : 1.10.x
  Threads       : 8
========================================================================

  PART A: Dense Matrix Multiplication Benchmark

  Matrix Size: 512 × 512
    Naive Sequential           0.3842 s     0.70 GFLOP/s
    Threaded Parallel          0.0614 s     4.37 GFLOP/s
    Blocked Parallel           0.0389 s     6.90 GFLOP/s
    BLAS (built-in)            0.0032 s    83.89 GFLOP/s

  PART B: Image Convolution — Sobel Edge Detection

  Image Size: 1024 × 1024 pixels
    Sequential         :   0.0156 s
    Parallel (8 thds)  :   0.0028 s
    Speedup            :  5.57×
    Edges detected     : 20480 pixels (1.9%)
```

---

## 🏗️ Architecture

```
                    ┌──────────────────────┐
                    │   Synthetic Image    │
                    │  (Matrix H × W)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Sobel Convolution    │
                    │                      │
              ┌─────┤  Kernel × Patch      ├─────┐
              │     │  (Matrix Multiply)   │     │
              │     └──────────────────────┘     │
              │                                  │
     ┌────────▼────────┐              ┌──────────▼────────┐
     │   Sobel-X (Gx)  │              │   Sobel-Y (Gy)    │
     │  Horizontal Edge │              │  Vertical  Edge   │
     └────────┬────────┘              └──────────┬────────┘
              │                                  │
              └─────────────┬────────────────────┘
                            │
                 ┌──────────▼───────────┐
                 │  Gradient Magnitude  │
                 │  √(Gx² + Gy²)       │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │   Edge-Detected      │
                 │   Output Image       │
                 └──────────────────────┘
```

---

## 🤔 When to Use Which?

| Use Case | Recommended |
|---|---|
| **Research / Prototyping** | Julia — interactive REPL, fast iteration |
| **Production HPC** | Either — both compile to fast native code |
| **Embedded / Systems** | C++ — no runtime overhead |
| **Data Science Pipeline** | Julia — first-class array support |
| **Existing C/C++ Codebase** | C++ — seamless integration |
| **Teaching Parallelism** | Julia — minimal boilerplate |

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -m 'Add amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Open a Pull Request

---

## ⭐ Acknowledgements

- [Julia Documentation — Multi-Threading](https://docs.julialang.org/en/v1/manual/multi-threading/)
- [OpenMP Specification](https://www.openmp.org/specifications/)
- [Sobel Operator — Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator)

---
