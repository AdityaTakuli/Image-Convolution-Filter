#!/usr/bin/env julia
# =============================================================================
#  Parallel Matrix Multiplication in Julia
#  Application: Image Convolution Filter (Edge Detection using Sobel Operator)
#
#  Demonstrates Julia's native multi-threading and distributed computing
#  capabilities for parallel matrix multiplication — applied to a real-world
#  image processing pipeline.
# =============================================================================

using Base.Threads
using LinearAlgebra
using Printf
using Random
using Statistics

# ─────────────────────────────────────────────────────────────────────────────
#  1. NAIVE (SEQUENTIAL) MATRIX MULTIPLICATION
# ─────────────────────────────────────────────────────────────────────────────
"""
    naive_multiply(A, B)

Standard O(n³) sequential matrix multiplication.
Used as a baseline for performance comparison.
"""
function naive_multiply(A::Matrix{Float64}, B::Matrix{Float64})::Matrix{Float64}
    m, k = size(A)
    _, n = size(B)
    C = zeros(Float64, m, n)

    for i in 1:m
        for j in 1:n
            s = 0.0
            for p in 1:k
                s += A[i, p] * B[p, j]
            end
            C[i, j] = s
        end
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
#  2. PARALLEL MATRIX MULTIPLICATION (Multi-threaded)
# ─────────────────────────────────────────────────────────────────────────────
"""
    parallel_multiply_threads(A, B)

Parallelized matrix multiplication using Julia's `Threads.@threads`.
Distributes row computation across available CPU threads.
"""
function parallel_multiply_threads(A::Matrix{Float64}, B::Matrix{Float64})::Matrix{Float64}
    m, k = size(A)
    _, n = size(B)
    C = zeros(Float64, m, n)

    @threads for i in 1:m
        for j in 1:n
            s = 0.0
            for p in 1:k
                s += A[i, p] * B[p, j]
            end
            C[i, j] = s
        end
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
#  3. BLOCK-PARALLEL MATRIX MULTIPLICATION (Cache-friendly + Threaded)
# ─────────────────────────────────────────────────────────────────────────────
"""
    parallel_multiply_blocked(A, B; block_size=64)

Cache-optimized blocked matrix multiplication with multi-threading.
Divides the matrices into sub-blocks for better cache locality, then
parallelizes across blocks.
"""
function parallel_multiply_blocked(A::Matrix{Float64}, B::Matrix{Float64};
                                    block_size::Int=64)::Matrix{Float64}
    m, k = size(A)
    _, n = size(B)
    C = zeros(Float64, m, n)

    # Calculate number of blocks
    mb = ceil(Int, m / block_size)
    nb = ceil(Int, n / block_size)
    kb = ceil(Int, k / block_size)

    # Create a list of (block_i, block_j) tasks
    tasks = [(bi, bj) for bi in 1:mb for bj in 1:nb]

    @threads for t in 1:length(tasks)
        bi, bj = tasks[t]
        i_start = (bi - 1) * block_size + 1
        i_end   = min(bi * block_size, m)
        j_start = (bj - 1) * block_size + 1
        j_end   = min(bj * block_size, n)

        for bk in 1:kb
            k_start = (bk - 1) * block_size + 1
            k_end   = min(bk * block_size, k)

            @inbounds for i in i_start:i_end
                for p in k_start:k_end
                    a_ip = A[i, p]
                    for j in j_start:j_end
                        C[i, j] += a_ip * B[p, j]
                    end
                end
            end
        end
    end
    return C
end

# ─────────────────────────────────────────────────────────────────────────────
#  4. BLAS-ACCELERATED MULTIPLICATION (Julia's built-in, for reference)
# ─────────────────────────────────────────────────────────────────────────────
"""
    blas_multiply(A, B)

Uses Julia's built-in BLAS-backed `*` operator.
This is the gold standard for dense matrix multiplication performance.
"""
function blas_multiply(A::Matrix{Float64}, B::Matrix{Float64})::Matrix{Float64}
    return A * B
end

# ─────────────────────────────────────────────────────────────────────────────
#  5. APPLICATION: IMAGE CONVOLUTION (Sobel Edge Detection)
# ─────────────────────────────────────────────────────────────────────────────

# Sobel kernels for edge detection
const SOBEL_X = Float64[-1  0  1;
                         -2  0  2;
                         -1  0  1]

const SOBEL_Y = Float64[-1 -2 -1;
                          0  0  0;
                          1  2  1]

"""
    generate_synthetic_image(height, width)

Generate a synthetic grayscale image (matrix) with patterns
to simulate a real image for edge detection demonstration.
"""
function generate_synthetic_image(height::Int, width::Int)::Matrix{Float64}
    img = zeros(Float64, height, width)
    for i in 1:height
        for j in 1:width
            # Create a pattern with edges (rectangles + gradient)
            if 0.25*height < i < 0.75*height && 0.25*width < j < 0.75*width
                img[i, j] = 200.0
            elseif 0.1*height < i < 0.9*height && 0.1*width < j < 0.9*width
                img[i, j] = 100.0
            else
                img[i, j] = Float64(i + j) / Float64(height + width) * 50.0
            end
        end
    end
    return img
end

"""
    apply_convolution_sequential(image, kernel)

Apply a convolution kernel to an image sequentially.
"""
function apply_convolution_sequential(image::Matrix{Float64},
                                       kernel::Matrix{Float64})::Matrix{Float64}
    h, w = size(image)
    kh, kw = size(kernel)
    pad_h = kh ÷ 2
    pad_w = kw ÷ 2
    output = zeros(Float64, h, w)

    for i in (pad_h+1):(h-pad_h)
        for j in (pad_w+1):(w-pad_w)
            s = 0.0
            for ki in 1:kh
                for kj in 1:kw
                    s += image[i - pad_h + ki - 1, j - pad_w + kj - 1] * kernel[ki, kj]
                end
            end
            output[i, j] = s
        end
    end
    return output
end

"""
    apply_convolution_parallel(image, kernel)

Apply a convolution kernel to an image using multi-threaded parallelism.
"""
function apply_convolution_parallel(image::Matrix{Float64},
                                     kernel::Matrix{Float64})::Matrix{Float64}
    h, w = size(image)
    kh, kw = size(kernel)
    pad_h = kh ÷ 2
    pad_w = kw ÷ 2
    output = zeros(Float64, h, w)

    @threads for i in (pad_h+1):(h-pad_h)
        for j in (pad_w+1):(w-pad_w)
            s = 0.0
            for ki in 1:kh
                for kj in 1:kw
                    s += image[i - pad_h + ki - 1, j - pad_w + kj - 1] * kernel[ki, kj]
                end
            end
            output[i, j] = s
        end
    end
    return output
end

"""
    sobel_edge_detect(image; parallel=true)

Full Sobel edge detection: applies Sobel-X and Sobel-Y kernels,
then computes gradient magnitude √(Gx² + Gy²).
"""
function sobel_edge_detect(image::Matrix{Float64}; parallel::Bool=true)::Matrix{Float64}
    if parallel
        gx = apply_convolution_parallel(image, SOBEL_X)
        gy = apply_convolution_parallel(image, SOBEL_Y)
    else
        gx = apply_convolution_sequential(image, SOBEL_X)
        gy = apply_convolution_sequential(image, SOBEL_Y)
    end
    return sqrt.(gx .^ 2 .+ gy .^ 2)
end

# ─────────────────────────────────────────────────────────────────────────────
#  6. BENCHMARKING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    benchmark(f, args...; warmup=1, runs=3)

Benchmark a function with warmup runs, returning median elapsed time in seconds.
"""
function benchmark(f::Function, args...; warmup::Int=1, runs::Int=3)::Float64
    # Warmup
    for _ in 1:warmup
        f(args...)
    end

    times = Float64[]
    for _ in 1:runs
        t = @elapsed f(args...)
        push!(times, t)
    end
    return median(times)
end

"""
    verify_correctness(A, B)

Verify that all multiplication methods produce identical results.
"""
function verify_correctness(A::Matrix{Float64}, B::Matrix{Float64})
    ref = blas_multiply(A, B)

    methods = [
        ("Naive Sequential",    naive_multiply),
        ("Threaded Parallel",   parallel_multiply_threads),
        ("Blocked Parallel",    parallel_multiply_blocked),
    ]

    println("  Correctness Verification (vs BLAS):")
    for (name, method) in methods
        result = method(A, B)
        max_err = maximum(abs.(result .- ref))
        status = max_err < 1e-10 ? "✓ PASS" : "✗ FAIL"
        @printf("    %-25s  max error = %.2e  %s\n", name, max_err, status)
    end
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
#  7. MAIN DRIVER
# ─────────────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 72)
    println("  PARALLEL MATRIX MULTIPLICATION IN JULIA")
    println("  Application: Image Processing — Sobel Edge Detection")
    println("=" ^ 72)
    @printf("  Julia version : %s\n", VERSION)
    @printf("  Threads       : %d\n", nthreads())
    @printf("  BLAS threads  : %d\n", BLAS.get_num_threads())
    println("=" ^ 72)

    # ── Part A: Matrix Multiplication Benchmark ──────────────────────────
    println("\n┌─────────────────────────────────────────────────────────────┐")
    println("│  PART A: Dense Matrix Multiplication Benchmark             │")
    println("└─────────────────────────────────────────────────────────────┘\n")

    sizes = [256, 512, 1024]

    for n in sizes
        println("  Matrix Size: $(n) × $(n)")
        println("  " * "─" ^ 55)

        Random.seed!(42)
        A = rand(Float64, n, n)
        B = rand(Float64, n, n)

        # Skip naive for large sizes (too slow)
        if n <= 512
            verify_correctness(A, B)
        end

        methods = if n <= 512
            [
                ("Naive Sequential",    () -> naive_multiply(A, B)),
                ("Threaded Parallel",   () -> parallel_multiply_threads(A, B)),
                ("Blocked Parallel",    () -> parallel_multiply_blocked(A, B)),
                ("BLAS (built-in)",     () -> blas_multiply(A, B)),
            ]
        else
            [
                ("Threaded Parallel",   () -> parallel_multiply_threads(A, B)),
                ("Blocked Parallel",    () -> parallel_multiply_blocked(A, B)),
                ("BLAS (built-in)",     () -> blas_multiply(A, B)),
            ]
        end

        println("  Benchmark Results:")
        blas_time = 0.0
        for (name, method) in methods
            t = benchmark(method; warmup=1, runs=3)
            if name == "BLAS (built-in)"
                blas_time = t
            end
            gflops = (2.0 * n^3) / t / 1e9
            @printf("    %-25s  %8.4f s   %6.2f GFLOP/s\n", name, t, gflops)
        end

        # Speedup table
        if blas_time > 0
            println("\n  Speedup vs BLAS:")
            for (name, method) in methods
                t = benchmark(method; warmup=0, runs=1)
                speedup = t / blas_time
                @printf("    %-25s  %.2f×\n", name, speedup)
            end
        end
        println()
    end

    # ── Part B: Image Convolution Application ────────────────────────────
    println("\n┌─────────────────────────────────────────────────────────────┐")
    println("│  PART B: Image Convolution — Sobel Edge Detection          │")
    println("└─────────────────────────────────────────────────────────────┘\n")

    img_sizes = [(512, 512), (1024, 1024), (2048, 2048)]

    for (h, w) in img_sizes
        @printf("  Image Size: %d × %d pixels\n", h, w)
        println("  " * "─" ^ 45)

        image = generate_synthetic_image(h, w)

        t_seq = benchmark(sobel_edge_detect, image; warmup=1, runs=3)
        t_par = benchmark(() -> sobel_edge_detect(image; parallel=true); warmup=1, runs=3)

        speedup = t_seq / t_par
        edges = sobel_edge_detect(image; parallel=true)
        edge_pixels = count(x -> x > 50.0, edges)

        @printf("    Sequential         : %8.4f s\n", t_seq)
        @printf("    Parallel (%d thds) : %8.4f s\n", nthreads(), t_par)
        @printf("    Speedup            : %5.2f×\n", speedup)
        @printf("    Edges detected     : %d pixels (%.1f%%)\n",
                edge_pixels, 100.0 * edge_pixels / (h * w))
        println()
    end

    # ── Summary ──────────────────────────────────────────────────────────
    println("=" ^ 72)
    println("  SUMMARY")
    println("=" ^ 72)
    println("  Julia's key advantages for parallel matrix operations:")
    println("    • Native multi-threading with @threads (zero boilerplate)")
    println("    • Built-in BLAS integration for production workloads")
    println("    • Type-stable code compiles to efficient native code")
    println("    • Easy transition from sequential to parallel")
    println("    • First-class support for linear algebra operations")
    println("=" ^ 72)
end

# Run
main()
