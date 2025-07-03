#include "headers.hpp"
#include <numbers>

int main(int argc, char* argv[]) {
    
    Kokkos::initialize(argc, argv);
    // https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html
    
    // auto benchmark_result1 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, RECTANGLE);
    // std::cout << "Result: " << benchmark_result1.result << std::endl;
    // std::cout << "Time: " << benchmark_result1.timeMs << " ms" << std::endl;

    // auto benchmark_result2 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, TRAPEZOIDAL);
    // std::cout << "Result: " << benchmark_result2.result << std::endl;
    // std::cout << "Time: " << benchmark_result2.timeMs << " ms" << std::endl;

    // auto benchmark_result3 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, SIMPSON);
    // std::cout << "Result: " << benchmark_result3.result << std::endl;
    // std::cout << "Time: " << benchmark_result3.timeMs << " ms" << std::endl;

    // auto benchmark_result4 = integrator.benchmark(TestFunctions::trigonometric, 0.0, std::numbers::pi, 50, RECTANGLE);
    // std::cout << "Result: " << benchmark_result4.result << std::endl;
    // std::cout << "Time: " << benchmark_result4.timeMs << " ms" << std::endl;

    // auto benchmark_result5 = integrator.benchmark(TestFunctions::trigonometric, 0.0, std::numbers::pi, 50, TRAPEZOIDAL);
    // std::cout << "Result: " << benchmark_result5.result << std::endl;
    // std::cout << "Time: " << benchmark_result5.timeMs << " ms" << std::endl;

    // auto benchmark_result6 = integrator.benchmark(TestFunctions::trigonometric, 0.0, std::numbers::pi, 50, SIMPSON);
    // std::cout << "Result: " << benchmark_result6.result << std::endl;
    // std::cout << "Time: " << benchmark_result6.timeMs << " ms" << std::endl;

    // auto benchmark_result7 = integrator.benchmark(TestFunctions::exponential, 0.0, 1.0, 50, RECTANGLE);
    // std::cout << "Result: " << benchmark_result7.result << std::endl;
    // std::cout << "Time: " << benchmark_result7.timeMs << " ms" << std::endl;

    // auto benchmark_result8 = integrator.benchmark(TestFunctions::exponential, 0.0, 1.0, 50, TRAPEZOIDAL);
    // std::cout << "Result: " << benchmark_result8.result << std::endl;
    // std::cout << "Time: " << benchmark_result8.timeMs << " ms" << std::endl;

    // auto benchmark_result9 = integrator.benchmark(TestFunctions::exponential, 0.0, 1.0, 50, SIMPSON);
    // std::cout << "Result: " << benchmark_result9.result << std::endl;
    // std::cout << "Time: " << benchmark_result9.timeMs << " ms" << std::endl;

        {
        NumericalIntegrator integrator;
        
        std::cout << "=== Kokkos Numerical Integration Demo ===" << std::endl;
        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "Execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
        
        // Original small tests (your existing code)
        std::cout << "\n--- Small Problem Tests (50 intervals) ---" << std::endl;
        
        auto benchmark_result1 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, RECTANGLE);
        std::cout << "Rectangle x^2: " << benchmark_result1.result << " (" << benchmark_result1.timeMs << " ms)" << std::endl;

        auto benchmark_result2 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, TRAPEZOIDAL);
        std::cout << "Trapezoidal x^2: " << benchmark_result2.result << " (" << benchmark_result2.timeMs << " ms)" << std::endl;

        auto benchmark_result3 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, SIMPSON);
        std::cout << "Simpson x^2: " << benchmark_result3.result << " (" << benchmark_result3.timeMs << " ms)" << std::endl;

        // Large problem tests to show parallel benefits
        std::cout << "\n--- Large Problem Serial vs Parallel Comparison ---" << std::endl;
        
        std::vector<long long> problemSizes = {100000, 1000000, 10000000};
        
        for (long long n : problemSizes) {
            std::cout << "\nProblem size: " << n << " intervals" << std::endl;
            
            // Test polynomial (x^2)
            auto serial = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, n, RECTANGLE);
            auto parallel = integrator.benchmarkParallel(TestFunctions::Polynomial{}, 0.0, 1.0, n, RECTANGLE);
            
            std::cout << "Rectangle x^2:" << std::endl;
            std::cout << "  Serial:   " << std::fixed << std::setprecision(6) << serial.result 
                      << " (" << std::setprecision(1) << serial.timeMs << " ms)" << std::endl;
            std::cout << "  Parallel: " << std::fixed << std::setprecision(6) << parallel.result 
                      << " (" << std::setprecision(1) << parallel.timeMs << " ms)" << std::endl;
            std::cout << "  Speedup:  " << std::setprecision(1) << serial.timeMs / parallel.timeMs << "x" << std::endl;
            std::cout << "  Expected: 0.333333" << std::endl;
        }
        
        // Test different functions with parallel versions
        std::cout << "\n--- Testing Different Functions (Parallel) ---" << std::endl;
        long long n = 1000000;
        
        std::cout << "\nPolynomial x^2 [0,1]:" << std::endl;
        auto poly_result = integrator.benchmarkParallel(TestFunctions::Polynomial{}, 0.0, 1.0, n, TRAPEZOIDAL);
        std::cout << "Result: " << poly_result.result << " (Expected: 0.333333)" << std::endl;
        std::cout << "Time: " << poly_result.timeMs << " ms" << std::endl;
        
        std::cout << "\nTrigonometric sin(x) [0,π]:" << std::endl;
        auto trig_result = integrator.benchmarkParallel(TestFunctions::Trigonometric{}, 0.0, std::numbers::pi, n, TRAPEZOIDAL);
        std::cout << "Result: " << trig_result.result << " (Expected: 2.0)" << std::endl;
        std::cout << "Time: " << trig_result.timeMs << " ms" << std::endl;
        
        std::cout << "\nExponential exp(x) [0,1]:" << std::endl;
        auto exp_result = integrator.benchmarkParallel(TestFunctions::Exponential{}, 0.0, 1.0, n, TRAPEZOIDAL);
        std::cout << "Result: " << exp_result.result << " (Expected: " << (std::exp(1.0) - 1.0) << ")" << std::endl;
        std::cout << "Time: " << exp_result.timeMs << " ms" << std::endl;
        
        // Method comparison
        std::cout << "\n--- Method Accuracy Comparison (Parallel) ---" << std::endl;
        std::cout << "Function: x^2 [0,1], Intervals: " << n << std::endl;
        
        auto rect_par = integrator.benchmarkParallel(TestFunctions::Polynomial{}, 0.0, 1.0, n, RECTANGLE);
        auto trap_par = integrator.benchmarkParallel(TestFunctions::Polynomial{}, 0.0, 1.0, n, TRAPEZOIDAL);
        auto simp_par = integrator.benchmarkParallel(TestFunctions::Polynomial{}, 0.0, 1.0, n, SIMPSON);
        
        double expected = 1.0/3.0;
        std::cout << "Rectangle:   " << rect_par.result << " (Error: " << std::abs(rect_par.result - expected) << ")" << std::endl;
        std::cout << "Trapezoidal: " << trap_par.result << " (Error: " << std::abs(trap_par.result - expected) << ")" << std::endl;
        std::cout << "Simpson:     " << simp_par.result << " (Error: " << std::abs(simp_par.result - expected) << ")" << std::endl;
    }
    
    // Finalize Kokkos
    Kokkos::finalize();
    return 0;



}