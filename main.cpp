#include "headers.hpp"
#include <numbers>

int main() {
    NumericalIntegrator integrator;
    
    auto benchmark_result1 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, RECTANGLE);
    std::cout << "Result: " << benchmark_result1.result << std::endl;
    std::cout << "Time: " << benchmark_result1.timeMs << " ms" << std::endl;

    auto benchmark_result2 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, TRAPEZOIDAL);
    std::cout << "Result: " << benchmark_result2.result << std::endl;
    std::cout << "Time: " << benchmark_result2.timeMs << " ms" << std::endl;

    auto benchmark_result3 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 50, SIMPSON);
    std::cout << "Result: " << benchmark_result3.result << std::endl;
    std::cout << "Time: " << benchmark_result3.timeMs << " ms" << std::endl;

    auto benchmark_result4 = integrator.benchmark(TestFunctions::trigonometric, 0.0, std::numbers::pi, 50, RECTANGLE);
    std::cout << "Result: " << benchmark_result4.result << std::endl;
    std::cout << "Time: " << benchmark_result4.timeMs << " ms" << std::endl;

    auto benchmark_result5 = integrator.benchmark(TestFunctions::trigonometric, 0.0, std::numbers::pi, 50, TRAPEZOIDAL);
    std::cout << "Result: " << benchmark_result5.result << std::endl;
    std::cout << "Time: " << benchmark_result5.timeMs << " ms" << std::endl;

    auto benchmark_result6 = integrator.benchmark(TestFunctions::trigonometric, 0.0, std::numbers::pi, 50, SIMPSON);
    std::cout << "Result: " << benchmark_result6.result << std::endl;
    std::cout << "Time: " << benchmark_result6.timeMs << " ms" << std::endl;

    auto benchmark_result7 = integrator.benchmark(TestFunctions::exponential, 0.0, 1.0, 50, RECTANGLE);
    std::cout << "Result: " << benchmark_result7.result << std::endl;
    std::cout << "Time: " << benchmark_result7.timeMs << " ms" << std::endl;

    auto benchmark_result8 = integrator.benchmark(TestFunctions::exponential, 0.0, 1.0, 50, TRAPEZOIDAL);
    std::cout << "Result: " << benchmark_result8.result << std::endl;
    std::cout << "Time: " << benchmark_result8.timeMs << " ms" << std::endl;

    auto benchmark_result9 = integrator.benchmark(TestFunctions::exponential, 0.0, 1.0, 50, SIMPSON);
    std::cout << "Result: " << benchmark_result9.result << std::endl;
    std::cout << "Time: " << benchmark_result9.timeMs << " ms" << std::endl;

    return 0;
}