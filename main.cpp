#include "headers.hpp"

int main() {
    NumericalIntegrator integrator;

    // Test: integrate x^2 from 0 to 1
    // Answer should be 0.333333

    auto benchmark_result1 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 100000000, RECTANGLE);
    cout << "Result: " << benchmark_result1.result << endl;
    cout << "Time: " << benchmark_result1.timeMs << " ms" << endl;

    auto benchmark_result2 = integrator.benchmark(TestFunctions::polynomial, 0.0, 1.0, 100000000, TRAPEZOIDAL);
    cout << "Result: " << benchmark_result2.result << endl;
    cout << "Time: " << benchmark_result2.timeMs << " ms" << endl;

    return 0;
}