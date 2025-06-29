#include "headers.hpp"

struct TestCase {
    std::function<double(double)> func;
    double a, b;
    double expected;
    std::string name;
};

void runTests(const NumericalIntegrator& integrator) {
    std::vector<TestCase> tests = { 
        {TestFunctions::polynomial, 0.0, 1.0, 1.0/3.0, "x^2"},
        {TestFunctions::trigonometric, 0.0, M_PI, 2.0, "sin(x)"},
        {TestFunctions::exponential, 0.0, 1.0, exp(1.0) - 1.0, "exp(x)"}
    };

    // Loop through each test case
    for (const TestCase& test : tests) {
        std::cout << "\n--- Testing function: " << test.name << " ---" << std::endl;
        std::cout << "Expected result: " << test.expected << std::endl;
        
        // Test rectangle rule
        auto rectResult = integrator.benchmark(test.func, test.a, test.b, 10000, RECTANGLE);
        std::cout << "Rectangle: " << rectResult.result << " (Error: " << std::abs(rectResult.result - test.expected) << ")" << std::endl;
        
        // Test trapezoidal rule  
        auto trapResult = integrator.benchmark(test.func, test.a, test.b, 10000, TRAPEZOIDAL);
        std::cout << "Trapezoidal: " << trapResult.result << " (Error: " << std::abs(trapResult.result - test.expected) << ")" << std::endl;
    }
}   