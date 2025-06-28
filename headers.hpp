#include <iostream>
#include <functional>
#include <chrono>
#include <cmath>
//#include <Kokkos_Core.hpp>
using namespace std;

namespace TestFunctions {
    double polynomial(double x) {
        return x * x;
    }
    double trignonometric(double x) {
        return sin(x);
    }
    double exponential(double x) {
        return exp(x);
    }
}

struct TestCase {
    function<double(double)> func;
    double a, b;
    double expected;
    string name;
};

enum Method {
    RECTANGLE,
    TRAPEZOIDAL
};

struct BenchmarkResult {
    double result;
    double timeMs;
};

class NumericalIntegrator {
    public:
        double rectanglerule(const function<double(double)>& f, double a, double b, int n) {
            double delta_x = (b - a) / n;
            double sum = 0;

            for(int i = 0; i < n; i++) {
                sum += f(a + (delta_x * i));
            }
            sum = sum * delta_x;

            return sum;
        }

        double trapezoidalrule(const std::function<double(double)>& f, double a, double b, int n) {
            double delta_x = (b - a) / n;
            double sum = 0.5 * (f(a) + f(b));

            for(int i = 1; i < n; i++) {
                sum += f(a + (delta_x * i));
            }
            sum = sum * delta_x;

            return sum;
        }

        double integrate(const std::function<double(double)>& f, double a, double b, int n, method method) {
            switch (method) {
                case method::RECTANGLE:
                    return rectanglerule(f, a, b, n);
                case method::TRAPEZOIDAL:
                    return trapezoidalrule(f, a, b, n);
                default:
                    return trapezoidalrule(f, a, b, n);
            }    
        }   

        BenchmarkResult benchmark(const std::function<double(double)>& f, double a, double b, int n, Method method) {
            auto start = chrono::high_resolution_clock::now();

            double result = integrate(f, a, b, n, method);

            auto end = chrono::high_resolution_clock::now();

            auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
            double timeMs = duration.count() / 1000.0;

            return {result, timeMs};
        }

        void runTests() {
            vector<TestCase> tests = { 
                {TestFunctions::polynomial, 0.0, 1.0, 1.0/3.0, "x^2"},
                {TestFunctions::trignonometric, 0.0, M_PI, 2.0, "sin(x)"},
                {TestFunctions::exponential, }
            };

            // Loop through each test case
            for (const TestCase& test : tests) {
                std::cout << "\n--- Testing function: " << test.name << " ---" << std::endl;
                std::cout << "Expected result: " << test.expected << std::endl;
                
                // Test rectangle rule
                auto rectResult = benchmark(test.func, test.a, test.b, 10000, RECTANGLE);
                std::cout << "Rectangle: " << rectResult.result << " (Error: " << std::abs(rectResult.result - test.expected) << ")" << std::endl;
                
                // Test trapezoidal rule  
                auto trapResult = benchmark(test.func, test.a, test.b, 10000, TRAPEZOIDAL);
                std::cout << "Trapezoidal: " << trapResult.result << " (Error: " << std::abs(trapResult.result - test.expected) << ")" << std::endl;
            }
        }   
};