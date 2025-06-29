#include <iostream>
#include <functional>
#include <vector>
#include <chrono>
#include <cmath>
//#include <Kokkos_Core.hpp>

namespace TestFunctions {
    double polynomial(double x) {
        return x * x;
    }
    double trigonometric(double x) {
        return sin(x);
    }
    double exponential(double x) {
        return exp(x);
    }
}

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
        const double integrate(const std::function<double(double)>& f, double a, double b, long long n, Method method);
        BenchmarkResult benchmark(const std::function<double(double)>& f, double a, double b, long long n, Method method);
    private:
        const double rectanglerule(const std::function<double(double)>& f, double a, double b, long long n);
        const double trapezoidalrule(const std::function<double(double)>& f, double a, double b, long long n);
};