#pragma once
#include <iostream>
#include <functional>
#include <vector>
#include <chrono>
#include <cmath>

// These three have been added on in order to run with Kokkos.
#include <thread>
#include <iomanip>
#include <Kokkos_Core.hpp>

namespace TestFunctions {
    inline double polynomial(double x) {
        return x * x;
    }
    inline double trigonometric(double x) {
        return sin(x);
    }
    inline double exponential(double x) {
        return exp(x);
    }

    // Kokkos Functors
    struct Polynomial {
        KOKKOS_INLINE_FUNCTION
        // https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html
        // https://kokkos.org/kokkos-core-wiki/API/core/macros-special/host_device_macros.html#kokkos-inline-function
        double operator()(double x) const { return x * x; }
    };

    struct Trigonometric {
        KOKKOS_INLINE_FUNCTION
        double operator()(double x) const { return Kokkos::sin(x); }
        // https://kokkos.org/kokkos-core-wiki/API/core/numerics/mathematical-functions.html
    };

    struct Exponential {
        KOKKOS_INLINE_FUNCTION
        double operator()(double x) const { return Kokkos::exp(x); }
        // https://kokkos.org/kokkos-core-wiki/API/core/numerics/mathematical-functions.html
    };
}

enum Method {
    RECTANGLE,
    TRAPEZOIDAL,
    SIMPSON
};

struct BenchmarkResult {
    double result;
    double timeMs;
    std::string version = "Serial"; // Added version tracking
};

class NumericalIntegrator {
    public:
        double integrate(const std::function<double(double)>& f, double a, double b, long long n, Method method) const;
        BenchmarkResult benchmark(const std::function<double(double)>& f, double a, double b, long long n, Method method);

        // New Kokkos methods
        template<typename FunctorType>
        double integrateParallel(FunctorType func, double a, double b, long long n, Method method) const;

        template<typename FunctorType>
        BenchmarkResult benchmarkParallel(FunctorType func, double a, double b, long long n, Method method);

        // Comparison method
        void compareSerialVsParallel(double a, double b, long long n, Method method);

    private:
        double rectanglerule(const std::function<double(double)>& f, double a, double b, long long n) const;
        double trapezoidalrule(const std::function<double(double)>& f, double a, double b, long long n) const;
        double simpson(const std::function<double(double)>& f, double a, double b, long long n) const;

        // New Kokkos methods
        template<typename FunctorType>
        double rectangleRuleParallel(FunctorType func, double a, double b, long long n) const;

        template<typename FunctorType>
        double trapezoidalRuleParallel(FunctorType func, double a, double b, long long n) const;

        template<typename FunctorType>
        double simpsonParallel(FunctorType func, double a, double b, long long n) const;
};