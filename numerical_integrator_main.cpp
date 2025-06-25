#include <iostream>
#include <functional>
#include <chrono>
using namespace std;

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
        double rectangleRule(const function<double(double)>& f, double a, double b, int n) {
            double delta_x = (b - a) / n;
            double sum = 0;

            for(int i = 0; i < n; i++) {
                sum += f(a + (delta_x * i));
            }
            sum = sum * delta_x;

            return sum;
        }

        double trapezoidalRule(const std::function<double(double)>& f, double a, double b, int n) {
            double delta_x = (b - a) / n;
            double sum = 0.5 * (f(a) + f(b));

            for(int i = 1; i < n; i++) {
                sum += f(a + (delta_x * i));
            }
            sum = sum * delta_x;

            return sum;
        }

        double integrate(const std::function<double(double)>& f, double a, double b, int n, Method method) {
            switch (method) {
                case Method::RECTANGLE:
                    return rectangleRule(f, a, b, n);
                case Method::TRAPEZOIDAL:
                    return trapezoidalRule(f, a, b, n);
                default:
                    return trapezoidalRule(f, a, b, n);
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
};

double testFunction(double x) { 
    return x * x;
}

int main() {
    NumericalIntegrator integrator;

    // Test: integrate x^2 from 0 to 1
    // Answer should be 0.333333

    auto benchmark_result1 = integrator.benchmark(testFunction, 0.0, 1.0, 100000000, RECTANGLE);
    cout << "Result: " << benchmark_result1.result << endl;
    cout << "Time: " << benchmark_result1.timeMs << " ms" << endl;

    auto benchmark_result2 = integrator.benchmark(testFunction, 0.0, 1.0, 100000000, TRAPEZOIDAL);
    cout << "Result: " << benchmark_result2.result << endl;
    cout << "Time: " << benchmark_result2.timeMs << " ms" << endl;

    return 0;
}