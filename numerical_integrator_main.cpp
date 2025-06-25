#include <iostream>
#include <functional>
using namespace std;

class NumericalIntegrator {
    public:
        double rectangleRule(const function<double(double)>& f, double a, double b, int n) {
            double delta_x = (b - a) / n;
            double sum = 0;

            for(double i = 0; i < n; i++) {
                sum += f(a + (delta_x * i));
            }
            sum = sum * delta_x;

            return sum;
        }
};

double testFunction(double x) { 
    return x * x;
}

int main() {
    NumericalIntegrator integrator;

    // Test: integrate x^2 from 0 to 1
    // Answer should be 0.333333

    double result = integrator.rectangleRule(testFunction, 0.0, 1.0, 1000000000);
    cout << "Rectangle rule result: " << result << endl;

    return 0;
}