# C-Projects

Compile using kokkos:


g++ -std=c++23 -O3 -fopenmp \
  -I/home/kaelyn/kokkos_install/include \
  -L/home/kaelyn/kokkos_install/lib \
  -lkokkoscore -lkokkoscontainers -lkokkosalgorithms -lkokkossimd \
  -ldl -lpthread \
  main.cpp numerical_integrator_main.cpp -o integrator