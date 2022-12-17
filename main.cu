#include<cuda.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cinttypes>


const double pi = std::atan(1) / 4;
const double delta = 1e-4;
const double eps = 1e-10;

double Ax = -0.353, Bx = 0.353, Ay = 0.3, By = Ay, C = 3 * pi / 8;

struct VarsToShare {
  double Ax;
  double Bx;
  double Ay;
  double By;
  double C;
  double Pi;
  double Delta;
  double Eps;
} typedef Vars;

__device__ double calculateDistance(double* x0, double* x1, size_t n) {
  double sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += std::pow(x0[i] - x1[i], 2);
  }
  return std::sqrt(sum);
}

__device__ void calculateF(double* result, double* input, Vars* vars, bool isSequential) {
   if (isSequential) {
    // printf("Sequential = true\n");
    result[0] = input[0] + input[2] * std::cos(1.5 * vars->Pi - input[3]) - vars->Ax;
    result[1] = input[1] + input[2] * std::cos(1.5 * vars->Pi + input[4]) - vars->Bx;
    result[2] = input[2] + input[2] * std::sin(1.5 * vars->Pi - input[3]) - vars->Ay;
    result[3] = (input[3] + input[4]) * input[2] + (input[1] - input[0]) - vars->C;
    result[4] = input[2] + input[2] * std::sin(1.5 * vars->Pi + input[4]) - vars->By;
  } else {
    std::uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x == 0) {
      // printf("Thread 0\n");
      // printf("Before:Res_0 = %.14lf\n", result[0]);
      result[0] = input[0] + input[2] * std::cos(1.5 * vars->Pi - input[3]) - vars->Ax;
      // printf("After:Res_0 = %.14lf\n", result[0]);
    } else if (threadIdx.x == 1){
      // printf("Thread 1\n");
      // printf("Before:Res_1 = %.14lf\n", result[1]);
      // printf("input[1] = %lf input[2] = %lf std::cos(1.5 * vars->Pi + input[4]) = %lf vars->Bx = %lf\n",input[1], input[2], std::cos(1.5 * vars->Pi + input[4]), vars->Bx);

      result[1] = input[1] + input[2] * std::cos(1.5 * vars->Pi + input[4]) - vars->Bx;
      // printf("After:Res_1 = %.14lf\n", result[1]);
    } else if (threadIdx.x == 2) {
      // printf("Thread 2\n");
      // printf("Before:Res_2 = %.14lf\n", result[2]);
      result[2] = input[2] + input[2] * std::sin(1.5 * vars->Pi - input[3]) - vars->Ay;
      // printf("After:Res_2 = %.14lf\n", result[2]);
    } else if (threadIdx.x == 3) {
      // printf("Thread 3\n");
      // printf("Before:Res_3 = %.14lf\n", result[3]);
      result[3] = (input[3] + input[4]) * input[2] + (input[1] - input[0]) - vars->C;
      // printf("After:Res_3 = %.14lf\n", result[3]);
    } else if (threadIdx.x == 4) {
      // printf("Thread 4\n");
      // printf("Before:Res_4 = %.14lf\n", result[4]);
      result[4] = input[2] + input[2] * std::sin(1.5 * vars->Pi + input[4]) - vars->By;
      // printf("After:Res_4 = %.14lf\n", result[4]);
    }
    __syncthreads();
  }
}

__device__ void print_progress(unsigned step, double* x0, double* x1) {
  printf("-------------------\n");
  printf("Progress\n");
  printf("Step: %u\n", step);
  printf("x1: %lf %lf\n", x0[0], x1[0]);
  printf("x2: %lf %lf\n", x0[1], x1[1]);
  printf("y: %lf %lf\n", x0[2], x1[2]);
  printf("phi1: %lf %lf\n", x0[3], x1[3]);
  printf("phi2: %lf %lf\n", x0[4], x1[4]);
  printf("-------------------\n");
}

__global__ void calculateValue(double* x0, double* x1, Vars* vars, size_t n, bool isSequential) {
  __shared__ unsigned count;
  double* FValue = new double[n];
  while(true) {
    calculateF(FValue, x0, vars, isSequential);
    __barrier_sync(0);
    if (threadIdx.x == 0) {
      for (size_t i = 0; i < n; ++i) {
        printf("FValue[%lu] %lf\n", i, FValue[i]);
        x1[i] = x0[i] - FValue[i] * vars->Delta;
      }
    }
    __barrier_sync(0);
    if (threadIdx.x == 0) atomicAdd(&count, 1);
    __barrier_sync(0);
    if (!(count%5000) && (threadIdx.x == 0)) print_progress(count, x0, x1);
    __barrier_sync(0);
    if (calculateDistance(x0, x1, n) < vars->Eps) break;
    if (threadIdx.x == 0) {
      for (size_t i = 0; i < n; ++i) {
        x0[i] = x1[i];
      }
    }
    __barrier_sync(0);
  }
  delete[] FValue;
}

__host__ void print_result(double* x) {
    printf("-------------------\n");
    printf("Result\n");
    printf("x1 : %lf\n", x[0]);
    printf("x2 : %lf\n", x[1]);
    printf("y : %lf\n", x[2]);
    printf("phi1 : %lf\n", x[3]);
    printf("phi2 : %lf\n", x[4]);
    printf("-------------------\n");
    printf("F(x) = {%.10e, %.10e, %.10e, %.10e, %.10e}\n",
      x[0] + x[2] * std::cos(1.5 * pi - x[3]) - Ax,
      x[1] + x[2] * std::cos(1.5 * pi + x[4]) - Bx,
      x[2] + x[2] * std::sin(1.5 * pi - x[3]) - Ay,
      (x[3] + x[4]) * x[2] + (x[1] - x[0]) - C,
      x[2] + x[2] * std::sin(1.5 * pi + x[4]) - By
    );
}

int main() {
  Vars* vars;
  const int NumOfEquations = 5;
  double *x0, *x1;
  cudaMallocManaged(&x0, sizeof(double) * NumOfEquations);
  cudaMallocManaged(&x1, sizeof(double) * NumOfEquations);
  //preassign vals
  x0[0] = -0.1;
  x0[1] = 0.1;
  x0[2] = 0.0;
  x0[3] = 2.0;
  x0[4] = 2.0;

  x1[0] = 0.0;
  x1[1] = 0.0;
  x1[2] = 0.0;
  x1[3] = 0.0;
  x1[4] = 0.0;
  cudaMallocManaged(&vars, sizeof(Vars));
  vars->Ax = Ax, vars->Ay = Ay, 
  vars->Bx = Bx, vars->By = By, 
  vars->C = C, vars->Pi = pi, 
  vars->Delta = delta, vars->Eps = eps;

  int numBlocks = 1, numThreadsPerBlock = 5;
  bool isSequential = ((numBlocks == 1) && (numThreadsPerBlock == 1)) ? true : false;

  calculateValue<<<numBlocks, numThreadsPerBlock>>>(x0, x1, vars, NumOfEquations, isSequential);
  cudaDeviceSynchronize();
  print_result(x0);
  cudaFree(&x0);
  cudaFree(&x1);
  return 0;
}