#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <sys/time.h>

const double G = 6.674e-11;
const double dt = 1e-3;

const double Threshold = 100;

struct args {
    size_t size;
    double G;
    double dt;
} typedef Args;

struct pair {
    double first;
    double second;
} typedef Pair;

struct matPoint {
    double x;
    double y;
    double vx;
    double vy;
    double m;
} typedef MatPoint;

__device__ Pair* calculateForce(MatPoint* Points, Args* args, Pair* Result) {
    for (size_t i = 0; i < args->size; ++i) {
        double sum_x = 0;
        double sum_y = 0;
        for (size_t j = 0; j < args->size; ++j) {
            if (i == j) continue;
            double distance = 
                std::sqrt(std::pow((Points[j].x - Points[i].x), 2) + std::pow((Points[j].y - Points[i].y), 2));
            sum_x += Points[j].m * (Points[j].x - Points[i].x) / std::pow(distance, 3);
            sum_y += Points[j].m * (Points[j].y - Points[i].y) / std::pow(distance, 3);
        }
        Result[i].first = args->G * Points[i].m * sum_x;
        Result[i].second = args->G * Points[i].m * sum_y;
    }
    return Result;
}

__device__ MatPoint* simulationStep(MatPoint* Points, Pair* Forces, Args* args) {
    for (size_t i = 0; i < args->size; ++i) {
        Points[i].vx += Forces[i].first / Points[i].m * args->dt;
        Points[i].vy += Forces[i].second / Points[i].m * args->dt;
        Points[i].x += Points[i].vx * args->dt;
        Points[i].y += Points[i].vy * args->dt;
    }
    return Points;
}


__global__ void simulationKernel(MatPoint* Points, Args* args, Pair* Result) {
    Pair* Forces = new Pair[args->size];
    Forces = calculateForce(Points, args, Result);
    Points = simulationStep(Points, Forces, args);
    delete[] Forces;
}


__host__ void readFromFile(std::ifstream& InputFile, std::vector<MatPoint>& Points) {
    double x, y, vx, vy, m;
    while (!InputFile.eof()) {
        InputFile >> x >> y >> vx >> vy >> m;
        Points.push_back({x, y, vx, vy, m});
    }
}

__host__ void printToFile(std::ofstream& File, double t, MatPoint* Points, size_t n, double ExecTime) {
    File << t << ",";
    for (size_t i = 0; i < n; ++i) {
        File << Points[i].x << "," << Points[i].y << ",";
    }
    File << ExecTime << "\n";
}

__host__ void fillMatPointsArr(MatPoint* Points, std::vector<MatPoint> PointsVec) {
    for (size_t i = 0; i < PointsVec.size(); ++i) {
        Points[i].x = PointsVec[i].x; 
        Points[i].y = PointsVec[i].y;
        Points[i].vx = PointsVec[i].vx;
        Points[i].vy = PointsVec[i].vy;
        Points[i].m = PointsVec[i].m;
    }
}

__host__ double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main() {
    std::string InputFilename = "input.txt";
    std::string OutputFilename = "output.csv";
    std::ifstream InputFile(InputFilename);

    std::vector<MatPoint> PointsVec;
    readFromFile(InputFile, PointsVec);
    size_t n = PointsVec.size();
    MatPoint* Points = new MatPoint[n];
    fillMatPointsArr(Points, PointsVec);
    Args* arguments = new Args;
    std::ofstream OutputFile(OutputFilename);
    OutputFile << "t,";
    for (size_t i = 0; i < n; ++i) {
        OutputFile << "x_" << i+1 << ",y_" << i+1 << ",";
    }
    OutputFile << "Time" << "\n";

    MatPoint* DevicePoints;
    Args* DeviceArgs;
    cudaMallocManaged(&DevicePoints, sizeof(MatPoint) * n);
    cudaMemcpy(DevicePoints, Points, n * sizeof(MatPoint), cudaMemcpyHostToDevice);
    cudaMallocManaged(&DeviceArgs, sizeof(Args));
    cudaMemcpy(DeviceArgs, arguments, sizeof(Args), cudaMemcpyHostToDevice);

    arguments->G = G; arguments->dt = dt; arguments->size = n;

    int numBlocks = 1, numThreadsPerBlock = 1;
    
    Pair* Result;
    cudaMallocManaged(&Result, sizeof(Pair) * n);

    for (double t = 0; t < Threshold; t += dt) {
        double StartTime = get_wall_time();
        simulationKernel<<<numBlocks, numThreadsPerBlock>>>(DevicePoints, DeviceArgs, Result);
        cudaDeviceSynchronize();
        std::cout << t << std::endl;
        cudaMemcpy(Points, DevicePoints, n * sizeof(MatPoint), cudaMemcpyDeviceToHost);
        double EndTime = get_wall_time();
        printToFile(OutputFile, t, Points, n, EndTime - StartTime);
    }

    cudaFree(&DeviceArgs);
    cudaFree(&DevicePoints);
    cudaFree(&Result);
    delete[] Points;
    delete arguments;

    return 0;
}