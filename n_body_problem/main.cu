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

__device__ Pair* calculateForce(MatPoint* Points, Args* args) {
    Pair* Result = new Pair[args->size];
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


__global__ void simulationKernel(MatPoint* Points, Args* args) {
    Pair* Forces = new Pair[args->size];
    // printf("I'm after init Forces\n");
    Forces = calculateForce(Points, args);
    // printf("I'm after calculate Forces\n");
    Points = simulationStep(Points, Forces, args);
    // printf("I'm after simulation step\n");
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
    // // debug output
    // std::cout << t << ", ";
    // for (size_t i = 0; i < n; ++i) {
    //     std::cout << Points[i].x << ", " << Points[i].y << ", ";
    // }
    // std::cout << ExecTime << "\n";
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
    // MatPoint* Points = new MatPoint[n];
    // fillMatPointsArr(Points, PointsVec);
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
    cudaMallocManaged(&DeviceArgs, sizeof(Args));
    fillMatPointsArr(DevicePoints, PointsVec);
    DeviceArgs->G = G; DeviceArgs->dt = dt; DeviceArgs->size = n;

    // arguments->G = G; arguments->dt = dt; arguments->size = n;
    // cudaMemcpy(DevicePoints, Points, sizeof(Points), cudaMemcpyHostToDevice);
    // cudaMemcpy(DeviceArgs, arguments, sizeof(arguments), cudaMemcpyHostToDevice);

    int numBlocks = 1, numThreadsPerBlock = 1;

    for (double t = 0; t < Threshold; t += dt) {
        double StartTime = get_wall_time();
        simulationKernel<<<numBlocks, numThreadsPerBlock>>>(DevicePoints, DeviceArgs);
        cudaDeviceSynchronize();
        // cudaMemcpy(Points, DevicePoints, sizeof(DevicePoints), cudaMemcpyDeviceToHost);
        double EndTime = get_wall_time();
        // file output
        // debug output
        if (!(int(t * 1000)%1000))
            std::cout << t << "\n";
        // for (size_t i = 0; i < n; ++i) {
        //     std::cout << DevicePoints[i].x << ", " << DevicePoints[i].y << ", ";
        // }
        // std::cout << "\n";
        // std::cout << EndTime - StartTime << "\n";
        printToFile(OutputFile, t, DevicePoints, n, EndTime - StartTime);
        // printToFile(OutputFile, t, DevicePoints, n, 0);
    }

    cudaFree(&DeviceArgs);
    cudaFree(&DevicePoints);
    // delete[] Points;
    // delete arguments;

    return 0;
}