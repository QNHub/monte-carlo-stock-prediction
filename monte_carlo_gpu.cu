// Stochastic Equity Price Forecaster GPU VER
// Parallelizes Monte Carlo simulations across GPU core to benchmark against CPU
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <random>
#include <curand_kernel.h>

// Each thread runs one independent Monte Carlo simulation on the GPU.
__global__ void monteCarloKernel(
    double* finalPricesGPU,
    double lastPrice,
    double drift,
    double stddev,
    int days,
    int simulations,
    unsigned long long seed
) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex >= simulations) {
        return;
    }
    curandState state;
    curand_init(seed, workIndex, 0, &state);
    double price = lastPrice;
    for (int d = 0; d < days; d++) {
        double Z = curand_normal_double(&state);
        price = price * exp(drift + stddev * Z);
    }
    finalPricesGPU[workIndex] = price;
}

// Returns prices by parsing through CSV.
std::vector<double> parsePricesFromCSV(const std::string& fileName) {
    std::ifstream inputFile(fileName);
    if (!inputFile.is_open()) {
        std::cerr << fileName + " could not open \n";
        exit(1);
    }
    std::string line;
    std::string header;
    std::getline(inputFile, header);
    std::vector<double> prices;
    while (std::getline(inputFile, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        while (std::getline(lineStream, cell, ',')) {
            parsedRow.push_back(cell);
        }
        if (parsedRow.size() < 2) {
            continue;
        }
        if (parsedRow[1].empty()) {
            continue;
        }
        //Strip quotation marks from the price string.
        parsedRow[1].erase(std::remove(parsedRow[1].begin(), parsedRow[1].end(), '"'), parsedRow[1].end());
        double price = std::stod(parsedRow[1]);
        prices.push_back(price);
    }
    inputFile.close();
    //Reverse is necessary as the current format is latest to oldest, but oldest to latest is necessary.
    std::reverse(prices.begin(), prices.end());
    return prices;
}

// Computes the mean for drift and volatility(standard deviation).
std::pair<double, double> computeMetrics(const std::vector<double>& prices) {
    std::vector<double> logReturn;
    for (int i = 1; i < prices.size(); i++) {
        logReturn.push_back(std::log(prices[i] / prices[i-1]));
    }
    double mean = std::accumulate(logReturn.begin(), logReturn.end(), 0.0) / logReturn.size();
    double variance = 0.0;
    for (double r : logReturn) {
        variance += std::pow(r - mean, 2);
    }
    double stddev = std::sqrt(variance / (logReturn.size() - 1));
    double drift = mean - 0.5 * stddev * stddev;
    return {drift, stddev};
}

// Runs simulations on both GPU and CPU, then compares performance.
std::tuple<double, double, double, double> runSimulations(double lastPrice, double drift, double stddev, int simulations, int days) {
    double* finalPricesGPU;
    cudaMalloc(&finalPricesGPU, simulations * sizeof(double));
    int threadsPerBlock = 256;
    int blocks = (simulations + threadsPerBlock - 1) / threadsPerBlock;
    auto gpuStart = std::chrono::high_resolution_clock::now();
    monteCarloKernel<<<blocks, threadsPerBlock>>>(finalPricesGPU, lastPrice, drift, stddev, days, simulations, 0ULL);
    cudaDeviceSynchronize();

    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();

    std::vector<double> finalPrices(simulations);
    cudaMemcpy(finalPrices.data(), finalPricesGPU, simulations * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(finalPricesGPU);
    
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> cpuPrices;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < simulations; s++) {
        double price = lastPrice;
        for (int d = 0; d < days; d++) {
            double Z = dist(gen);
            price = price * std::exp(drift + stddev * Z);
        }
        cpuPrices.push_back(price);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    std::sort(finalPrices.begin(), finalPrices.end());
    double median = finalPrices[simulations / 2];
    double percentile5  = finalPrices[simulations * 0.05];
    double percentile95 = finalPrices[simulations * 0.95];
    double avg = std::accumulate(finalPrices.begin(), finalPrices.end(), 0.0) / simulations;

    std::cout << "GPU time:  " << gpuMs << " ms\n";
    std::cout << "CPU time:  " << cpuMs << " ms\n";
    std::cout << "Speedup:   " << cpuMs / gpuMs << "x\n";
    return {avg, median, percentile5, percentile95};
}

void printResults(double avg, double median, double percentile5th, double percentile95) {
    std::cout << "Average:    $" << avg    << "\n";
    std::cout << "Median:     $" << median << "\n";
    std::cout << "5th  pct:   $" << percentile5th    << "\n";
    std::cout << "95th pct:   $" << percentile95   << "\n";
}
int main(int argc, char* argv[]) {
    if (argc > 4 || argc < 2) {
        std::cout << "Invalid argument";
        return 1;
    }
    std::string fileName = argv[1];
    int simulationCount = std::stoi(argv[2]);
    int days = std::stoi(argv[3]);
    auto prices = parsePricesFromCSV(fileName);
    auto [drift, stddev] = computeMetrics(prices);
    auto [avg, median, percentile5, percentile95] = runSimulations(prices.back(), drift, stddev, simulationCount, days);
    printResults(avg, median, percentile5, percentile95);
}