#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <tuple>

std::vector<double> csvPricesParser(const std::string& fileName) {
    std::ifstream inputFile(fileName);
    if (!inputFile.is_open()) {
        std:: cerr << fileName + " could not open \n";
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

std::pair<double, double> computeMetrics(const std::vector<double>& prices) {
    //Compute log return i.e how much the price changed from one day to the next.
    std::vector<double> logReturn;
    for (int i = 1; i < prices.size(); i++) {
        logReturn.push_back(std::log(prices[i] / prices[i-1]));
    }
    
    //Compute mean/drift, i.e how much does this stock go up or down typically.
    double mean = std::accumulate(logReturn.begin(), logReturn.end(), 0.0) / logReturn.size();
    
    /*
    Compute variance and standard deviation.
    Variance measures how far daily returns deviate from the mean(squared).
    Standard deviation or volatility is the square root of variance, meant to show
    how much the stock usually swing on a given day.
    */
    double variance = 0.0;
    for (double r : logReturn) {
        variance += std::pow(r - mean, 2);
    }
    double stddev = std::sqrt(variance / (logReturn.size() - 1));
    return {mean, stddev};
}

std::tuple<double, double, double, double> runSimulations(double lastPrice, double mean, double stddev, int simulations, int days) {
    /*
    Returns the 5th percentile (bad scenario), the median (typical scenario),
    95th percentile (great scenario), and the average price.
    The model used is geometric Brownian motion.
    Note - The drift utilized is a constant computed from 10 years of historical data.
    It does not account for anything outside of the norms(good quarters, bad new hits, etc).
    */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> finalPrices;
    for (int s = 0; s < simulations; s++) {
        double price = lastPrice;
        for (int d = 0; d < days; d++) {
            double Z = dist(gen);
            price = price * std::exp((mean - 0.5 * stddev * stddev) + stddev * Z);
        }
        finalPrices.push_back(price);
    }
    std::sort(finalPrices.begin(), finalPrices.end());
    double median = finalPrices[simulations / 2];
    double percentile5  = finalPrices[simulations * 0.05];
    double percentile95 = finalPrices[simulations * 0.95];
    double avg = std::accumulate(finalPrices.begin(), finalPrices.end(), 0.0) / simulations;
    return {avg, median, percentile5, percentile95};
}

void printResults(double avg, double median, double percentile5th, double percentile95) {
    std::cout << "Average:    $" << avg    << "\n";
    std::cout << "Median:     $" << median << "\n";
    std::cout << "5th  pct:   $" << percentile5th    << "\n";
    std::cout << "95th pct:   $" << percentile95   << "\n";
}
int main() {
    auto prices = csvPricesParser("AMD Stock Price History.csv");
    auto [mean, stdDev] = computeMetrics(prices);
    auto [avg, median, percentile5, percentile95] = runSimulations(prices.back(), mean, stdDev, 10000, 30);
    printResults(avg, median, percentile5, percentile95);
}