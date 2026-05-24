# Stochastic Equity Price Forecaster
 
A Monte Carlo stock price simulator that models future price trajectories using geometric Brownian motion. Includes both a single-threaded CPU implementation and a CUDA GPU-parallelized implementation, achieving a **49x speedup** at 1,000,000 simulations.
 
## How It Works
 
The simulator computes drift and volatility from historical closing prices, then runs thousands of independent price path simulations over a configurable time horizon. Each simulation applies daily random shocks drawn from a normal distribution, scaled by the stock's historical volatility. The result is a probability distribution of future prices with confidence intervals.
 
The core formula (geometric Brownian motion):
 
```
S(t+1) = S(t) × exp((μ - σ²/2) + σ × Z)
```
 
- `μ` — mean daily log return (drift)
- `σ` — standard deviation of daily log returns (volatility)
- `Z` — random sample from standard normal distribution
## Benchmark Results
 
Tested with 1,000,000 simulations over 30 days on an NVIDIA T4 GPU (Google Colab):
 
| Metric | CPU | GPU |
|--------|-----|-----|
| Time | 1,748 ms | 35 ms |
| Speedup | 1x | **49x** |
| Price calculations | 30,000,000 | 30,000,000 |
 
The GPU version assigns one simulation per CUDA thread, with each thread using its own cuRAND state for independent random number generation. At 1,000,000 simulations, the parallel execution fully saturates the GPU cores and dominates the kernel launch overhead.
 
## Build
 
**CPU only** (any system with g++):
```bash
make cpu
```
 
**GPU version** (requires NVIDIA GPU + CUDA Toolkit):
```bash
make gpu
```
 
## Usage
 
```bash
# CPU — default settings (10,000 simulations, 30 days)
./monte_carlo_cpu
 
# CPU — custom CSV, 100K simulations, 60 day forecast
./monte_carlo_cpu "AMD Stock Price History.csv" 100000 60
 
# GPU — 1M simulations for full speedup
./monte_carlo_gpu "AMD Stock Price History.csv" 1000000 30
```
 
## Sample Output
<img width="1329" height="169" alt="Screenshot 2026-05-24 at 7 33 18 PM" src="https://github.com/user-attachments/assets/1fb9511f-a31f-4741-9bb6-b06b775d2673" />


 
## CSV Format
 
Input CSV should have a header row with the closing price in the second column:
 
```csv
"Date","Price","Open","High","Low","Vol.","Change %"
"05/22/2026","467.51","469.84","481.41","461.71","34.76M","3.99%"
```
 
## Project Structure
 
```
├── monte_carlo_cpu.cpp    # Single-threaded C++ implementation
├── monte_carlo_gpu.cu     # CUDA GPU-parallelized implementation
├── Makefile               # Build targets for CPU and GPU
├── AMD Stock Price History.csv
└── README.md
```
 
## Technologies
 
- **C++17** — core language for both implementations
- **CUDA** — GPU parallelization across 2,560 cores
- **cuRAND** — GPU-native random number generation
- **Geometric Brownian Motion** — stochastic price modeling
