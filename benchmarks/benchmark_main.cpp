#include "kernel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

struct Options {
  std::string mode = "coherent";
  int objects = 10000;
  int warmup = 20;
  int samples = 10;
  int steps = 20;
  std::string output = "benchmark.csv";
};

void printUsage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --mode naive|scattered|coherent  update path (default: coherent)\n"
      << "  --objects N                       number of boids (default: 10000)\n"
      << "  --warmup N                        untimed steps (default: 20)\n"
      << "  --samples N                       timed samples (default: 10)\n"
      << "  --steps N                         steps per sample (default: 20)\n"
      << "  --output PATH                     CSV destination (default: benchmark.csv)\n"
      << "  --help                            show this help\n";
}

bool parsePositiveInt(const char *text, int &value) {
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

bool parseOptions(int argc, char **argv, Options &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--help") {
      printUsage(argv[0]);
      return false;
    }
    if (argument == "--mode" || argument == "--objects" ||
        argument == "--warmup" || argument == "--samples" ||
        argument == "--steps" || argument == "--output") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << argument << "\n";
        return false;
      }
      const std::string value(argv[++i]);
      if (argument == "--mode") {
        options.mode = value;
      } else if (argument == "--output") {
        options.output = value;
      } else if (argument == "--objects") {
        if (!parsePositiveInt(value.c_str(), options.objects)) {
          std::cerr << "Invalid --objects value: " << value << "\n";
          return false;
        }
      } else if (argument == "--warmup") {
        if (!parsePositiveInt(value.c_str(), options.warmup)) {
          std::cerr << "Invalid --warmup value: " << value << "\n";
          return false;
        }
      } else if (argument == "--samples") {
        if (!parsePositiveInt(value.c_str(), options.samples)) {
          std::cerr << "Invalid --samples value: " << value << "\n";
          return false;
        }
      } else if (argument == "--steps") {
        if (!parsePositiveInt(value.c_str(), options.steps)) {
          std::cerr << "Invalid --steps value: " << value << "\n";
          return false;
        }
      }
      continue;
    }
    std::cerr << "Unknown argument: " << argument << "\n";
    return false;
  }

  if (options.mode != "naive" && options.mode != "scattered" &&
      options.mode != "coherent") {
    std::cerr << "--mode must be naive, scattered, or coherent\n";
    return false;
  }
  if (options.objects <= 0 || options.samples <= 0 || options.steps <= 0) {
    std::cerr << "--objects, --samples, and --steps must be greater than zero\n";
    return false;
  }
  return true;
}

void step(const std::string &mode) {
  constexpr float dt = 0.2f;
  if (mode == "naive") {
    Boids::stepSimulationNaive(dt);
  } else if (mode == "scattered") {
    Boids::stepSimulationScatteredGrid(dt);
  } else {
    Boids::stepSimulationCoherentGrid(dt);
  }
}

bool checkCuda(const char *operation) {
  const cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(error) << "\n";
  return false;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parseOptions(argc, argv, options)) {
    return argc > 1 && std::string(argv[1]) == "--help" ? 0 : 2;
  }

  cudaDeviceProp device{};
  if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) {
    std::cerr << "Could not query CUDA device 0\n";
    return 1;
  }

  std::ofstream csv(options.output, std::ios::out | std::ios::trunc);
  if (!csv) {
    std::cerr << "Could not open output CSV: " << options.output << "\n";
    return 1;
  }
  csv << "mode,objects,warmup_steps,samples,steps_per_sample,sample,"
         "elapsed_ms,step_ms,steps_per_second,gpu\n";

  std::cout << "GPU: " << device.name << "\n"
            << "mode=" << options.mode << " objects=" << options.objects
            << " warmup=" << options.warmup << " samples=" << options.samples
            << " steps_per_sample=" << options.steps << "\n";

  Boids::initSimulation(options.objects);
  if (!checkCuda("Boids::initSimulation")) {
    Boids::endSimulation();
    return 1;
  }

  for (int i = 0; i < options.warmup; ++i) {
    step(options.mode);
  }
  if (cudaDeviceSynchronize() != cudaSuccess ||
      !checkCuda("warmup") ) {
    Boids::endSimulation();
    return 1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess ||
      cudaEventCreate(&stop) != cudaSuccess) {
    std::cerr << "Could not create CUDA timing events\n";
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    Boids::endSimulation();
    return 1;
  }

  csv << std::setprecision(8);
  bool measurementSucceeded = true;
  for (int sample = 0; sample < options.samples; ++sample) {
    if (cudaEventRecord(start) != cudaSuccess) {
      std::cerr << "cudaEventRecord(start) failed\n";
      measurementSucceeded = false;
      break;
    }
    for (int i = 0; i < options.steps; ++i) {
      step(options.mode);
    }
    if (cudaEventRecord(stop) != cudaSuccess ||
        cudaEventSynchronize(stop) != cudaSuccess) {
      std::cerr << "CUDA timing event synchronization failed\n";
      measurementSucceeded = false;
      break;
    }

    float elapsedMs = 0.0f;
    if (cudaEventElapsedTime(&elapsedMs, start, stop) != cudaSuccess) {
      std::cerr << "cudaEventElapsedTime failed\n";
      measurementSucceeded = false;
      break;
    }
    const double stepMs = elapsedMs / options.steps;
    const double stepsPerSecond = 1000.0 / stepMs;
    csv << options.mode << ',' << options.objects << ',' << options.warmup
        << ',' << options.samples << ',' << options.steps << ',' << sample
        << ',' << elapsedMs << ',' << stepMs << ',' << stepsPerSecond
        << ",\"" << device.name << "\"\n";
    std::cout << "sample " << sample << ": " << stepMs
              << " ms/step (" << stepsPerSecond << " steps/s)\n";
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  Boids::endSimulation();
  csv.close();
  return measurementSucceeded ? 0 : 1;
}
