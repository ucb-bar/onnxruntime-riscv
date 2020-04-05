#pragma once

#include "cxxopts.hpp"

cxxopts::ParseResult parse(int argc, char* argv[]) {
  try {
    cxxopts::Options options("ONNX Model Runner", "\n");
    
    options.custom_help("[required args...] [optional args...]");

    options.add_options("Required")
        ("m,model", "ONNX model path", cxxopts::value<std::string>(), "[path]");

    options.add_options("Optional")
        ("k,kernel", "Use custom kernels",  cxxopts::value<bool>()->default_value("false"), "[output path]")
        ("t,trace", "Profiling trace output file", cxxopts::value<std::string>(), "[output path]")
        ("O,optimization", "Optimization level. NHWC transformation is applied at level 1.",
                            cxxopts::value<int>()->default_value("1"), "[0 (none) / 1 (basic) / 2 (extended) / 99 (all)]")
        ("d,debug", "Debug level", cxxopts::value<int>()->default_value("3"), "[0-5, with 0 being most verbose]")
        ("s,save_model", "Save transformed model to path", cxxopts::value<std::string>(), "[path]");

    options.add_options("Info")
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        printf("%s", options.help().c_str());
        exit(0);
    }

    return result;

  } catch (const cxxopts::OptionException& e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}
