// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_OPTIONS_H
#define PBRT_OPTIONS_H

#include <pbrt/pbrt.h>
#include <pbrt/util/log.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// RenderingCoordinateSystem Definition
enum class RenderingCoordinateSystem { Camera, CameraWorld, World };
std::string ToString(const RenderingCoordinateSystem &);

// BasicPBRTOptions Definition
// used in both the cpu and gpu render pipeline
struct BasicPBRTOptions {
    // any time an RNG is initialized in pbrt, the seed value in the options
    //   should be incorporated in the seed passed to its constructor
    // In this way, the renderer will generate independent images
    //   if the user specifies different –seed values using command-line arguments
    int seed = 0;
    bool quiet = false;
    bool disablePixelJitter = false, disableWavelengthJitter = false;
    bool disableTextureFiltering = false;
    bool disableImageTextures = false;
    bool forceDiffuse = false;
    bool useGPU = false;
    bool wavefront = false;
    bool interactive = false;
    bool fullscreen = false;
    RenderingCoordinateSystem renderingSpace = RenderingCoordinateSystem::CameraWorld;
};

// PBRTOptions Definition
// a number of additional options
//   that are mostly used when processing the scene description
//   and not during rendering
// Mainly cpu-side parameter options, since gpu can't access std::string variables
struct PBRTOptions : BasicPBRTOptions {
    int nThreads = 0;
    LogLevel logLevel = LogLevel::Error;
    std::string logFile;
    bool logUtilization = false;
    bool writePartialImages = false;
    bool recordPixelStatistics = false;
    bool printStatistics = false;
    pstd::optional<int> pixelSamples;
    pstd::optional<int> gpuDevice;
    bool quickRender = false;
    bool upgrade = false;
    std::string imageFile;
    std::string mseReferenceImage, mseReferenceOutput;
    std::string debugStart;
    std::string displayServer;
    pstd::optional<Bounds2f> cropWindow;
    pstd::optional<Bounds2i> pixelBounds;
    pstd::optional<Point2i> pixelMaterial;
    Float displacementEdgeScale = 1;

    std::string ToString() const;
};

// Options Global Variable Declaration
// In code that only runs on the CPU, the options can be accessed via this global variable
extern PBRTOptions *Options;

#if defined(PBRT_BUILD_GPU_RENDERER)
#if defined(__CUDACC__)
extern __constant__ BasicPBRTOptions OptionsGPU;
#endif  // __CUDACC__

void CopyOptionsToGPU();
#endif  // PBRT_BUILD_GPU_RENDERER

// Options Inline Functions
// For code that runs on both the CPU and GPU
//   options must be accessed through the GetOptions() function
//   which returns a copy of the options that is either stored in CPU or GPU memory
//   depending on which type of processor the code is executing
PBRT_CPU_GPU inline const BasicPBRTOptions &GetOptions();

PBRT_CPU_GPU inline const BasicPBRTOptions &GetOptions() {
#if defined(PBRT_IS_GPU_CODE)
    return OptionsGPU;
#else
    return *Options;
#endif
}

}  // namespace pbrt

#endif  // PBRT_OPTIONS_H
