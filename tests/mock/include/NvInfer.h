// Fake TensorRT, only for testing the harness logic without a GPU (tests/mock).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
// -DMOCK_TRT_LEGACY simulates TensorRT 8.2 (JetPack 4): bindings API and enqueueV2
#ifdef MOCK_TRT_LEGACY
#define NV_TENSORRT_MAJOR 8
#define NV_TENSORRT_MINOR 2
#else
#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 3
#endif
namespace nvinfer1 {
enum class DataType : int32_t { kFLOAT = 0, kHALF = 1, kINT8 = 2, kINT32 = 3 };
enum class TensorIOMode : int32_t { kNONE = 0, kINPUT = 1, kOUTPUT = 2 };
struct Dims { int32_t nbDims; int64_t d[8]; };
class ILogger {
 public:
  enum class Severity : int32_t { kINTERNAL_ERROR = 0, kERROR = 1, kWARNING = 2, kINFO = 3, kVERBOSE = 4 };
  virtual void log(Severity s, const char* msg) noexcept = 0;
  virtual ~ILogger() = default;
};
class IExecutionContext {
 public:
  void* in = nullptr; void* out = nullptr;
#ifdef MOCK_TRT_LEGACY
  bool enqueueV2(void* const* bindings, cudaStream_t s, void* ev);
#else
  bool setTensorAddress(const char* name, void* p);
  bool enqueueV3(cudaStream_t s);
#endif
  bool run_mock();
};
class ICudaEngine {
 public:
#ifdef MOCK_TRT_LEGACY
  int32_t getNbBindings() const { return 2; }
  const char* getBindingName(int32_t i) const { return i == 0 ? "images" : "output"; }
  bool bindingIsInput(int32_t i) const { return i == 0; }
  Dims getBindingDimensions(int32_t i) const { return getTensorShape(getBindingName(i)); }
  DataType getBindingDataType(int32_t i) const { return getTensorDataType(getBindingName(i)); }
#else
  int32_t getNbIOTensors() const { return 2; }
  const char* getIOTensorName(int32_t i) const { return i == 0 ? "images" : "output"; }
  TensorIOMode getTensorIOMode(const char* n) const;
#endif
  Dims getTensorShape(const char* n) const;
  DataType getTensorDataType(const char*) const;
  IExecutionContext* createExecutionContext() { return new IExecutionContext(); }
};
class IRuntime {
 public:
  ICudaEngine* deserializeCudaEngine(const void*, size_t) { return new ICudaEngine(); }
};
IRuntime* createInferRuntime(ILogger&);
}  // namespace nvinfer1
