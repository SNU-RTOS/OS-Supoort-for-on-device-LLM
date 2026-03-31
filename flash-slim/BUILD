# Copyright 2024 OS-Support-for-On-Device-LLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@litert//tflite:build_def.bzl", "tflite_linkopts")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        ":utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/experimental/genai:genai_ops",
        "@litert//tflite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "aligned_allocator",
    srcs = ["aligned_allocator.h"],
    deps = [
        "@litert//tflite:framework",
        "@litert//tflite:util",
    ],
)

cc_library(
    name = "lora_adapter",
    srcs = [
        "lora_adapter.cc",
    ],
    hdrs = [
        "lora_adapter.h",
    ],
    deps = [
        ":aligned_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/experimental/genai:genai_ops",
        "@litert//tflite/kernels:builtin_ops",
    ],
)


cc_library(
    name = "weight_chunk_prefetcher",
    srcs = [
        "weight_chunk_io_engine.cc",
        "weight_chunk_prefetcher.cc",
    ],
    hdrs = [
        "weight_chunk_io_engine.h",
        "weight_chunk_prefetcher.h",
    ],
    deps = [
        "@litert//tflite/delegates/xnnpack:streaming_weight_cache",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:linux": ["-luring"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "mutable_tensor_tracker",
    srcs = ["mutable_tensor_tracker.cc"],
    hdrs = ["mutable_tensor_tracker.h"],
    deps = [
        "@litert//tflite:framework",
    ],
)

cc_library(
    name = "prefetch_planner",
    srcs = ["prefetch_planner.cc"],
    hdrs = ["prefetch_planner.h"],
    deps = [
        ":weight_chunk_prefetcher",
    ],
)

cc_library(
    name = "weight_chunk_controller_utils",
    srcs = ["weight_chunk_controller_utils.cc"],
    hdrs = ["weight_chunk_controller_utils.h"],
    deps = [
        ":mutable_tensor_tracker",
        ":prefetch_planner",
        ":weight_chunk_prefetcher",
        "//flash-slim/nlohmann:nlohmann_json",
        "@litert//tflite/delegates/xnnpack:streaming_weight_cache",
    ],
)

cc_library(
    name = "weight_chunk_controller",
    srcs = ["weight_chunk_controller.cc"],
    hdrs = [
        "weight_chunk_controller.h",
    ],
    deps = [
        ":weight_chunk_controller_utils",
        ":weight_chunk_prefetcher",
        "@litert//tflite/delegates/xnnpack:streaming_weight_cache",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        ":aligned_allocator",
        ":lora_adapter",
        ":profiler",
        ":sampler",
        ":utils",
        ":weight_chunk_controller",
        ":weight_chunk_controller_utils",
        ":weight_chunk_prefetcher",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_sentencepiece//:sentencepiece_processor",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/delegates/xnnpack:streaming_weight_cache",
        "@litert//tflite/delegates/xnnpack:weight_cache",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/experimental/genai:genai_ops",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/profiling:profile_summarizer",
        "@litert//tflite/profiling:profiler",
        "@litert//tflite/tools/benchmark:benchmark_model_lib",
    ],
)

cc_binary(
    name = "text_generator_main",
    srcs = [
        "text_generator_main.cc",
    ],
    defines = [
        "TF_LITE_TENSORFLOW_PROFILER",
    ],
    linkopts = tflite_linkopts(),
    deps = [
        ":common",
    ],
)

cc_binary(
    name = "text_generator_mmap",
    srcs = [
        "text_generator_main.cc",
    ],
    defines = [
        "TF_LITE_TENSORFLOW_PROFILER",
    ],
    linkopts = tflite_linkopts(),
    deps = [
        ":common",
    ],
)

cc_binary(
    name = "cmt_generator",
    srcs = [
        "cmt_generator.cc",
        "cmt_generator.h",
    ],
    defines = [
        "TF_LITE_TENSORFLOW_PROFILER",
    ],
    linkopts = tflite_linkopts(),
    deps = [
        ":common",
        ":mutable_tensor_tracker",
        ":prefetch_planner",
    ],
)

cc_binary(
    name = "inference_driver",
    srcs = ["inference_driver.cc"],
    copts = [
        "-std=c++17",
        "-DUSE_OPENCV",
        "-I/usr/include/opencv4",
    ],
    linkopts = tflite_linkopts() + [
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-ldl",
        "-lpthread",
    ],
    deps = [
        ":common",
    ],
)