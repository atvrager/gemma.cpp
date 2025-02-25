// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>

#include "compression/io.h"
#include "gemma/gemma.h"
#include "hwy/base.h"

#if (!defined(HWY_VERSION_LT) || HWY_VERSION_LT(1, 2)) && !HWY_IDE
#error "Please update to version 1.2 of github.com/google/highway."
#endif
#if HWY_CXX_LANG < 201703L
#error "Gemma.cpp requires C++17, please pass -std=c++17."
#endif

#if defined(__KELVIN__)
#include "image_ppm.h"
#endif

namespace gcpp {

void Run() {
    NestedPools pools(0);

    Path vitpath;
    vitpath.path = "vit.sbs";
    ModelInfo info;
    info.model = Model::PALIGEMMA_224_VIT;
    info.weight = Type::kSFP;
    Gemma model(Path(), vitpath, info, pools);

    Image image;
    ImageTokens image_tokens;
    image_tokens = ImageTokens(Extents2D(model.GetModelConfig().vit_seq_len, model.GetModelConfig().model_dim));
#if !defined(__KELVIN__)
    std::string image_filename = "image.ppm";
    HWY_ASSERT(image.ReadPPM(image_filename));
#else
  printf("%s:%d\n", __func__, __LINE__);
    HWY_ASSERT(image.ReadPPM(hwy::Span(reinterpret_cast<const char*>(image_ppm), image_ppm_len)));
  printf("%s:%d\n", __func__, __LINE__);
#endif
    RuntimeConfig runtime_config = {
      .verbosity = 1,
      .use_spinning = Tristate::kDefault,
    };
    model.GenerateImageTokens(runtime_config, image, image_tokens);

#if !defined(__KELVIN__)
    Path image_tokens_out;
    image_tokens_out.path = "image.tokens";
    if (!image_tokens_out.path.empty()) {
      auto image_tokens_out_file = gcpp::OpenFileOrNull(image_tokens_out, "w+");
      image_tokens_out_file->Write(image_tokens.All(), image_tokens.NumBytes(), 0);
      fprintf(stderr,
        "Wrote tokens to %s\n", image_tokens_out.path.c_str()
      );
    }
#endif
}

}  // namespace gcpp

int main(int argc, char** argv) {
    gcpp::Run();

    return 0;
}