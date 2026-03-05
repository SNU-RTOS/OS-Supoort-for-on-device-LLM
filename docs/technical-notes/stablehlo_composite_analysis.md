# STABLEHLO_COMPOSITE 분석

- 이 문서는 TensorFlow Lite에서 사용되는 `STABLEHLO_COMPOSITE` 연산자의 동작 방식과 내부적으로 다른 subgraph를 어떻게 호출하는지를 설명한다. 실제 동작 코드를 기반으로 분석함.

- reference: LiteRT/tflite/kernels/stablehlo_composite.cc

```cpp
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tflite/c/common.h"
#include "tflite/context_util.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/subgraph.h"
#include "tflite/kernels/control_flow_common.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_composite {

struct State {
  int32_t subgraph_index;
  bool subgraph_has_dynamic_output_tensors = false;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  auto data = std::make_unique<State>();
  const TfLiteStablehloCompositeParams* params =
      reinterpret_cast<const TfLiteStablehloCompositeParams*>(options);
  data->subgraph_index = params->subgraph_index;
  return data.release();
}

void Free(TfLiteContext* context, void* node_data) {
  delete static_cast<State*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  State* op_state = reinterpret_cast<State*>(node->user_data);

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  const auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE(context, op_state->subgraph_index < subgraphs->size());

  Subgraph* decomposition_subgraph =
      (*subgraphs)[op_state->subgraph_index].get();

  TF_LITE_ENSURE_EQ(context, num_inputs,
                    decomposition_subgraph->inputs().size());
  TF_LITE_ENSURE_EQ(context, num_outputs,
                    decomposition_subgraph->outputs().size());

  // Remove unused inputs of subgraph to skip copying unnecessary inputs.
  decomposition_subgraph->RemoveUnusedInputs();

  std::vector<int> node_inputs(node->inputs->data,
                               node->inputs->data + num_inputs);

  // Prepare and check the subgraphs.
  TF_LITE_ENSURE_OK(context,
                    CopyTensorsShapeAndType(context, this_subgraph, node_inputs,
                                            decomposition_subgraph,
                                            decomposition_subgraph->inputs(),
                                            /*resize_subgraph_inputs=*/true));

  // Handle resource input tensors.
  for (int i = 0; i < num_inputs; ++i) {
    int input_idx = decomposition_subgraph->inputs()[i];
    if (input_idx == kTfLiteOptionalTensor) {
      continue;
    }
    TfLiteTensor* subgraph_input = decomposition_subgraph->tensor(input_idx);
    if (!IsResourceOrVariant(subgraph_input)) {
      // Set the allocation type to custom to prevent memory allocation.
      subgraph_input->allocation_type = kTfLiteCustom;
    }
  }

  // Allocate the memory for the subgraph.
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());
  op_state->subgraph_has_dynamic_output_tensors |=
      decomposition_subgraph->HasDynamicTensors();

  for (int i = 0; i < num_outputs; ++i) {
    if (node->outputs->data[i] == kTfLiteOptionalTensor) {
      continue;
    }
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    if (op_state->subgraph_has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      TfLiteTensor* subgraph_output =
          decomposition_subgraph->tensor(decomposition_subgraph->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(subgraph_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }
  return kTfLiteOk;
}

// Evaluate the COMPOSITE op when the subgraph has dynamic outputs.
TfLiteStatus Eval_dynamic(TfLiteContext* context, TfLiteNode* node,
                          Subgraph* this_subgraph,
                          Subgraph* decomposition_subgraph) {
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());
  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data;
  std::vector<int> node_inputs(start, start + num_inputs);
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   decomposition_subgraph, decomposition_subgraph->inputs()));

  // Invoke decomposition_subgraph subgraph
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->Invoke());
  for (int tensor_index : decomposition_subgraph->outputs()) {
    decomposition_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  // subgraph->outputs -> node->outputs
  TF_LITE_ENSURE_OK(context,
                    DeepCopyTensorsShapeTypeData(
                        context, node, decomposition_subgraph,
                        decomposition_subgraph->outputs(), this_subgraph,
                        TfLiteIntArrayView(node->outputs), true));

  for (int i = 0; i < num_outputs; ++i) {
    const int input_pos = OutputIsInput(decomposition_subgraph->outputs()[i],
                                        decomposition_subgraph->inputs());
    if (input_pos != -1) {
      TfLiteTensor* this_input =
          this_subgraph->tensor(node->inputs->data[input_pos]);
      TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
      TfLiteTensorCopy(this_input, this_output);
    }
  }
  return kTfLiteOk;
}

// Evaluate the COMPOSITE op when the subgraph has static outputs.
TfLiteStatus Eval_static(TfLiteContext* context, TfLiteNode* node,
                         Subgraph* this_subgraph,
                         Subgraph* decomposition_subgraph) {
  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data;
  std::vector<int> node_inputs(start, start + num_inputs);
  for (int i = 0; i < num_outputs; ++i) {
    int output_idx = decomposition_subgraph->outputs()[i];
    if (output_idx == kTfLiteOptionalTensor) continue;
    TfLiteTensor* subgraph_output = decomposition_subgraph->tensor(output_idx);
    if (!IsResourceOrVariant(subgraph_output) &&
        !IsConstantTensor(subgraph_output)) {
      subgraph_output->allocation_type = kTfLiteCustom;
    }
  }
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   decomposition_subgraph, decomposition_subgraph->inputs()));

  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsShapeAndType(context, decomposition_subgraph,
                              decomposition_subgraph->outputs(), this_subgraph,
                              TfLiteIntArrayView(node->outputs), false));
  for (int i = 0; i < num_outputs; ++i) {
    TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
    TfLiteTensor* subgraph_output =
        decomposition_subgraph->tensor(decomposition_subgraph->outputs()[i]);
    if (decomposition_subgraph->outputs()[i] == kTfLiteOptionalTensor) {
      TfLiteTensor* this_input = this_subgraph->tensor(node->inputs->data[i]);
      TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
      TfLiteTensorCopy(this_input, this_output);
    } else {
      const int input_pos = OutputIsInput(decomposition_subgraph->outputs()[i],
                                          decomposition_subgraph->inputs());
      if (input_pos != -1) {
        TfLiteTensor* this_input =
            this_subgraph->tensor(node->inputs->data[input_pos]);
        TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
        TfLiteTensorCopy(this_input, this_output);
      } else if (IsConstantTensor(subgraph_output)) {
        TfLiteTensorCopy(subgraph_output, this_output);
      } else {
        subgraph_output->data = this_output->data;
      }
    }
  }

  // Invoke subgraph
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->Invoke());
  for (int tensor_index : decomposition_subgraph->outputs()) {
    decomposition_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  State* op_state = reinterpret_cast<State*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* decomposition_subgraph =
      (*subgraphs)[op_state->subgraph_index].get();

  if (op_state->subgraph_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_OK(context, Eval_dynamic(context, node, this_subgraph,
                                            decomposition_subgraph));
  } else {
    TF_LITE_ENSURE_OK(context, Eval_static(context, node, this_subgraph,
                                           decomposition_subgraph));
  }

  if (!this_subgraph->ShouldPreserveAllTensors()) {
    TF_LITE_ENSURE_OK(context, decomposition_subgraph->ReleaseMemory());
  }

  return kTfLiteOk;
}

}  // namespace stablehlo_composite

TfLiteRegistration* Register_STABLEHLO_COMPOSITE() {
  static TfLiteRegistration r = {/*.init=*/stablehlo_composite::Init,
                                 /*.free=*/stablehlo_composite::Free,
                                 /*.prepare=*/stablehlo_composite::Prepare,
                                 /*.invoke=*/stablehlo_composite::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

```

---

## 1. 주요 기능 개요

`STABLEHLO_COMPOSITE`는 내부적으로 다른 subgraph를 호출하는 Composite Operator이다. 각 호출 연산자 인스턴스는 `TfLiteStablehloCompositeParams`를 통해 어떤 subgraph를 실행할지 결정한다. 이 구조는 StableHLO에서 복잡한 논리를 하나의 연산으로 감싸고, TFLite의 `Subgraph`로 위임하는 구조다.

---

## 2. 주요 구조체와 함수 분석

### 2.1 `State` 구조체

```cpp
struct State {
  int32_t subgraph_index;
  bool subgraph_has_dynamic_output_tensors = false;
};
```

- 이 operator의 context를 담는 구조체.
- 어떤 subgraph를 호출할지 인덱스를 저장하고, output tensor가 dynamic한지 flag를 가짐.

### 2.2 Init()

```cpp
void* Init(TfLiteContext* context, const char* options, size_t options_len)
```

- FlatBuffer로부터 전달된 `TfLiteStablehloCompositeParams`를 해석하여 `subgraph_index`를 가져옴.
- 동적 할당한 `State`를 반환함으로써 이후 Prepare, Eval에서 사용 가능하도록 유지.

### 2.3 Free()

```cpp
void Free(TfLiteContext* context, void* node_data)
```

- Init에서 생성한 State 구조체를 해제.

### 2.4 Prepare()

```cpp
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node)
```

- 현재 subgraph로부터 전체 subgraph 리스트를 획득하고, 지정된 `subgraph_index`로 target subgraph 획득.
- Input/output 수 확인 및 일치성 검사.
- 필요시 target subgraph의 input shape/type을 resize하고, tensor를 allocate.
- output 중 dynamic tensor 여부를 판단해 flag를 설정.

---

## 3. Eval() 경로

```cpp
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node)
```

- State에 설정된 flag에 따라 `Eval_dynamic` 또는 `Eval_static`을 호출.
- 호출된 subgraph는 내부적으로 `subgraph->Invoke()`에 의해 실행.
- 실행 완료 후, output tensor를 원래 subgraph로 복사.
- `ShouldPreserveAllTensors()`가 false인 경우 subgraph 메모리는 해제.

### 3.1 Eval_dynamic()

- output shape이 dynamic한 경우 사용.
- input을 복사 후 `AllocateTensors()` 호출.
- Invoke() 수행 후 output을 deep copy해서 원래 subgraph로 전달.
- 일부 output은 input과 동일한 경우 input 데이터를 그대로 복사.

### 3.2 Eval_static()

- output shape이 static한 경우 사용.
- input/output tensor는 shallow copy로 memory pointer 공유도 가능.
- subgraph output이 constant인 경우에도 tensor 복사를 처리함.

---

## 4. 결론

- `STABLEHLO_COMPOSITE`는 delegate가 아닌 **operator-level composite 연산자**로, 내부적으로 subgraph 호출을 통해 연산 위임.
- 호출 시점에서 subgraph의 `Invoke()`가 실행되며, 이후 텐서 복사 또는 공유 방식으로 결과를 원래 subgraph에 반영.
- 하나의 subgraph 내부에서 nested subgraph가 재귀적으로 호출될 수도 있으며, 이는 call stack 기반 프로파일링 또는 로깅으로 추적 가능.
