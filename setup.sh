#!/bin/bash

source .env

########## Setup env ##########
ROOT_PATH=$(pwd)
AI_EDGE_TORCH_PATH=${ROOT_PATH}/${AI_EDGE_TORCH_PATH}
TENSORFLOW_PATH=${ROOT_PATH}/${TENSORFLOW_PATH}
TENSORFLOW_COMMIT_HASH=117a62ac439ed87eb26f67208be60e01c21960de

LLM_APP_SRC=${ROOT_PATH}/src
LLM_APP_BINARY_NAME=text_generator_main
LLM_APP_BINARY_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/ai_edge_torch/generative/examples/cpp/${LLM_APP_BINARY_NAME}

echo "[INFO] AI_EDGE_TORCH_PATH: ${AI_EDGE_TORCH_PATH}"
echo "[INFO] TENSORFLOW_PATH: ${TENSORFLOW_PATH}"

if [ ! -d "./external" ]; then
    mkdir -p ./external
fi

########## Setup external sources ##########
cd external

echo "[INFO] Installing ai-edge-torch"
## Clone ai-edge-torch
if [ ! -d "./ai-edge-torch" ]; then

    git clone https://github.com/SNU-RTOS/ai-edge-torch.git
    rm -r ./ai-edge-torch/ai_edge_torch/generative/examples/cpp

    ln -s ${LLM_APP_SRC} ./ai-edge-torch/ai_edge_torch/generative/examples/cpp

    # Update Tensorflow PATH in ai-edge-torch/WORKSPACE for build
    WORKSPACE_FILE="${AI_EDGE_TORCH_PATH}/WORKSPACE"
    sed -i "s|path = \".*\"|path = \"${TENSORFLOW_PATH}\"|" "$WORKSPACE_FILE"
    echo "[INFO] Updated tensorflow local_repository path in ${TENSORFLOW_PATH}/WORKSPACE to: ${TENSORFLOW_PATH}"
else
    echo "[INFO] ai-edge-torch is already installed, skipping ..."
fi

## Clone tensorflow
echo "[INFO] Installing tensorflow"
if [ ! -d "./tensorflow" ]; then
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    git switch --detach ${TENSORFLOW_COMMIT_HASH}
    echo "[INFO] Patching tensorflow source to build with ai-edge-torch"
    patch -p1 <../ai-edge-torch/bazel/org_tensorflow_system_python.diff
else
    echo "[INFO] tensorflow is already installed, skipping ..."
fi

########## Build LiteRT_LLM_Inference_app ##########
echo "[INFO] Build ${LLM_APP_BINARY_NAME}"
echo "========================"
cd ${ROOT_PATH}
${ROOT_PATH}/build.sh
cd ${ROOT_PATH}
echo "========================"

########## Make soft symlink ##########
echo "[INFO] Succefully built ${LLM_APP_BINARY_NAME}"
echo "[INFO] Making soft symbolic link ${LLM_APP_BINARY_NAME} from ${LLM_APP_BINARY_PATH} to ${ROOT_PATH}"
if [ -f "${LLM_APP_BINARY_NAME}" ]; then
    rm ${LLM_APP_BINARY_NAME}
fi
ln -s ${LLM_APP_BINARY_PATH} ${LLM_APP_BINARY_NAME}

echo "[INFO] Setup finished."
