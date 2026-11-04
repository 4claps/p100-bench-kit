#!/usr/bin/env bash
# Build one llama.cpp engine into /opt/engines/<name>.
# usage: build-engine.sh NAME REPO COMMIT LABEL SPLIT ENV_JSON MTP_JSON NOTES
set -euo pipefail
name=$1 repo=$2 commit=$3 label=$4 split=$5 env_json=$6 mtp_json=$7 notes=$8
archs=${CUDA_ARCHS:-60;61;70}
jobs=${JOBS:-$(nproc)}
src=/src/$name
dest=/opt/engines/$name

git init -q "$src"
git -C "$src" remote add origin "$repo"
git -C "$src" fetch -q --depth 1 origin "$commit"
git -C "$src" checkout -q FETCH_HEAD

# Portable build: CPU code for every x86 variant chosen at runtime
# (old Xeons without AVX2 still run the offload module), CUDA for the
# listed architectures. CUDA 12.x is the last toolkit that targets sm_60/70.
cmake -S "$src" -B "$src/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$archs" \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
ninja -C "$src/build" -j"$jobs" llama-server llama-bench

mkdir -p "$dest/bin"
cp -a "$src/build/bin/." "$dest/bin/"
cat > "$dest/engine.json" <<EOF
{"name": "$name", "label": "$label", "repo": "$repo", "commit": "$commit",
 "cuda_archs": "$archs", "multi_gpu_split": "$split", "env": $env_json,
 "mtp_args": $mtp_json, "notes": "$notes"}
EOF
rm -rf "$src"
echo "built $name at $commit"
