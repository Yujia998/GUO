#!/usr/bin/env bash

./benchmark.py \
    --batching paged-attn \
    --block_size 64\
    --swap_policy eager \
    --prompt_count 100 \
    --prompt_lens_mean 55 \
    --generation_lens_mean 7 \
    --cluster ./data/clusters/1_a100/h1.json \
    --qps 50 \
    --eviction_policy lru \
    --psla ./data/psla/llama-7b.json \
    --dataset_json_path ./dataset/longbench/converted/llama-2-7b.json \
  
