#!/usr/bin/env bash

./benchmark.py \
    --batching paged-attn \
    --block_size 32\
    --swap_policy eager \
    --prompt_count 1000 \
    --prompt_lens_mean 55 \
    --generation_lens_mean 7 \
    --cluster ./data/clusters/1_a100/h1.json \
    --results_path ./results/benchmark/wikipedia_structured/llama-2-7b.json \
    --qps 50 \
    --eviction_policy lru \
    --psla ./data/psla/llama-7b-shared.json \
    --dataset_json_path ./dataset/wikipedia_structured/converted/llama-2-7b.json \
  
