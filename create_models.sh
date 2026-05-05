#!/bin/bash

ollama create gemma_9b_q4:latest -f ./models/benchmark/gemma_9b_q4
ollama create gemma_9b_q8:latest -f ./models/benchmark/gemma_9b_q8

ollama create llama_8b_q4:latest -f ./models/benchmark/llama_8b_q4
ollama create llama_8b_q8:latest -f ./models/benchmark/llama_8b_q8

ollama create dolphin_8b_q4:latest -f ./models/benchmark/dolphin_8b_q4
ollama create dolphin_8b_q8:latest -f ./models/benchmark/dolphin_8b_q8

ollama create qwen_7b_q4:latest -f ./models/benchmark/qwen_7b_q4
ollama create qwen_7b_q8:latest -f ./models/benchmark/qwen_7b_q8

ollama create cogito_8b_q4:latest -f ./models/benchmark/cogito_8b_q4
ollama create cogito_8b_q8:latest -f ./models/benchmark/cogito_8b_q8

ollama create qwen_8b_q4:latest -f ./models/benchmark/qwen_8b_q4
ollama create qwen_8b_q8:latest -f ./models/benchmark/qwen_8b_q8

ollama create granite_8b_q4:latest -f ./models/benchmark/granite_8b_q4 --quantize q4_k_m
ollama create granite_8b_q8:latest -f ./models/benchmark/granite_8b_q8

ollama create gemma_9b_q4:latest -f ./models/benchmark/gemma_9b_q4
ollama create gemma_9b_q8:latest -f ./models/benchmark/gemma_9b_q8
