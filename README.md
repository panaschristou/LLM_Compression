# Compressing Large Language Models With Pruning, Distillation, and Quantization

# Objectives, Solution Approach, Value of Solution 

Objectives:
- To apply structured pruning to LLMs, thereby reducing the number of active weights and connections, based on the principles of Unstructured and Structured Pruning (SparseGPT and LLM-Pruner).
- To implement quantization techniques that convert floating-point representations to lower bit representations, drawing from the 4-bit NormalFloat (NF4) quantization and Double Quantization (DQ) methods discussed in QLORA.
- To explore the combined application of pruning, distillation, and quantization to balance the trade-offs between inference speed, memory usage, and model accuracy.
- To maintain the task-agnostic capabilities of LLMs post-compression, ensuring that they continue to function as versatile task solvers.
Solution Approach:
- We used LLM-Pruner to perform structured pruning on 4 different pruning ratios for 2 different levels, block and channel and we used SparseGPT to perform unstructured pruning with 9 different ratios.
- We run unstructured pruning twice to find the effect on the perplexity and accuracy and compare with the same pruning ratio applied once.
- We run pruning and then quantization to see the effect on the accuracy and perplexity and we evaluated the inference performance.
- We recorded a demo to show the capabilities of the structured models compared to the pretrained model.
Value of Solution:
- We examine the usability of pruned and quantized models and the effectiveness of different types of pruning and quantization techniques and how these allow for different types of applications like embedded devices.


# Solution Approach 
- Pruning
  - Structured Pruning
  - Unstructured Pruning
- Distillation
  - Post Pruning Distillation after structured Pruning
- Quantization
  - 4 bit NormalFloat quantization

![image](results/methodpaper_model_structure_new.png)

# Results

![image](results/bubble_wikitext2.png)
![image](results/bubble_ptb.png)
![image](results/bubble_c4.png)

![image](https://github.com/panaschristou/LLM_Compression/blob/main/results/quantized_model_performance_wikitext2.png)
![image](https://github.com/panaschristou/LLM_Compression/blob/main/results/quantized_model_performance_ptb.png)
![image](https://github.com/panaschristou/LLM_Compression/blob/main/results/quantized_model_performance_c4.png)


![results/performance_infer_compute_line_wikitext2_Computational%20Complexity%20(GMac).png](https://github.com/panaschristou/LLM_Compression/blob/main/results/performance_infer_compute_line_wikitext2_Number%20of%20Parameters%20(M).png)
![[image](results/performance_infer_compute_line_wikitext2_Number of Parameters (M).png)](https://github.com/panaschristou/LLM_Compression/blob/main/results/performance_infer_compute_line_wikitext2_GPU%20Memory%20Requirements%20(MiB).png)https://github.com/panaschristou/LLM_Compression/blob/main/results/performance_infer_compute_line_wikitext2_GPU%20Memory%20Requirements%20(MiB).png
![[image](results/performance_infer_compute_line_wikitext2_Computational Complexity (GMac).png)](https://github.com/panaschristou/LLM_Compression/blob/main/results/performance_infer_compute_line_wikitext2_Computational%20Complexity%20(GMac).png)https://github.com/panaschristou/LLM_Compression/blob/main/results/performance_infer_compute_line_wikitext2_Computational%20Complexity%20(GMac).png
