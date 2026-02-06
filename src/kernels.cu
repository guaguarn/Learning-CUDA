#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../tester/utils.h"
/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
// Trace
template <typename T>
__global__ void traceKernel(const T *d_input, T *d_ans, size_t rows, size_t cols, size_t n)
{
	__shared__ T sdata[256];

	size_t tid = threadIdx.x;
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	T local_sum = 0;

	// 网格跨步循环
	for (; idx < n; idx += stride)
	{
		local_sum += d_input[idx * cols + idx]; // 取对角线元素
	}

	sdata[tid] = local_sum;
	__syncthreads();

	// 归约求和
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		atomicAdd(d_ans, sdata[0]);
	}
}

template <typename T>
T trace(const std::vector<T> &h_input, size_t rows, size_t cols)
{
	size_t n = (rows > cols) ? cols : rows;
	T h_ans = 0;
	T *d_input, *d_ans;

	cudaMalloc(&d_input, rows * cols * sizeof(T));
	cudaMalloc(&d_ans, sizeof(T));

	cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ans, &h_ans, sizeof(T), cudaMemcpyHostToDevice);

	// 启动足够的block
	int threadSum = 256;
	int blockSum = (n + threadSum - 1) / threadSum;
	traceKernel<<<blockSum, threadSum>>>(d_input, d_ans, rows, cols, n);

	cudaDeviceSynchronize();
	cudaMemcpy(&h_ans, d_ans, sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_ans);
	return h_ans;
}
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
// Flash Attention
template <typename T>
__global__ void flashAttentionkernel(const T *q, const T *k, const T *v, T *o,
									 int batch_size, int target_seq_len, int src_seq_len,
									 int query_heads, int kv_heads, int head_dim, bool is_causal)
{
	// [z,x,y]
	// q: [batch_size, target_seq_len, query_heads, head_dim]
	// k/v: [batch_size, src_seq_len, kv_heads, head_dim]
	// o: [batch_size, target_seq_len, query_heads, head_dim]

	int batch_idx = blockIdx.z;
	int token_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int query_heads_idx = blockIdx.y;

	// 边界处理
	if (batch_idx >= batch_size || query_heads_idx >= query_heads || token_idx >= target_seq_len)
		return;
	if (head_dim > 256)
		return;

	// GQA
	int g = query_heads / kv_heads;
	int kv_heads_idx = query_heads_idx / g;

	// 计算偏移 ((batch_idx * seq_len + seq_idx) * heads + head_idx) * head_dim
	int q_offset = ((batch_idx * target_seq_len + token_idx) * query_heads + query_heads_idx) * head_dim;
	int o_offset = ((batch_idx * target_seq_len + token_idx) * query_heads + query_heads_idx) * head_dim;

	const T *q_ptr = q + q_offset;
	T *o_ptr = o + o_offset;

	float maxv = -INFINITY;
	float sum = 0.0f; // 分母
	float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

	// 使用float以减少本地内存占用，精度对于attention足够
	float a[256];
	for (int d = 0; d < head_dim; d++)
		a[d] = 0.0f;

	for (int i = 0; i < src_seq_len; i++)
	{
		if (is_causal && i > token_idx)
			continue;

		int k_offset = ((batch_idx * src_seq_len + i) * kv_heads + kv_heads_idx) * head_dim;
		const T *k_ptr = k + k_offset;
		const T *v_ptr = v + k_offset;

		// 计算点积
		float s = 0.0f;
		for (int d = 0; d < head_dim; d++)
		{
			s += (float)q_ptr[d] * (float)k_ptr[d]; // 点积
		}
		s = s * scale; // 压缩

		float oldmax = maxv;
		if (s > maxv)
			maxv = s;
		float exp_factor = expf(oldmax - maxv); // 旧值衰减因子
		float curr_exp = expf(s - maxv);		 // 当前值指数

		sum = sum * exp_factor + curr_exp;

		// 更新
		for (int d = 0; d < head_dim; d++)
		{
			a[d] = a[d] * exp_factor + static_cast<float>(v_ptr[d]) * curr_exp;
		}
	}
	// 写入输出（归一化）
	for (int d = 0; d < head_dim; d++)
	{
		o_ptr[d] = static_cast<T>(a[d] / sum);
	}
}

template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
					const std::vector<T> &h_v, std::vector<T> &h_o,
					int batch_size, int target_seq_len, int src_seq_len,
					int query_heads, int kv_heads, int head_dim, bool is_causal)
{
	if (query_heads % kv_heads != 0)
	{
		printf("query_heads must be divisible by kv_heads\n");
		return;
	}
	if (head_dim > 256)
	{
		printf("head_dim(%d) exceeds maximum supported size (256)\n", head_dim);
		return;
	}
	size_t q_o_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
	size_t k_v_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
	h_o.resize(q_o_size);

	T *d_q, *d_k, *d_v, *d_o;
	cudaMalloc(&d_q, q_o_size * sizeof(T));
	cudaMalloc(&d_k, k_v_size * sizeof(T));
	cudaMalloc(&d_v, k_v_size * sizeof(T));
	cudaMalloc(&d_o, q_o_size * sizeof(T));

	cudaMemcpy(d_q, h_q.data(), q_o_size * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_k, h_k.data(), k_v_size * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, h_v.data(), k_v_size * sizeof(T), cudaMemcpyHostToDevice);

	dim3 block(256);
	dim3 grid((target_seq_len + 255) / 256, query_heads, batch_size);
	flashAttentionkernel<<<grid, block>>>(d_q, d_k, d_v, d_o,
										  batch_size, target_seq_len, src_seq_len,
										  query_heads, kv_heads, head_dim, is_causal);

	cudaDeviceSynchronize();
	cudaMemcpy(h_o.data(), d_o, q_o_size * sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(d_q);
	cudaFree(d_k);
	cudaFree(d_v);
	cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int> &, size_t, size_t);
template float trace<float>(const std::vector<float> &, size_t, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
									const std::vector<float> &, std::vector<float> &,
									int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half> &, const std::vector<half> &,
								   const std::vector<half> &, std::vector<half> &,
								   int, int, int, int, int, int, bool);