// Compute exponential on vector element-wise (for softmax)
__kernel void exp_vec(
    __global const float *in,
    __global float *out)
{
    int gid = get_global_id(0);
    out[gid] = exp(in[gid]);
}
__kernel void matmul(
    const int n,
    const int m,
    __global const float *A,
    __global const float *B,
    __global float *Y)
{
	int i = get_global_id(0); //gets the global work-item ID (threadID.x) of the thread.
	int j = get_global_id(1); //gets the global work-item ID (threadID.y) of the thread.
	for(int k = 0; k < m; k++){
	    Y[i*n+j] += B[i*n+k]*A[k*n+j];
    }
}