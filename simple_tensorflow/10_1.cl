// Compute exponential on vector element-wise (for softmax)
__kernel void exp_vec(
    __global const float *in,
    __global float *out)
{
    int gid = get_global_id(0);
    out[gid] = exp(in[gid]);
}
#define TS 32
__kernel void matmul(
    const int M,
    const int K,
    __global const float *A,
    __global const float *B,
    __global float *Y)
{
    // local offsets
	const int row = get_local_id(0);
	const int col = get_local_id(1);
    // global row and col
	const int globalRow = get_group_id(0) * TS + row;
	const int globalCol = get_group_id(1) * TS + col;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
     
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //'Return'
    Y[globalCol*M + globalRow] = acc;
}