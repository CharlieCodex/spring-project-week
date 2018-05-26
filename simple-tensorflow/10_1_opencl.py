import numpy as np
import pyopencl as cl

# export PYOPENCL_CTX='0:0'
mf=cl.mem_flags

class ClWrapper:
    def __init__(self, src_path):
        print('New ClWrapper on file {}'.format(src_path))
        self.cl_ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.cl_ctx)
        self.mf = cl.mem_flags
        self.cl_prog = cl.Program(self.cl_ctx, open(src_path).read()).build()
    
    def compute(self, in_np, out_np, func_name, global_size, local_size=None):
        func = [x for x in self.cl_prog.all_kernels() if x.function_name == func_name][0]
        in_buffs = ()
        for tensor in in_np:
            in_buffs += (cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tensor),)
        out_buffs = ()
        for tensor in out_np:
            out_buffs += (cl.Buffer(self.cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tensor),)
        print('running func')
        func(self.queue, global_size, local_size, *(in_buffs+out_buffs))
        print('copying tensors')
        for i, tensor in enumerate(out_np):
            print(tensor, i)
            cl.enqueue_copy(self.queue, tensor, out_buffs[i])
        return out_np

if __name__ == '__main__':
    n, m = 100, 100
    M = np.random.rand(n, m)
    N = np.random.rand(m, n)
    Y = np.zeros(shape=(n,n,))
    clw = ClWrapper('10_1.cl')
    #           inputs, output, name output dimensions, common axis 
    clw.compute(
        in_np=(np.int32(n),np.int32(m),M,N,),
        out_np=(Y,),
        func_name='matmul',
        global_size=(1,1,))
    print(Y)
