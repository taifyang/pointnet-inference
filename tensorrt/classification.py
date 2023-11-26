import numpy as np
import tensorrt as trt
import pycuda.autoinit 
import pycuda.driver as cuda  


point_num = 1024

 
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


if __name__ == '__main__':
    logger = trt.Logger(trt.Logger.WARNING)
    with open("cls.trt", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_output0 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    h_output1 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(2)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output0 = cuda.mem_alloc(h_output0.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    stream = cuda.Stream()
    
    file = './bed_0610.txt'
    data = np.loadtxt(file, delimiter=',').astype(np.float32)
    point_set = data[:, 0:3]
    point_set = point_set[0:point_num, :]     
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    points = np.reshape(point_set, ((1, point_num, 3)))
    points = points.swapaxes(2, 1)
    
    np.copyto(h_input, points.ravel())

    with engine.create_execution_context() as context:
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output0), int(d_output1)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output0, d_output0, stream)
        cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
        stream.synchronize()
        outputs = np.argmax(h_output1)
        print(outputs)
