import numpy as np
import onnxruntime


point_num = 4096
class_num = 13
stride = 0.5
block_size = 1.0


if __name__ == '__main__':
    data = np.load('Area_1_conferenceRoom_1.npy')
    points = data[:,:6]
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride) + 1)
    data_room, index_room = np.array([]), np.array([])
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            point_idxs = np.where((points[:, 0] >= s_x) & (points[:, 0] <= e_x) & (points[:, 1] >= s_y) & (points[:, 1] <= e_y))[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / point_num))
            point_size = int(num_batch * point_num)
            replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            data_batch = points[point_idxs, :]
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
            data_batch[:, 3:6] /= 255.0
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
    data_room = data_room.reshape((-1, point_num, data_room.shape[1]))
    index_room = index_room.reshape((-1, point_num))

    onnx_session = onnxruntime.InferenceSession("best_model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    input_name=[]
    for node in onnx_session.get_inputs():
        input_name.append(node.name)

    output_name=[]
    for node in onnx_session.get_outputs():
        output_name.append(node.name)

    vote_label_pool = np.zeros((points.shape[0], class_num))
    num_blocks = data_room.shape[0]
    batch_data = np.zeros((1, point_num, 9))
    batch_point_index = np.zeros((1, point_num))

    for sbatch in range(num_blocks):
        print(sbatch, range(num_blocks))
        start_idx = sbatch
        end_idx = min(sbatch + 1, num_blocks)
        real_batch_size = end_idx - start_idx
        batch_data[0:real_batch_size, ...] = data_room[start_idx:end_idx, ...]
        batch_point_index[0:real_batch_size, ...] = index_room[start_idx:end_idx, ...]

        input_feed={}
        for name in input_name:
            input_feed[name] = batch_data.swapaxes(2, 1).astype(np.float32)

        seg_pred = onnx_session.run(None, input_feed)[0]

        batch_pred_label = np.argmax(seg_pred, 2)

        point_idx = batch_point_index[0:real_batch_size, ...]
        pred_label = batch_pred_label[0:real_batch_size, ...]
        for b in range(pred_label.shape[0]):
            for n in range(pred_label.shape[1]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1

    pred = np.argmax(vote_label_pool, 1)
    fout = open('pred.txt', 'w')
    for i in range(points.shape[0]):
        fout.write('%f %f %f %d\n' % (points[i, 0], points[i, 1], points[i, 2], pred[i]))
    fout.close()
