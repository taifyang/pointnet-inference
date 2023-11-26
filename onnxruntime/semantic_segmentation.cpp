#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <onnxruntime_cxx_api.h>


const int point_num = 4096;
const int class_num = 13;


struct point
{
	float m_x, m_y, m_z, m_r, m_g, m_b, m_normal_x, m_normal_y, m_normal_z;
	point() :
		m_x(0), m_y(0), m_z(0), m_r(0), m_g(0), m_b(0), m_normal_x(0), m_normal_y(0), m_normal_z(0) {}
	point(float x, float y, float z, float r, float g, float b) :
		m_x(x), m_y(y), m_z(z), m_r(r), m_g(g), m_b(b), m_normal_x(0), m_normal_y(0), m_normal_z(0) {}
	point(float x, float y, float z, float r, float g, float b, float normal_x, float normal_y, float normal_z) :
		m_x(x), m_y(y), m_z(z), m_r(r), m_g(g), m_b(b), m_normal_x(normal_x), m_normal_y(normal_y), m_normal_z(normal_z) {}
};


int main()
{
	float x, y, z, r, g, b, l;
	std::vector<point> pts;
	std::vector<float> points_x, points_y, points_z;
	int points_num = 0;
	std::ifstream infile("Area_1_conferenceRoom_1.txt");
	while (infile >> x >> y >> z >> r >> g >> b >> l)
	{
		point pt(x, y, z, r, g, b);
		pts.push_back(pt);
		points_x.push_back(x);
		points_y.push_back(y);
		points_z.push_back(z);
		points_num++;
	}

	float x_min = *std::min_element(points_x.begin(), points_x.end());
	float y_min = *std::min_element(points_y.begin(), points_y.end());
	float z_min = *std::min_element(points_z.begin(), points_z.end());
	float x_max = *std::max_element(points_x.begin(), points_x.end());
	float y_max = *std::max_element(points_y.begin(), points_y.end());
	float z_max = *std::max_element(points_z.begin(), points_z.end());

	float stride = 0.5;
	float block_size = 1.0;
	srand((int)time(0));

	int grid_x = ceil((x_max - x_min - block_size) / stride) + 1;
	int grid_y = ceil((y_max - y_min - block_size) / stride) + 1;

	std::vector<point> data_room;
	std::vector<int> index_room;
	for (size_t index_y = 0; index_y < grid_y; index_y++)
	{
		for (size_t index_x = 0; index_x < grid_x; index_x++)
		{
			float s_x = x_min + index_x * stride;
			float e_x = std::min(s_x + block_size, x_max);
			s_x = e_x - block_size;
			float s_y = y_min + index_y * stride;
			float e_y = std::min(s_y + block_size, y_max);
			s_y = e_y - block_size;

			std::vector<int> point_idxs;
			for (size_t i = 0; i < points_num; i++)
			{
				if (points_x[i] >= s_x && points_x[i] <= e_x && points_y[i] >= s_y && points_y[i] <= e_y)
					point_idxs.push_back(i);
			}
			if (point_idxs.size() == 0)
				continue;

			int num_batch = ceil(point_idxs.size() * 1.0 / point_num);
			int point_size = num_batch * point_num;
			bool replace = (point_size - point_idxs.size() <= point_idxs.size() ? false : true);

			std::vector<int> point_idxs_repeat;
			if (replace)
			{
				for (size_t i = 0; i < point_size - point_idxs.size(); i++)
				{
					int id = rand() % point_idxs.size();
					point_idxs_repeat.push_back(point_idxs[id]);
				}
			}
			else
			{
				std::vector<bool> flags(pts.size(), false);
				for (size_t i = 0; i < point_size - point_idxs.size(); i++)
				{
					int id = rand() % point_idxs.size();
					while (true)
					{
						if (flags[id] == false)
						{
							flags[id] = true;
							break;
						}
						id = rand() % point_idxs.size();
					}
					point_idxs_repeat.push_back(point_idxs[id]);
				}
			}
			point_idxs.insert(point_idxs.end(), point_idxs_repeat.begin(), point_idxs_repeat.end());

			std::random_device rd;
			std::mt19937 g(rd());	// 随机数引擎:基于梅森缠绕器算法的随机数生成器
			std::shuffle(point_idxs.begin(), point_idxs.end(), g);	// 打乱顺序，重新排序（随机序列）

			std::vector<point> data_batch;
			for (size_t i = 0; i < point_idxs.size(); i++)
			{
				data_batch.push_back(pts[point_idxs[i]]);
			}

			for (size_t i = 0; i < point_size; i++)
			{
				data_batch[i].m_normal_x = data_batch[i].m_x / x_max;
				data_batch[i].m_normal_y = data_batch[i].m_y / y_max;
				data_batch[i].m_normal_z = data_batch[i].m_z / z_max;
				data_batch[i].m_x -= (s_x + block_size / 2.0);
				data_batch[i].m_y -= (s_y + block_size / 2.0);
				data_batch[i].m_r /= 255.0;
				data_batch[i].m_g /= 255.0;
				data_batch[i].m_b /= 255.0;
				data_room.push_back(data_batch[i]);
				index_room.push_back(point_idxs[i]);
			}
		}
	}

	int n = point_num, m = index_room.size() / n;
	std::vector<std::vector<point>> data_rooms(m, std::vector<point>(n, point()));
	std::vector<std::vector<int>> index_rooms(m, std::vector<int>(n, 0));
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			data_rooms[i][j] = data_room[i * n + j];
			index_rooms[i][j] = index_room[i * n + j];
		}
	}

	std::vector<std::vector<int>> vote_label_pool(points_num, std::vector<int>(class_num, 0));
	int num_blocks = data_rooms.size();

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sem_seg");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	OrtCUDAProviderOptions cuda_option;
	cuda_option.device_id = 0;
	cuda_option.arena_extend_strategy = 0;
	cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	cuda_option.gpu_mem_limit = SIZE_MAX;
	cuda_option.do_copy_in_default_stream = 1;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.AppendExecutionProvider_CUDA(cuda_option);

	const wchar_t* model_path = L"sem_seg.onnx";
	Ort::Session session(env, model_path, session_options);
	Ort::AllocatorWithDefaultOptions allocator;

	std::vector<const char*>  input_node_names;
	for (size_t i = 0; i < session.GetInputCount(); i++)
	{
		input_node_names.push_back(session.GetInputName(i, allocator));
	}

	std::vector<const char*> output_node_names;
	for (size_t i = 0; i < session.GetOutputCount(); i++)
	{
		output_node_names.push_back(session.GetOutputName(i, allocator));
	}

	const size_t input_tensor_size = 1 * 9 * point_num;
	std::vector<float> input_tensor_values(input_tensor_size);

	for (int sbatch = 0; sbatch < num_blocks; sbatch++)
	{
		int start_idx = sbatch;
		int end_idx = std::min(sbatch + 1, num_blocks);
		int real_batch_size = end_idx - start_idx;
		std::vector<point> batch_data = data_rooms[start_idx];
		std::vector<int> point_idx = index_rooms[start_idx];
		std::vector<float> batch(point_num * 9);
		for (size_t i = 0; i < point_num; i++)
		{
			batch[9 * i + 0] = batch_data[i].m_x;
			batch[9 * i + 1] = batch_data[i].m_y;
			batch[9 * i + 2] = batch_data[i].m_z;
			batch[9 * i + 3] = batch_data[i].m_r;
			batch[9 * i + 4] = batch_data[i].m_g;
			batch[9 * i + 5] = batch_data[i].m_b;
			batch[9 * i + 6] = batch_data[i].m_normal_x;
			batch[9 * i + 7] = batch_data[i].m_normal_y;
			batch[9 * i + 8] = batch_data[i].m_normal_z;
		}

		for (size_t i = 0; i < 9; i++)
		{
			for (size_t j = 0; j < point_num; j++)
			{
				input_tensor_values[i * point_num + j] = batch[9 * j + i];
			}
		}

		std::vector<int64_t> input_node_dims = { 1, 9, point_num };
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

		std::vector<Ort::Value> ort_inputs;
		ort_inputs.push_back(std::move(input_tensor));

		std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

		const float* rawOutput = output_tensors[0].GetTensorData<float>();
		std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
		size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
		std::vector<float> prob(rawOutput, rawOutput + count);

		std::vector<std::vector<float>> outputs(point_num, std::vector<float>(class_num, 0));

		for (size_t i = 0; i < point_num; i++)
		{
			for (size_t j = 0; j < class_num; j++)
			{
				outputs[i][j] = prob[i * class_num + j];
			}
		}

		std::vector<int> pred_label(point_num, 0);
		for (size_t i = 0; i < point_num; i++)
		{
			pred_label[i] = std::max_element(outputs[i].begin(), outputs[i].end()) - outputs[i].begin();
			vote_label_pool[point_idx[i]][pred_label[i]] += 1;
		}
	}

	std::ofstream outfile("pred.txt");
	for (size_t i = 0; i < points_num; i++)
	{
		int max_index = std::max_element(vote_label_pool[i].begin(), vote_label_pool[i].end()) - vote_label_pool[i].begin();
		outfile << pts[i].m_x << " " << pts[i].m_y << " " << pts[i].m_z << " " << max_index << std::endl;
	}
	outfile.close();

	return 0;
}
