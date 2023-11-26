#include <iostream>
#include <vector>
#include <fstream>
#include <onnxruntime_cxx_api.h>


const int point_num = 2048;
const int class_num = 16;
const int parts_num = 50;


void pc_normalize(std::vector<float>& points)
{
	float mean_x = 0, mean_y = 0, mean_z = 0;
	for (size_t i = 0; i < point_num; ++i)
	{
		mean_x += points[3 * i];
		mean_y += points[3 * i + 1];
		mean_z += points[3 * i + 2];
	}
	mean_x /= point_num;
	mean_y /= point_num;
	mean_z /= point_num;

	for (size_t i = 0; i < point_num; ++i)
	{
		points[3 * i] -= mean_x;
		points[3 * i + 1] -= mean_y;
		points[3 * i + 2] -= mean_z;
	}

	float m = 0;
	for (size_t i = 0; i < point_num; ++i)
	{
		if (sqrt(pow(points[3 * i], 2) + pow(points[3 * i + 1], 2) + pow(points[3 * i + 2], 2)) > m)
			m = sqrt(pow(points[3 * i], 2) + pow(points[3 * i + 1], 2) + pow(points[3 * i + 2], 2));
	}

	for (size_t i = 0; i < point_num; ++i)
	{
		points[3 * i] /= m;
		points[3 * i + 1] /= m;
		points[3 * i + 2] /= m;
	}
}


void resample(std::vector<float>& points)
{
	srand((int)time(0));
	std::vector<int> choice(point_num);
	for (size_t i = 0; i < point_num; i++)
	{
		choice[i] = rand() % (points.size() / 3);
	}

	std::vector<float> temp_points(3 * point_num);
	for (size_t i = 0; i < point_num; i++)
	{
		temp_points[3 * i] = points[3 * choice[i]];
		temp_points[3 * i + 1] = points[3 * choice[i] + 1];
		temp_points[3 * i + 2] = points[3 * choice[i] + 2];
	}
	points = temp_points;
}


std::vector<int> classfier(std::vector<float> & points, std::vector<float> & labels)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "part_seg");
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

	const wchar_t* model_path = L"part_seg.onnx";
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

	const size_t input_tensor_size0 = 1 * 3 * point_num;
	std::vector<float> input_tensor_values0(input_tensor_size0);
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < point_num; j++)
		{
			input_tensor_values0[point_num * i + j] = points[3 * j + i];
		}
	}
	std::vector<int64_t> input_node_dims0 = { 1, 3, point_num };
	auto memory_info0 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor0 = Ort::Value::CreateTensor<float>(memory_info0, input_tensor_values0.data(), input_tensor_size0, input_node_dims0.data(), input_node_dims0.size());

	const size_t input_tensor_size1 = 1 * 1 * class_num;
	std::vector<float> input_tensor_values1(input_tensor_size0);
	for (size_t i = 0; i < class_num; i++)
	{
		input_tensor_values1[i] = labels[i];
	}
	std::vector<int64_t> input_node_dims1 = { 1, 1, class_num };
	auto memory_info1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor1 = Ort::Value::CreateTensor<float>(memory_info1, input_tensor_values1.data(), input_tensor_size1, input_node_dims1.data(), input_node_dims1.size());

	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(input_tensor0));
	inputs.push_back(std::move(input_tensor1));

	std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

	const float* rawOutput = outputs[0].GetTensorData<float>();
	std::vector<int64_t> outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	std::vector<float> pred(rawOutput, rawOutput + count);

	std::vector<std::vector<float>> preds(point_num, std::vector<float>(parts_num, 0));

	for (size_t i = 0; i < point_num; i++)
	{
		for (size_t j = 0; j < parts_num; j++)
		{
			preds[i][j] = pred[i * parts_num + j];
		}
	}

	std::vector<int> max_index(point_num, 0);

	for (size_t i = 0; i < point_num; i++)
	{
		max_index[i]= std::max_element(preds[i].begin(), preds[i].end()) - preds[i].begin();
	}
	return max_index;
}


int main()
{
	std::vector<float> points, labels;
	float x, y, z, nx, ny, nz, label;
	std::ifstream infile("85a15c26a6e9921ae008cc4902bfe3cd.txt");
	while (infile >> x >> y >> z >> nx >> ny >> nz >> label)
	{
		points.push_back(x);
		points.push_back(y);
		points.push_back(z);
	}
	for (size_t i = 0; i < class_num; i++)
	{
		labels.push_back(0.0);
	}
	labels[0] = 1.0;
	infile.close();

	pc_normalize(points);

	resample(points);

	std::vector<int> result = classfier(points, labels);

	std::fstream outfile("pred.txt", 'w');
	for (size_t i = 0; i < point_num; i++)
	{
		outfile << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << " " << result[i]<< std::endl;
	}
	outfile.close();

	return 0;
}
