#include <iostream>
#include <vector>
#include <fstream>
#include <onnxruntime_cxx_api.h>


const int point_num = 1024;
const int class_num = 10;


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


void classfier(std::vector<float> & points)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cls");
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

	const wchar_t* model_path = L"cls.onnx";
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

	const size_t input_tensor_size = 1 * 3 * point_num ;
	std::vector<float> input_tensor_values(input_tensor_size);
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < point_num; j++)
		{
			input_tensor_values[point_num * i + j] = points[3 * j + i];
		}
	}

	std::vector<int64_t> input_node_dims = { 1, 3, point_num };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

	const float* rawOutput = output_tensors[0].GetTensorData<float>();
	std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	std::vector<float> output(rawOutput, rawOutput + count);

	int predict_label = std::max_element(output.begin(), output.end()) - output.begin();
	std::cout << predict_label << std::endl;
}


int main()
{
	std::vector<float> points;
	float x, y, z, nx, ny, nz;
	char ch;
	std::ifstream infile("bed_0610.txt");
	for (size_t i = 0; i < point_num; i++)
	{
		infile >> x >> ch >> y >> ch >> z >> ch >> nx >> ch >> ny >> ch >> nz;
		points.push_back(x);
		points.push_back(y);
		points.push_back(z);
	}
	infile.close();

	pc_normalize(points);

	classfier(points);

	return 0;
}
