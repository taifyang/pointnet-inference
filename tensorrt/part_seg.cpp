#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>


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


class TRTLogger : public nvinfer1::ILogger
{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
	{
		if (severity <= Severity::kINFO)
			printf(msg);
	}
} logger;

std::vector<unsigned char> load_file(const std::string& file)
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
		return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)& data[0], length);
	}
	in.close();
	return data;
}

std::vector<int> classfier(std::vector<float>& points, std::vector<float>& labels)
{
	std::vector<int> max_index(point_num, 0);

	TRTLogger logger;
	nvinfer1::ICudaEngine* engine;

	auto engine_data = load_file("part_seg.engine");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if (engine == nullptr)
	{
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return max_index;
	}

	nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
	cudaStream_t stream = nullptr;
	cudaStreamCreate(&stream);

	float* input_data_host1 = nullptr;
	const size_t input_numel = 1 * 3 * point_num;
	cudaMallocHost(&input_data_host1, input_numel * sizeof(float));
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < point_num; j++)
		{
			input_data_host1[point_num * i + j] = points[3 * j + i];
		}
	}

	float* input_data_host2 = nullptr;
	cudaMallocHost(&input_data_host2, 1 * 1 * class_num * sizeof(float));
	for (size_t i = 0; i < class_num; i++)
	{
		input_data_host2[i] = labels[i];
	}

	float* input_data_device1 = nullptr;
	float* input_data_device2 = nullptr;
	float output_data_host1[1 * 128 * 128];
	float* output_data_device1 = nullptr;
	float output_data_host2[1 * point_num * parts_num];
	float* output_data_device2 = nullptr;
	cudaMalloc(&input_data_device1, input_numel * sizeof(float));
	cudaMalloc(&input_data_device2, class_num * sizeof(float));
	cudaMalloc(&output_data_device1, sizeof(output_data_host1));
	cudaMalloc(&output_data_device2, sizeof(output_data_host2));
	cudaMemcpyAsync(input_data_device1, input_data_host1, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(input_data_device2, input_data_host2, class_num * sizeof(float), cudaMemcpyHostToDevice, stream);
	float* bindings[] = { input_data_device1, input_data_device2, output_data_device1, output_data_device2 };

	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
	cudaMemcpyAsync(output_data_host1, output_data_device1, sizeof(output_data_host1), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(output_data_host2, output_data_device2, sizeof(output_data_host2), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	std::vector<std::vector<float>> outputs(point_num, std::vector<float>(parts_num, 0));
	for (size_t i = 0; i < point_num; i++)
	{
		for (size_t j = 0; j < parts_num; j++)
		{
			outputs[i][j] = output_data_host2[i * parts_num + j];
		}
	}

	for (size_t i = 0; i < point_num; i++)
	{
		max_index[i] = std::max_element(outputs[i].begin(), outputs[i].end()) - outputs[i].begin();
	}

	cudaStreamDestroy(stream);
	execution_context->destroy();
	engine->destroy();
	runtime->destroy();

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
		outfile << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << " " << result[i] << std::endl;
	}
	outfile.close();

	return 0;
}