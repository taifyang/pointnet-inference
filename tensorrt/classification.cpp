#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>


const int point_num = 1024;


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

void classfier(std::vector<float> & points)
{
	TRTLogger logger;
	nvinfer1::ICudaEngine* engine;

	auto engine_data = load_file("cls.engine");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if (engine == nullptr)
	{
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return;
	}

	nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
	cudaStream_t stream = nullptr;
	cudaStreamCreate(&stream);

	float* input_data_host = nullptr;
	const size_t input_numel = 1 * 3 * point_num;
	cudaMallocHost(&input_data_host, input_numel * sizeof(float));
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < point_num; j++)
		{
			input_data_host[point_num * i + j] = points[3 * j + i];
		}
	}

	float* input_data_device = nullptr;
	float output_data_host1[4096];
	float* output_data_device1 = nullptr;
	float output_data_host2[10];
	float* output_data_device2 = nullptr;
	cudaMalloc(&input_data_device, input_numel * sizeof(float));
	cudaMalloc(&output_data_device1, sizeof(output_data_host1));
	cudaMalloc(&output_data_device2, sizeof(output_data_host2));
	cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream);
	float* bindings[] = { input_data_device, output_data_device1, output_data_device2 };

	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
	cudaMemcpyAsync(output_data_host1, output_data_device1, sizeof(output_data_host1), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(output_data_host2, output_data_device2, sizeof(output_data_host2), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	int predict_label = std::max_element(output_data_host2, output_data_host2 + 10) - output_data_host2;
	std::cout << "\npredict_label: " << predict_label << std::endl;

	cudaStreamDestroy(stream);
	execution_context->destroy();
	engine->destroy();
	runtime->destroy();
}


int main()
{
	std::vector<float> points;
	std::ifstream infile;
	float x, y, z, nx, ny, nz;
	char ch;
	infile.open("bed_0610.txt");
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
