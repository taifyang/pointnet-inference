#include <iostream>
#include <vector>
#include <fstream>
#include <torch/script.h>


const int point_num = 2048;


void pc_normalize(std::vector<float>& points)
{
	int N = points.size() / 3;
	float mean_x = 0, mean_y = 0, mean_z = 0;
	for (size_t i = 0; i < N; ++i)
	{
		mean_x += points[3 * i];
		mean_y += points[3 * i + 1];
		mean_z += points[3 * i + 2];
	}
	mean_x /= N;
	mean_y /= N;
	mean_z /= N;

	for (size_t i = 0; i < N; ++i)
	{
		points[3 * i] -= mean_x;
		points[3 * i + 1] -= mean_y;
		points[3 * i + 2] -= mean_z;
	}

	float m = 0;
	for (size_t i = 0; i < N; ++i)
	{
		if (sqrt(pow(points[3 * i], 2) + pow(points[3 * i + 1], 2) + pow(points[3 * i + 2], 2)) > m)
			m = sqrt(pow(points[3 * i], 2) + pow(points[3 * i + 1], 2) + pow(points[3 * i + 2], 2));
	}

	for (size_t i = 0; i < N; ++i)
	{
		points[3 * i] /= m;
		points[3 * i + 1] /= m;
		points[3 * i + 2] /= m;
	}
}


void resample(std::vector<float> & points, int nums)
{
	srand((int)time(0));
	std::vector<int> choice(nums);
	for (size_t i = 0; i < nums; i++)
	{
		choice[i] = rand() % (points.size() / 3);
	}

	std::vector<float> temp_points(3 * nums);
	for (size_t i = 0; i < nums; i++)
	{
		temp_points[3 * i] = points[3 * choice[i]];
		temp_points[3 * i + 1] = points[3 * choice[i] + 1];
		temp_points[3 * i + 2] = points[3 * choice[i] + 2];
	}
	points = temp_points;
}


at::Tensor classfier(std::vector<float> & points, std::vector<float> & labels)
{
	torch::Tensor points_tensor = torch::from_blob(points.data(), { 1, point_num, 3 }, torch::kFloat);
	torch::Tensor labels_tensor = torch::from_blob(labels.data(), { 1, 1, 1 }, torch::kFloat);

	points_tensor = points_tensor.to(torch::kCUDA);
	points_tensor = points_tensor.permute({ 0, 2, 1 });
	//std::cout << points_tensor << std::endl;
	labels_tensor = labels_tensor.to(torch::kCUDA);
	//std::cout << labels_tensor << std::endl;

	torch::jit::script::Module module = torch::jit::load("best_model.pt");
	module.to(torch::kCUDA);

	auto outputs = module.forward({ points_tensor, labels_tensor }).toTuple();
	torch::Tensor out0 = outputs->elements()[0].toTensor();
	//std::cout << out0 << std::endl; //[ CUDAFloatType{1,2048,4} ]
	out0 = torch::squeeze(out0);
	//std::cout << out0 << std::endl; //[ CUDAFloatType{2048,4} ]

	auto max_classes = out0.max(1);
	auto max_index = std::get<1>(max_classes);
	//std::cout << max_index << std::endl;

	return max_index;
}


int main()
{
	std::vector<float> points, labels;
	float x, y, z, nx, ny, nz, label;
	std::ifstream infile;
	infile.open("85a15c26a6e9921ae008cc4902bfe3cd.txt");
	while (infile >> x >> y >> z >> nx >> ny >> nz >> label)
	{
		points.push_back(x);
		points.push_back(y);
		points.push_back(z);
	}
	labels.push_back(1.0);
	infile.close();

	pc_normalize(points);

	resample(points, point_num);

	at::Tensor result = classfier(points, labels);

	std::fstream outfile;
	outfile.open("85a15+.txt", 'w');
	for (size_t i = 0; i < point_num; i++)
	{
		outfile << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << " " << result[i].item<int>() << std::endl;
	}
	outfile.close();

	return 0;
}
