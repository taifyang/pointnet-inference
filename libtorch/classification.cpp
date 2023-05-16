#include <iostream>
#include <vector>
#include <fstream>
#include <torch/script.h>


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


void classfier(std::vector<float> & points)
{
	torch::Tensor points_tensor = torch::from_blob(points.data(), { 1, point_num, 3 }, torch::kFloat);
	points_tensor = points_tensor.to(torch::kCUDA);
	points_tensor = points_tensor.permute({ 0, 2, 1 });
	//std::cout << points_tensor << std::endl;

	torch::jit::script::Module module = torch::jit::load("cls.pt");
	module.to(torch::kCUDA);

	auto outputs = module.forward({ points_tensor }).toTuple();
	torch::Tensor out0 = outputs->elements()[0].toTensor();
	std::cout << out0 << std::endl;

	auto max_classes = out0.max(1);
	//auto max_result = std::get<0>(max_classes).item<float>();
	auto max_index = std::get<1>(max_classes).item<int>();
	std::cout << max_index << std::endl;
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
