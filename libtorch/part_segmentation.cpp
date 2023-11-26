#include <iostream>
#include <vector>
#include <fstream>
#include <torch/script.h>


const int point_num = 2048;
const int class_num = 16;


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


void resample(std::vector<float> & points)
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


at::Tensor classfier(std::vector<float> & points, std::vector<float> & labels)
{
	torch::Tensor points_tensor = torch::from_blob(points.data(), { 1, point_num, 3 }, torch::kFloat);
	torch::Tensor labels_tensor = torch::from_blob(labels.data(), { 1, 1, class_num }, torch::kFloat);

	points_tensor = points_tensor.to(torch::kCUDA);
	points_tensor = points_tensor.permute({ 0, 2, 1 });
	labels_tensor = labels_tensor.to(torch::kCUDA);

	torch::jit::script::Module module = torch::jit::load("part_seg.pt");
	module.to(torch::kCUDA);

	auto outputs = module.forward({ points_tensor, labels_tensor }).toTuple();
	torch::Tensor out0 = outputs->elements()[0].toTensor();
	out0 = torch::squeeze(out0);

	auto max_classes = out0.max(1);
	auto max_result = std::get<0>(max_classes);
	auto max_index = std::get<1>(max_classes);

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

	at::Tensor result = classfier(points, labels);

	std::fstream outfile("pred.txt", 'w');
	for (size_t i = 0; i < point_num; i++)
	{
		outfile << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << " " << result[i].item<int>() << std::endl;
	}
	outfile.close();

	return 0;
}
