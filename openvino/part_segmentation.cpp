#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <openvino/openvino.hpp>


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


std::vector<int> classfier(std::vector<float> & points, std::vector<float> & labels)
{
	std::vector<int> max_index(point_num, 0);

	ov::Core core;
	//auto model = core.compile_model("part_seg.onnx", "CPU");
	auto model = core.compile_model("./part_seg/part_seg_fp16.xml", "CPU");
	auto iq = model.create_infer_request();

	auto input0 = iq.get_input_tensor(0);
	input0.set_shape({ 1, 3, point_num });
	float* input_data_host0 = input0.data<float>();
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < point_num; j++)
		{
			input_data_host0[i * point_num + j] = points[3 * j + i];
		}
	}

	auto input1 = iq.get_input_tensor(1);
	input1.set_shape({ 1, 1, class_num });
	float* input_data_host1 = input1.data<float>();
	for (size_t i = 0; i < class_num; i++)
	{
		input_data_host1[i] = labels[i];
	}

	iq.infer();

	auto output = iq.get_output_tensor(0);
	float* prob = output.data<float>();
	std::vector<std::vector<float>> outputs(point_num, std::vector<float>(parts_num, 0));

	for (size_t i = 0; i < point_num; i++)
	{
		for (size_t j = 0; j < parts_num; j++)
		{
			outputs[i][j] = prob[i * parts_num + j];
		}
	}

	for (size_t i = 0; i < point_num; i++)
	{
		max_index[i] = std::max_element(outputs[i].begin(), outputs[i].end()) - outputs[i].begin();
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
		outfile << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << " " << result[i] << std::endl;
	}
	outfile.close();

	return 0;
}
