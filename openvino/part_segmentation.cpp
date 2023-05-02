#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <openvino/openvino.hpp>


const int point_num = 2048;
const int parts_num = 4;


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


std::vector<int> classfier(std::vector<float> & points, std::vector<float> & labels)
{
	std::vector<int> max_index(point_num, 0);

	ov::Core core;
	auto model = core.compile_model("best_model.onnx", "CPU");
	auto iq = model.create_infer_request();
	auto input = iq.get_input_tensor(0);
	auto output = iq.get_output_tensor(0);
	input.set_shape({ 1, 3, point_num });
	float* input_data_host = input.data<float>();


	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < point_num; j++)
		{
			input_data_host[i * point_num + j] = points[3 * j + i];
			//std::cout << input_data_host[i * point_num + j] << " ";
		}
		//std::cout << std::endl;
	}

	iq.infer();

	float* prob = output.data<float>();
	std::vector<std::vector<float>> outputs(point_num, std::vector<float>(parts_num, 0));

	for (size_t i = 0; i < point_num; i++)
	{
		for (size_t j = 0; j < parts_num; j++)
		{
			outputs[i][j] = prob[i * parts_num + j];
			//std::cout <<outputs[i][j] << " ";
		}
		//std::cout << std::endl;
	}

	for (size_t i = 0; i < point_num; i++)
	{
		max_index[i]= std::max_element(outputs[i].begin(), outputs[i].end()) - outputs[i].begin();
		//std::cout << max_index[i] << " ";
	}
	return max_index;
}


int main()
{
	std::vector<float> points, labels;
	float x, y, z, nx, ny, nz, label;
	std::ifstream infile;
	infile.open("plane.txt");
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

	std::vector<int> result = classfier(points, labels);

	std::fstream outfile;
	outfile.open("85a15+.txt", 'w');
	for (size_t i = 0; i < point_num; i++)
	{
		outfile << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << " " << result[i]<< std::endl;
	}
	outfile.close();

	return 0;
}
