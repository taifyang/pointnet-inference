#include <iostream>
#include <vector>
#include <fstream>
#include <openvino/openvino.hpp>


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
	ov::Core core;                
	//auto model = core.compile_model("cls.onnx","CPU"); 
	auto model = core.compile_model("./cls/cls_fp16.xml","CPU"); 
	auto iq = model.create_infer_request();
	auto input = iq.get_input_tensor(0);
	auto output = iq.get_output_tensor(0);
	input.set_shape({ 1, 3, point_num });
	float* input_data_host = input.data<float>();  

	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < point_num; j++)
		{
			input_data_host[point_num * i + j] = points[3 * j + i];
		}
	}
	
	iq.infer();
	
	float* prob = output.data<float>();
	int predict_label = std::max_element(prob, prob + class_num) - prob;
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
