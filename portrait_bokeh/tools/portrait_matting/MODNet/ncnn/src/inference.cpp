// C++
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// NCNN
#include <ncnn/cpu.h>
#include <ncnn/net.h>
#include <ncnn/datareader.h>


void cv_show(const cv::Mat& image, const std::string& message="crane") {
	cv::imshow(message, image);
	cv::waitKey(0);
	cv::destroyAllWindows();
}



int main() {
	// 查看工作目录
	std::cout << std::filesystem::current_path().string() << std::endl;

	// 加载 onnx
	ncnn::Net network;
	network.load_param("./MODNet.param");
	network.load_model("./MODNet.bin");

	// 读取一张图像
	cv::Mat image = cv::imread("./demo3.png");
	const uint32_t height = image.rows;
	const uint32_t width  = image.cols;

	// 准备推理会话
	ncnn::Extractor executor = network.create_extractor();
	executor.set_light_mode(true);

	// 准备输入
	constexpr uint32_t img_h = 384;
	constexpr uint32_t img_w = 384;
	ncnn::Mat input_tensor = ncnn::Mat::from_pixels_resize(
		image.data, ncnn::Mat::PIXEL_BGR, width, height, img_h, img_w);
	executor.input("input", input_tensor);

	// 准备输出
	ncnn::Mat output_tensor;
	executor.extract("output", output_tensor);
	std::cout << output_tensor.c << ", " << output_tensor.h << ", " << output_tensor.w << std::endl;

	// 取出数据指针
	float* res_ptr = (float*)output_tensor.data;

	// 准备一块数据来存放解析结果
	cv::Mat matting_mask(img_h, img_w, CV_32F);
	float* mask_ptr = matting_mask.ptr<float>();
	const uint32_t length = img_h * img_w;
	for (uint32_t i = 0; i < length; ++i) {
		mask_ptr[i] = res_ptr[i] * 255;
	}

	// resize
	cv::resize(matting_mask, matting_mask, {width, height});

	// 转成 mask
	matting_mask.convertTo(matting_mask, CV_8UC1);

	// 展示
	cv_show(matting_mask);

	return 0;
}