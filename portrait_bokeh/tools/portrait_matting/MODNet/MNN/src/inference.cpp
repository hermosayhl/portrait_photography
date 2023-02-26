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
// MNN
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>


void cv_show(const cv::Mat& image, const std::string& message="crane") {
	cv::imshow(message, image);
	cv::waitKey(0);
	cv::destroyAllWindows();
}



/*
	尚未完成的
		1. dynamic shape inference
		2. Android Studio 部署到手机上
*/

int main() {
	// 查看工作目录
	std::cout << std::filesystem::current_path().string() << std::endl;

	// 加载 mnn 模型
	const std::string model_path("./MODNet-static-sim.mnn");
	std::unique_ptr<MNN::Interpreter> network(MNN::Interpreter::createFromFile(model_path.c_str()));

	// 准备一些任务调度中的参数
	constexpr uint32_t forward   = MNN_FORWARD_CPU;
	constexpr uint32_t precision = 2;
	constexpr uint32_t power     = 0;
	constexpr uint32_t memory    = 0;
	constexpr uint32_t threads   = 1;
	// 配置
	MNN::ScheduleConfig config;
    config.numThread        = threads;
    config.type             = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power     = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory    = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig    = &backendConfig;

    // 4. 创建session
    auto session = network->createSession(config);
    network->releaseModel();

    // 读取图像
	const std::string image_path("demo3.png");
	cv::Mat input_image          = cv::imread(image_path);
	const uint32_t origin_height = input_image.rows;
	const uint32_t origin_width  = input_image.cols;

	// 前处理, BGR → RGB, resize to (384, 384)
	constexpr uint32_t infer_h = 384;
	constexpr uint32_t infer_w = 384;
	constexpr uint32_t infer_c = 3;
	cv::Mat infer_image;
	cv::resize(input_image, infer_image, {infer_w, infer_h}, 0, 0, cv::INTER_LINEAR);
	cv::cvtColor(infer_image, infer_image, cv::COLOR_BGR2RGB);

	// 从网络中取出输入
	auto input_tensor = network->getSessionInput(session, "input");

	// 准备一块区域, 数据从 hwc 转换成 chw
	constexpr uint32_t infer_hw = infer_h * infer_w;
	// make_unique
	auto input_buffer = new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE);
	for (uint32_t c = 0; c < infer_c; ++c) {
		float* write_ptr = input_buffer->host<float>() + c * infer_hw;
		for (uint32_t i  = 0; i < infer_hw; ++i) {
			write_ptr[i] = 1.f * infer_image.data[i * infer_c + c];
		}
	}

	// 把输入拷贝到网络中, 如果需要的话
	input_tensor->copyFromHostTensor(input_buffer);
	std::cout << "infer===> infer_image\t" << (void*)infer_image.data << "\n";
	std::cout << "infer===> input_buffer\t" << input_buffer->host<float>() << "\n";
	std::cout << "infer===> infer_image\t" << input_tensor->host<float>() << "\n";

	// 推理
	network->runSession(session);

	// 获取输出
	auto output_tensor = network->getSessionOutput(session, "output");

	// host 上准备一块数据
	auto output_buffer = new MNN::Tensor(output_tensor, MNN::Tensor::CAFFE);

	// 把数据拷贝到 host
	output_tensor->copyToHostTensor(output_buffer);
	
	// 解析输出缓冲区中的内容
	cv::Mat matting_mask(infer_h, infer_w, CV_32FC1, output_buffer->host<float>());
	// shape 不大对劲得看 MNN/Tensor.hpp 是怎么写的, 这个输出比较奇怪
	// 还有这个 copy 难道是深拷贝吗?????? 逆天
	std::cout << "parse===> output shape ["  << output_tensor->height() << ", " << output_tensor->width() << ", " << output_tensor->channel() << "]\n"; 
	std::cout << "parse===> output_tensor "  << output_tensor->host<float>() << "\n";
	std::cout << "parse===> output_buffer [" << output_buffer->height() << ", " << output_buffer->width() << ", " << output_buffer->channel() << "]\n"; 
	std::cout << "parse===> output_buffer "  << output_buffer->host<float>() << "\n";
	std::cout << "parse===> matting_mask  "  << (void*)matting_mask.data << "\n";
	std::cout << "parse===> resized to ["    << origin_height << ", " << origin_width << "]\n"; 
	
	// 放缩到原始大小
	cv::resize(matting_mask, matting_mask, {origin_width, origin_height}, 0, 0, cv::INTER_CUBIC);
	// 这一步 * 255 可能是无所谓的消耗
	matting_mask = matting_mask * 255;
	matting_mask.convertTo(matting_mask, CV_8UC1);

	// 保存结果
	cv::imwrite("./demo3_mask.png", matting_mask, {cv::IMWRITE_PNG_COMPRESSION, 0});
	cv_show(matting_mask);

	return 0;
}