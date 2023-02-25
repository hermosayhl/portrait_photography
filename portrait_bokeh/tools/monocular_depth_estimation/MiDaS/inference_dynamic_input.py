import os
import cv2
import numpy
import onnxruntime


def cv_show(image):
	cv2.imshow("crane", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def depth_to_image(depth, grayscale=False, bits=2):
	if not numpy.isfinite(depth).all():
		depth = numpy.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
		print("WARNING: Non-finite depth values present")

	depth_min = depth.min()
	depth_max = depth.max()

	if (not grayscale):
		bits = 1
	max_val = (2 ** (8 * bits)) - 1

	if depth_max - depth_min > numpy.finfo("float").eps:
		out = max_val * (depth - depth_min) / (depth_max - depth_min)
	else:
		out = numpy.zeros(depth.shape, dtype=depth.dtype)

	if not grayscale:
		out = cv2.applyColorMap(numpy.uint8(out), cv2.COLORMAP_INFERNO)

	return out.astype("uint8") if (bits == 1) else out.astype("uint16")




# python -m onnxsim MiDas-tiny.onnx MiDas-tiny-sim.onnx
# 加载 onnx 模型
task = onnxruntime.InferenceSession(
	"./MiDas-tiny-preproc-dynamic_input-sim.onnx",
	providers=["CPUExecutionProvider"])


# 读取图像
image_path = "./demo2.png"
image = cv2.imread(image_path)
height, width, _ = image.shape

# numpy -> tensor
image_tensor = numpy.expand_dims(image, axis=0)
print("image_tensor  ", image_tensor.shape)

# 推理
[depth_estimation] = task.run(["depth"], {"monocular_image": image_tensor})
print("depth_estimation  ", depth_estimation.shape, depth_estimation.min(), depth_estimation.max(), depth_estimation.mean(), depth_estimation.std())
# depth_estimation   (1, 256, 256) 0.0 2269.7942 1175.3344 818.6526

# 放缩到原始大小
depth_estimation = cv2.resize(depth_estimation[0], (width, height), interpolation=cv2.INTER_CUBIC)
print("depth_estimation  ", depth_estimation.shape)

# 保存
# numpy.save("./onnx_result.npy", depth_estimation)

# 可视化
depth_visualize = depth_to_image(depth_estimation, False)
cv2.imwrite(image_path.replace(".png", "_depth.png"), depth_visualize)
cv_show(depth_visualize)
