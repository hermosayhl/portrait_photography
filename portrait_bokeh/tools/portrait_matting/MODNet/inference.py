import os
import sys
import cv2
import numpy
import onnxruntime


def cv_show(image):
	cv2.imshow("crane", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 加载模型
task = onnxruntime.InferenceSession(
	"modnet_photographic_portrait_matting.onnx",
	providers=["CPUExecutionProvider"])

# 读取图像
image_path = "./demo3.png"
image = cv2.imread(image_path)
height, width, _ = image.shape

# 做前处理
def convert_to_tensor(x, clip=32):
	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x = x.astype("float32")
	x = (x - 127.5) / 127.5
	# resize
	new_height = x.shape[0] - x.shape[0] % clip
	new_width  = x.shape[1] - x.shape[1] % clip
	if (new_height != x.shape[0] or new_width != x.shape[1]):
		print("do resizing from {} to {}".format((x.shape[0], x.shape[1]), (new_height, new_width)))
		x = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_AREA)
	x = numpy.ascontiguousarray(x.transpose(2, 0, 1))
	x = numpy.expand_dims(x, axis=0)
	return x, new_height, new_width

# numpy → tensor
image_tensor, new_height, new_width = convert_to_tensor(image)
print("image_tensor  ", image_tensor.shape)

# 推理
[matting_result] = task.run(["output"], {"input": image_tensor})
print("matting_result   ", matting_result.shape, matting_result.min(), matting_result.max())



# 后处理
matting_result = numpy.squeeze(matting_result)
if (new_height != image.shape[0] or new_width != image.shape[1]):
	matting_result = cv2.resize(matting_result, (width, height), interpolation=cv2.INTER_AREA)
	print("resize the matting result back to {}".format((height, width)))

# 可以锐化一下
# matting_result = numpy.where(matting_result < 0.5, 0, 1)

# 展示
matting_result = (matting_result * 255).astype("uint8")
cv2.imwrite(image_path.replace(".png", "_output.png"), matting_result)
cv_show(matting_result)