
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_CONVERTER=true -DCMAKE_INSTALL_PREFIX=E:/environments/C++/mnn/MNN/install_gcc

MNNConvert --framework ONNX --modelFile modnet_portrait_matting_dynamic-sim.onnx --MNNModel MODNet-static-sim.mnn --bizCode MNN