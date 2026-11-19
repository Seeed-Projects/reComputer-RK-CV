# Sources

The Web layout and endpoint conventions are adapted from Seeed-Projects/reComputer-RK-CV/src/rk3588_yolov8 (local base commit a64ea10).

YOLO26 six-head decode and model conversion follow airockchip/rknn3-model-zoo at 7f8c7d848448fa14cf16ba359ca98405c8bb94b5. Upstream: https://github.com/airockchip/rknn3-model-zoo/tree/main/examples/yolo26

The included RKNN3 Lite wheel, ARM64 runtime libraries, host tools, PCIe driver source and M.2 firmware are from the user-provided Rockchip RK1820/RK1828 AI SDK 1.1.0. Their original notices and licensing terms apply. The host script does not silently flash firmware.

The YOLO26n model was converted locally using RKNN3 Toolkit 1.1.0 (rk1820, one core, W8A8 with upstream W16A16 subgraphs, 20 COCO calibration images, ONNX opset 19). Weights originate from Ultralytics/assets v8.4.0/yolo26n.pt.

video/test.mp4 is an exact copy of the existing rk3588_yolov8 test video. SHA256SUMS records bundled binary assets.
