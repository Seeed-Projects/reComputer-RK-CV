Web layout and API conventions follow Seeed-Projects/reComputer-RK-CV.

Task decoding and model export follow airockchip/rknn3-model-zoo at 7f8c7d848448fa14cf16ba359ca98405c8bb94b5, examples/yolo26. See LICENSE.model-zoo. Original model licenses apply; YOLO weights originate from Ultralytics.

The included RKNN3 Lite wheel, ARM64 libraries, host tools, driver and M.2 firmware are from the user-provided RK1820/RK1828 AI SDK 1.1.0; their original notices and licensing terms apply.

The bundled .rknn models were converted with RKNN3 Toolkit 1.1.0, W8A8, quantization enabled, without manually assigned W16A16 subgraphs. External floating-point inputs/intermediates or compiler-managed floating-point operators are not a claim of an integer-only graph.

video/test.mp4 is the user-supplied 30_1080p.mp4 traffic video. RAFT uses the official paired images instead.
