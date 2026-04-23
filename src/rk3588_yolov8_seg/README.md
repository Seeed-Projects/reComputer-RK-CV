# RK3588 YOLOv8-Seg Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains YOLOv8-Seg instance segmentation inference code optimized for RK3588.

**Special Note**: The segmentation post-processing logic in this project (including bounding box decoding, NMS, mask parsing, scaling, and contour extraction) has been **completely rewritten using pure Numpy and OpenCV**, thoroughly removing the heavy dependencies on `torch` and `torchvision` found in the original code. This significantly reduces memory footprint and improves startup and inference speeds on embedded boards.

## Core Features
- **Hardware Acceleration**: Optimized for RK3588's 6 TOPS NPU architecture.
- **Instance Segmentation**: High-performance mask generation and polygon contour extraction.
- **Flexible Input**: Supports camera and local MP4 video input.

## Directory Structure
- `lib/`: Contains `librknnrt.so` for RK3588.
- `model/`: Stores `.rknn` models converted for RK3588 (e.g., `yolov8n-seg.rknn`).
- `py_utils/`: Inference engine wrapper and lightweight segmentation post-processing utilities.
- `web_detection.py`: Main program (supports Web preview and API).

## Quick Start

### 1. Run the Project (One command, dual-mode preview)

This project supports simultaneous preview via **Local GUI** and **Web Browser**. The program automatically detects the display environment and downgrades to Web mode if no display is connected.

#### Step A: Configure Display Permissions (Optional)
If you have a monitor connected and want to see the window locally:
```bash
xhost +local:docker
```

#### Step B: One-click Run
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-seg:latest \
    python3 web_detection.py --model_path model/yolov8n-seg.rknn --video video/test.mp4
```
Access via: `http://<Board_IP>:8000`

> **Note**: If you need custom classes, you can add `-v $(pwd)/class_config.txt:/app/class_config.txt \` mount and `--class_path` parameter. The program defaults to COCO 80 classes.

Example:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    -v $(pwd)/class_config.txt:/app/class_config.txt \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    rghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-seg:latest \
    python3 web_detection.py --model_path model/yolov8n-seg.rknn --video video/test.mp4 --class_path class_config.txt
```

> **Note**: If you want to test with a local video instead of a camera, use the `--video_path` parameter:
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    -v $(pwd)/video:/app/video \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-seg:latest \
    python3 web_detection.py --model_path model/yolov8n-seg.rknn --video_path video/test.mp4
```

---

## 🔌 API Documentation

This project provides RESTful interfaces compatible with the Ultralytics Cloud API standard, supporting instance segmentation via image, video uploads or direct camera calls.

### 1. Model Inference Interface (Predict)

**Endpoint:** `POST /api/models/yolov8_seg/predict` (or `/api/models/yolo11/predict` depending on exact script mapping)

#### Request Parameters (Multipart/Form-Data):
- `file`: (Optional) Image file to be detected.
- `video`: (Optional) MP4 video file to be detected.
- `timestamp`: (Optional) Timestamp in the video file (seconds), returns detection results for the frame at that point. Default is 0.
- `realtime`: (Optional) Boolean. If `true` or if no `file`/`video` parameters are provided, returns detection results for the current camera frame.
- `conf`: (Optional) Confidence threshold for a single request, range 0.0-1.0.
- `iou`: (Optional) NMS IOU threshold for a single request, range 0.0-1.0.

#### Usage Examples:

**1. Image Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolov8_seg/predict" -F "file=@/home/cat/001.jpg"
```

**2. Video Specific Frame Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolov8_seg/predict" -F "video=@/home/cat/test.mp4" -F "timestamp=5.5"
```

**3. Get Current Camera Frame Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolov8_seg/predict" -F "realtime=true"
```

#### Response Format (JSON):
```json
{
  "success": true,
  "source": "realtime camera frame",
  "predictions": [
    {
      "class": "cup",
      "confidence": 0.9840346574783325,
      "box": { "x1": 100, "y1": 200, "x2": 300, "y2": 500 },
      "polygons": [
        [{"x":558.0,"y":336.0},{"x":558.0,"y":767.0},{"x":560.0,"y":767.0},{"x":561.0,"y":768.0},{"x":561.0,"y":773.0},{"x":563.0,"y":773.0},{"x":564.0,"y":774.0},{"x":564.0,"y":782.0},{"x":566.0,"y":782.0},{"x":567.0,"y":783.0},{"x":567.0,"y":788.0},{"x":569.0,"y":788.0},{"x":570.0,"y":789.0},{"x":570.0,"y":791.0},{"x":572.0,"y":791.0},{"x":573.0,"y":792.0},{"x":573.0,"y":794.0},{"x":575.0,"y":794.0},{"x":576.0,"y":795.0},{"x":576.0,"y":806.0},{"x":578.0,"y":806.0},{"x":579.0,"y":807.0},{"x":579.0,"y":815.0},{"x":581.0,"y":815.0},{"x":582.0,"y":816.0},{"x":582.0,"y":824.0},{"x":584.0,"y":824.0},{"x":585.0,"y":825.0},{"x":585.0,"y":839.0},{"x":587.0,"y":839.0},{"x":588.0,"y":840.0},{"x":588.0,"y":854.0},{"x":590.0,"y":854.0},{"x":591.0,"y":855.0},{"x":591.0,"y":866.0},{"x":593.0,"y":866.0},{"x":594.0,"y":867.0},{"x":594.0,"y":875.0},{"x":596.0,"y":875.0},{"x":597.0,"y":876.0},{"x":597.0,"y":884.0},{"x":599.0,"y":884.0},{"x":600.0,"y":885.0},{"x":600.0,"y":893.0},{"x":602.0,"y":893.0},{"x":603.0,"y":894.0},{"x":603.0,"y":899.0},{"x":605.0,"y":899.0},{"x":606.0,"y":900.0},{"x":606.0,"y":902.0},{"x":608.0,"y":902.0},{"x":609.0,"y":903.0},{"x":609.0,"y":905.0},{"x":611.0,"y":905.0},{"x":612.0,"y":906.0},{"x":612.0,"y":908.0},{"x":614.0,"y":908.0},{"x":615.0,"y":909.0},{"x":615.0,"y":911.0},{"x":620.0,"y":911.0},{"x":621.0,"y":912.0},{"x":621.0,"y":917.0},{"x":623.0,"y":917.0},{"x":624.0,"y":918.0},{"x":624.0,"y":920.0},{"x":626.0,"y":920.0},{"x":627.0,"y":921.0},{"x":627.0,"y":923.0},{"x":629.0,"y":923.0},{"x":630.0,"y":924.0},{"x":630.0,"y":926.0},{"x":632.0,"y":926.0},{"x":633.0,"y":927.0},{"x":633.0,"y":929.0},{"x":635.0,"y":929.0},{"x":636.0,"y":930.0},{"x":636.0,"y":932.0},{"x":641.0,"y":932.0},{"x":642.0,"y":933.0},{"x":642.0,"y":935.0},{"x":644.0,"y":935.0},{"x":645.0,"y":936.0},{"x":645.0,"y":938.0},{"x":650.0,"y":938.0},{"x":651.0,"y":939.0},{"x":651.0,"y":944.0},{"x":656.0,"y":944.0},{"x":657.0,"y":945.0},{"x":657.0,"y":947.0},{"x":662.0,"y":947.0},{"x":663.0,"y":948.0},{"x":663.0,"y":950.0},{"x":665.0,"y":950.0},{"x":666.0,"y":951.0},{"x":666.0,"y":953.0},{"x":671.0,"y":953.0},{"x":672.0,"y":954.0},{"x":672.0,"y":956.0},{"x":677.0,"y":956.0},{"x":678.0,"y":957.0},{"x":678.0,"y":959.0},{"x":686.0,"y":959.0},{"x":687.0,"y":960.0},{"x":687.0,"y":962.0},{"x":695.0,"y":962.0},{"x":696.0,"y":963.0},{"x":696.0,"y":965.0},{"x":701.0,"y":965.0},{"x":702.0,"y":966.0},{"x":702.0,"y":968.0},{"x":713.0,"y":968.0},{"x":714.0,"y":969.0},{"x":714.0,"y":971.0},{"x":869.0,"y":971.0},{"x":869.0,"y":969.0},{"x":870.0,"y":968.0},{"x":884.0,"y":968.0},{"x":884.0,"y":966.0},{"x":885.0,"y":965.0},{"x":893.0,"y":965.0},{"x":893.0,"y":963.0},{"x":894.0,"y":962.0},{"x":899.0,"y":962.0},{"x":899.0,"y":960.0},{"x":900.0,"y":959.0},{"x":911.0,"y":959.0},{"x":911.0,"y":957.0},{"x":912.0,"y":956.0},{"x":917.0,"y":956.0},{"x":917.0,"y":954.0},{"x":918.0,"y":953.0},{"x":923.0,"y":953.0},{"x":923.0,"y":951.0},{"x":924.0,"y":950.0},{"x":929.0,"y":950.0},{"x":929.0,"y":948.0},{"x":930.0,"y":947.0},{"x":935.0,"y":947.0},{"x":935.0,"y":945.0},{"x":936.0,"y":944.0},{"x":941.0,"y":944.0},{"x":941.0,"y":942.0},{"x":942.0,"y":941.0},{"x":947.0,"y":941.0},{"x":947.0,"y":939.0},{"x":948.0,"y":938.0},{"x":953.0,"y":938.0},{"x":953.0,"y":936.0},{"x":954.0,"y":935.0},{"x":962.0,"y":935.0},{"x":962.0,"y":933.0},{"x":963.0,"y":932.0},{"x":968.0,"y":932.0},{"x":968.0,"y":930.0},{"x":969.0,"y":929.0},{"x":971.0,"y":929.0},{"x":971.0,"y":927.0},{"x":972.0,"y":926.0},{"x":977.0,"y":926.0},{"x":977.0,"y":924.0},{"x":978.0,"y":923.0},{"x":983.0,"y":923.0},{"x":983.0,"y":921.0},{"x":984.0,"y":920.0},{"x":989.0,"y":920.0},{"x":989.0,"y":918.0},{"x":990.0,"y":917.0},{"x":992.0,"y":917.0},{"x":992.0,"y":915.0},{"x":993.0,"y":914.0},{"x":998.0,"y":914.0},{"x":998.0,"y":912.0},{"x":999.0,"y":911.0},{"x":1004.0,"y":911.0},{"x":1004.0,"y":909.0},{"x":1005.0,"y":908.0},{"x":1007.0,"y":908.0},{"x":1007.0,"y":906.0},{"x":1008.0,"y":905.0},{"x":1010.0,"y":905.0},{"x":1010.0,"y":903.0},{"x":1011.0,"y":902.0},{"x":1013.0,"y":902.0},{"x":1013.0,"y":900.0},{"x":1014.0,"y":899.0},{"x":1019.0,"y":899.0},{"x":1019.0,"y":897.0},{"x":1020.0,"y":896.0},{"x":1022.0,"y":896.0},{"x":1022.0,"y":894.0},{"x":1023.0,"y":893.0},{"x":1025.0,"y":893.0},{"x":1025.0,"y":891.0},{"x":1026.0,"y":890.0},{"x":1028.0,"y":890.0},{"x":1028.0,"y":888.0},{"x":1029.0,"y":887.0},{"x":1031.0,"y":887.0},{"x":1031.0,"y":885.0},{"x":1032.0,"y":884.0},{"x":1034.0,"y":884.0},{"x":1034.0,"y":882.0},{"x":1035.0,"y":881.0},{"x":1037.0,"y":881.0},{"x":1037.0,"y":879.0},{"x":1038.0,"y":878.0},{"x":1040.0,"y":878.0},{"x":1040.0,"y":876.0},{"x":1041.0,"y":875.0},{"x":1043.0,"y":875.0},{"x":1043.0,"y":873.0},{"x":1044.0,"y":872.0},{"x":1046.0,"y":872.0},{"x":1046.0,"y":867.0},{"x":1047.0,"y":866.0},{"x":1049.0,"y":866.0},{"x":1049.0,"y":864.0},{"x":1050.0,"y":863.0},{"x":1052.0,"y":863.0},{"x":1052.0,"y":861.0},{"x":1053.0,"y":860.0},{"x":1055.0,"y":860.0},{"x":1055.0,"y":858.0},{"x":1056.0,"y":857.0},{"x":1058.0,"y":857.0},{"x":1058.0,"y":852.0},{"x":1059.0,"y":851.0},{"x":1061.0,"y":851.0},{"x":1061.0,"y":849.0},{"x":1062.0,"y":848.0},{"x":1064.0,"y":848.0},{"x":1064.0,"y":846.0},{"x":1065.0,"y":845.0},{"x":1067.0,"y":845.0},{"x":1067.0,"y":840.0},{"x":1068.0,"y":839.0},{"x":1070.0,"y":839.0},{"x":1070.0,"y":837.0},{"x":1071.0,"y":836.0},{"x":1073.0,"y":836.0},{"x":1073.0,"y":834.0},{"x":1074.0,"y":833.0},{"x":1076.0,"y":833.0},{"x":1076.0,"y":831.0},{"x":1077.0,"y":830.0},{"x":1085.0,"y":830.0},{"x":1085.0,"y":828.0},{"x":1086.0,"y":827.0},{"x":1094.0,"y":827.0},{"x":1094.0,"y":825.0},{"x":1095.0,"y":824.0},{"x":1103.0,"y":824.0},{"x":1103.0,"y":822.0},{"x":1104.0,"y":821.0},{"x":1109.0,"y":821.0},{"x":1109.0,"y":819.0},{"x":1110.0,"y":818.0},{"x":1121.0,"y":818.0},{"x":1121.0,"y":816.0},{"x":1122.0,"y":815.0},{"x":1136.0,"y":815.0},{"x":1136.0,"y":813.0},{"x":1137.0,"y":812.0},{"x":1142.0,"y":812.0},{"x":1142.0,"y":810.0},{"x":1143.0,"y":809.0},{"x":1148.0,"y":809.0},{"x":1148.0,"y":807.0},{"x":1149.0,"y":806.0},{"x":1154.0,"y":806.0},{"x":1154.0,"y":804.0},{"x":1155.0,"y":803.0},{"x":1163.0,"y":803.0},{"x":1163.0,"y":801.0},{"x":1164.0,"y":800.0},{"x":1166.0,"y":800.0},{"x":1166.0,"y":798.0},{"x":1167.0,"y":797.0},{"x":1172.0,"y":797.0},{"x":1172.0,"y":795.0},{"x":1173.0,"y":794.0},{"x":1175.0,"y":794.0},{"x":1175.0,"y":792.0},{"x":1176.0,"y":791.0},{"x":1181.0,"y":791.0},{"x":1181.0,"y":789.0},{"x":1182.0,"y":788.0},{"x":1187.0,"y":788.0},{"x":1187.0,"y":786.0},{"x":1188.0,"y":785.0},{"x":1190.0,"y":785.0},{"x":1190.0,"y":783.0},{"x":1191.0,"y":782.0},{"x":1196.0,"y":782.0},{"x":1196.0,"y":780.0},{"x":1197.0,"y":779.0},{"x":1202.0,"y":779.0},{"x":1202.0,"y":777.0},{"x":1203.0,"y":776.0},{"x":1205.0,"y":776.0},{"x":1205.0,"y":774.0},{"x":1206.0,"y":773.0},{"x":1208.0,"y":773.0},{"x":1208.0,"y":771.0},{"x":1209.0,"y":770.0},{"x":1214.0,"y":770.0},{"x":1214.0,"y":768.0},{"x":1215.0,"y":767.0},{"x":1217.0,"y":767.0},{"x":1217.0,"y":765.0},{"x":1218.0,"y":764.0},{"x":1223.0,"y":764.0},{"x":1223.0,"y":762.0},{"x":1224.0,"y":761.0},{"x":1226.0,"y":761.0},{"x":1226.0,"y":759.0},{"x":1227.0,"y":758.0},{"x":1229.0,"y":758.0},{"x":1229.0,"y":756.0},{"x":1230.0,"y":755.0},{"x":1235.0,"y":755.0},{"x":1235.0,"y":753.0},{"x":1236.0,"y":752.0},{"x":1238.0,"y":752.0},{"x":1238.0,"y":750.0},{"x":1239.0,"y":749.0},{"x":1241.0,"y":749.0},{"x":1241.0,"y":747.0},{"x":1242.0,"y":746.0},{"x":1244.0,"y":746.0},{"x":1244.0,"y":744.0},{"x":1245.0,"y":743.0},{"x":1247.0,"y":743.0},{"x":1247.0,"y":741.0},{"x":1248.0,"y":740.0},{"x":1250.0,"y":740.0},{"x":1250.0,"y":738.0},{"x":1251.0,"y":737.0},{"x":1253.0,"y":737.0},{"x":1253.0,"y":732.0},{"x":1254.0,"y":731.0},{"x":1256.0,"y":731.0},{"x":1256.0,"y":726.0},{"x":1257.0,"y":725.0},{"x":1259.0,"y":725.0},{"x":1259.0,"y":720.0},{"x":1260.0,"y":719.0},{"x":1262.0,"y":719.0},{"x":1262.0,"y":714.0},{"x":1263.0,"y":713.0},{"x":1265.0,"y":713.0},{"x":1265.0,"y":711.0},{"x":1266.0,"y":710.0},{"x":1268.0,"y":710.0},{"x":1268.0,"y":705.0},{"x":1269.0,"y":704.0},{"x":1271.0,"y":704.0},{"x":1271.0,"y":696.0},{"x":1272.0,"y":695.0},{"x":1274.0,"y":695.0},{"x":1274.0,"y":690.0},{"x":1275.0,"y":689.0},{"x":1277.0,"y":689.0},{"x":1277.0,"y":687.0},{"x":1278.0,"y":686.0},{"x":1280.0,"y":686.0},{"x":1280.0,"y":681.0},{"x":1281.0,"y":680.0},{"x":1283.0,"y":680.0},{"x":1283.0,"y":675.0},{"x":1284.0,"y":674.0},{"x":1286.0,"y":674.0},{"x":1286.0,"y":663.0},{"x":1287.0,"y":662.0},{"x":1289.0,"y":662.0},{"x":1289.0,"y":657.0},{"x":1290.0,"y":656.0},{"x":1292.0,"y":656.0},{"x":1292.0,"y":579.0},{"x":1290.0,"y":579.0},{"x":1289.0,"y":578.0},{"x":1289.0,"y":567.0},{"x":1287.0,"y":567.0},{"x":1286.0,"y":566.0},{"x":1286.0,"y":555.0},{"x":1284.0,"y":555.0},{"x":1283.0,"y":554.0},{"x":1283.0,"y":549.0},{"x":1281.0,"y":549.0},{"x":1280.0,"y":548.0},{"x":1280.0,"y":543.0},{"x":1278.0,"y":543.0},{"x":1277.0,"y":542.0},{"x":1277.0,"y":537.0},{"x":1275.0,"y":537.0},{"x":1274.0,"y":536.0},{"x":1274.0,"y":531.0},{"x":1272.0,"y":531.0},{"x":1271.0,"y":530.0},{"x":1271.0,"y":525.0},{"x":1269.0,"y":525.0},{"x":1268.0,"y":524.0},{"x":1268.0,"y":519.0},{"x":1266.0,"y":519.0},{"x":1265.0,"y":518.0},{"x":1265.0,"y":516.0},{"x":1263.0,"y":516.0},{"x":1262.0,"y":515.0},{"x":1262.0,"y":513.0},{"x":1260.0,"y":513.0},{"x":1259.0,"y":512.0},{"x":1259.0,"y":507.0},{"x":1257.0,"y":507.0},{"x":1256.0,"y":506.0},{"x":1256.0,"y":501.0},{"x":1254.0,"y":501.0},{"x":1253.0,"y":500.0},{"x":1253.0,"y":498.0},{"x":1251.0,"y":498.0},{"x":1250.0,"y":497.0},{"x":1250.0,"y":495.0},{"x":1248.0,"y":495.0},{"x":1247.0,"y":494.0},{"x":1247.0,"y":489.0},{"x":1245.0,"y":489.0},{"x":1244.0,"y":488.0},{"x":1244.0,"y":486.0},{"x":1242.0,"y":486.0},{"x":1241.0,"y":485.0},{"x":1241.0,"y":483.0},{"x":1239.0,"y":483.0},{"x":1238.0,"y":482.0},{"x":1238.0,"y":480.0},{"x":1236.0,"y":480.0},{"x":1235.0,"y":479.0},{"x":1235.0,"y":477.0},{"x":1233.0,"y":477.0},{"x":1232.0,"y":476.0},{"x":1232.0,"y":474.0},{"x":1230.0,"y":474.0},{"x":1229.0,"y":473.0},{"x":1229.0,"y":471.0},{"x":1227.0,"y":471.0},{"x":1226.0,"y":470.0},{"x":1226.0,"y":468.0},{"x":1224.0,"y":468.0},{"x":1223.0,"y":467.0},{"x":1223.0,"y":465.0},{"x":1221.0,"y":465.0},{"x":1220.0,"y":464.0},{"x":1220.0,"y":462.0},{"x":1218.0,"y":462.0},{"x":1217.0,"y":461.0},{"x":1217.0,"y":459.0},{"x":1212.0,"y":459.0},{"x":1211.0,"y":458.0},{"x":1211.0,"y":456.0},{"x":1209.0,"y":456.0},{"x":1208.0,"y":455.0},{"x":1208.0,"y":453.0},{"x":1206.0,"y":453.0},{"x":1205.0,"y":452.0},{"x":1205.0,"y":450.0},{"x":1203.0,"y":450.0},{"x":1202.0,"y":449.0},{"x":1202.0,"y":447.0},{"x":1197.0,"y":447.0},{"x":1196.0,"y":446.0},{"x":1196.0,"y":444.0},{"x":1191.0,"y":444.0},{"x":1190.0,"y":443.0},{"x":1190.0,"y":441.0},{"x":1188.0,"y":441.0},{"x":1187.0,"y":440.0},{"x":1187.0,"y":438.0},{"x":1185.0,"y":438.0},{"x":1184.0,"y":437.0},{"x":1184.0,"y":435.0},{"x":1182.0,"y":435.0},{"x":1181.0,"y":434.0},{"x":1181.0,"y":432.0},{"x":1176.0,"y":432.0},{"x":1175.0,"y":431.0},{"x":1175.0,"y":429.0},{"x":1173.0,"y":429.0},{"x":1172.0,"y":428.0},{"x":1172.0,"y":426.0},{"x":1170.0,"y":426.0},{"x":1169.0,"y":425.0},{"x":1169.0,"y":423.0},{"x":1164.0,"y":423.0},{"x":1163.0,"y":422.0},{"x":1163.0,"y":420.0},{"x":1155.0,"y":420.0},{"x":1154.0,"y":419.0},{"x":1154.0,"y":417.0},{"x":1149.0,"y":417.0},{"x":1148.0,"y":416.0},{"x":1148.0,"y":414.0},{"x":1143.0,"y":414.0},{"x":1142.0,"y":413.0},{"x":1142.0,"y":411.0},{"x":1137.0,"y":411.0},{"x":1136.0,"y":410.0},{"x":1136.0,"y":408.0},{"x":1128.0,"y":408.0},{"x":1127.0,"y":407.0},{"x":1127.0,"y":405.0},{"x":1122.0,"y":405.0},{"x":1121.0,"y":404.0},{"x":1121.0,"y":402.0},{"x":1116.0,"y":402.0},{"x":1115.0,"y":401.0},{"x":1115.0,"y":399.0},{"x":1110.0,"y":399.0},{"x":1109.0,"y":398.0},{"x":1109.0,"y":396.0},{"x":1101.0,"y":396.0},{"x":1100.0,"y":395.0},{"x":1100.0,"y":393.0},{"x":1095.0,"y":393.0},{"x":1094.0,"y":392.0},{"x":1094.0,"y":390.0},{"x":1092.0,"y":390.0},{"x":1091.0,"y":389.0},{"x":1091.0,"y":387.0},{"x":1089.0,"y":387.0},{"x":1088.0,"y":386.0},{"x":1088.0,"y":384.0},{"x":1083.0,"y":384.0},{"x":1082.0,"y":383.0},{"x":1082.0,"y":381.0},{"x":1080.0,"y":381.0},{"x":1079.0,"y":380.0},{"x":1079.0,"y":375.0},{"x":1074.0,"y":375.0},{"x":1073.0,"y":374.0},{"x":1073.0,"y":372.0},{"x":1065.0,"y":372.0},{"x":1064.0,"y":371.0},{"x":1064.0,"y":369.0},{"x":1059.0,"y":369.0},{"x":1058.0,"y":368.0},{"x":1058.0,"y":366.0},{"x":1050.0,"y":366.0},{"x":1049.0,"y":365.0},{"x":1049.0,"y":363.0},{"x":1032.0,"y":363.0},{"x":1031.0,"y":362.0},{"x":1031.0,"y":360.0},{"x":1002.0,"y":360.0},{"x":1001.0,"y":359.0},{"x":1001.0,"y":357.0},{"x":990.0,"y":357.0},{"x":989.0,"y":356.0},{"x":989.0,"y":354.0},{"x":981.0,"y":354.0},{"x":980.0,"y":353.0},{"x":980.0,"y":351.0},{"x":978.0,"y":351.0},{"x":977.0,"y":350.0},{"x":977.0,"y":348.0},{"x":972.0,"y":348.0},{"x":971.0,"y":347.0},{"x":971.0,"y":345.0},{"x":966.0,"y":345.0},{"x":965.0,"y":344.0},{"x":965.0,"y":342.0},{"x":954.0,"y":342.0},{"x":953.0,"y":341.0},{"x":953.0,"y":339.0},{"x":930.0,"y":339.0},{"x":929.0,"y":338.0},{"x":929.0,"y":336.0}]
      ]
    }
  ],
  "image": { "width": 1280, "height": 720 }
}
```

### 2. System Configuration Interface (Config)

Used to dynamically adjust thresholds for real-time video streams and default inference.

#### Get Current Configuration
- **Endpoint:** `GET /api/config`
- **Response:** `{"obj_thresh": 0.25, "nms_thresh": 0.45}`

#### Update System Configuration
- **Endpoint:** `POST /api/config`
- **Request Body (JSON):** `{"obj_thresh": 0.3, "nms_thresh": 0.5}`
- **Response:** `{"status": "success"}`

### 3. Real-time Video Stream Interface (Video Feed)

Get real-time MJPEG video stream with detection boxes and segmentation masks drawn, can be directly embedded in HTML `<img>` tags.

- **Endpoint:** `GET /api/video_feed`
- **Example Usage:** `<img src="http://<Board_IP>:8000/api/video_feed">`

---

## 🛠️ Developer Guide (Production Recommendations)
### Code Description
- `web_detection.py`:
    - **Dual-mode Support**: Integrates FastAPI, supporting both local rendering and MJPEG streaming output.
    - **Environment Adaptive**: Automatically detects the `DISPLAY` environment variable, silently skipping GUI initialization if not present.
    - **RKNN Inference**: Encapsulates RKNN initialization, model loading, and multi-core inference logic.
    - **Dynamic Loading**: Supports dynamic class configuration loading via `--class_path`.
    - **Post-processing**: High-performance Numpy-based bounding box decoding, NMS, and contour extraction (`cv2.findContours`).

### Modifying Models
1. Place the trained and converted .rknn model into the `model/` directory.
2. Add the `--model_path` argument to the running command to point to the new model.
