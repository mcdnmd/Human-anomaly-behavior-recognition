install:
	@poetry install
	@wget -P data https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
