cfg/coco.data             - главный конфиг
cfg/yolov3.cfg            - конфиг архитектуры
data/my_data/train.txt    - пути к картинкам
data/my_data/classes.txt  - имена классов


--- run detector ---
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0.5
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -ext_output data/dog.jpg -thresh 0.5

--- train ---
./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74

--- get coordintes ---
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -ext_output < images_files.txt 2>/dev/null
./darknet detector test cfg/coco.data cfg/yolov3-spp.cfg yolov3-spp.backup -ext_output < images_files.txt 2>/dev/null

 --- usefull flags ---
 -thresh 0.1
