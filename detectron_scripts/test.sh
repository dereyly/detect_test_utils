export PYTHONPATH=$PYTHONPATH:/home/dereyly/progs/Detectron/lib:/home/dereyly/progs/caffe2/build
detectron_path=/home/dereyly/progs/Detectron
python2 $detectron_path/tools/test_net.py --cfg configs/e2e_food_R-50-FPN_2x.yaml    TEST.WEIGHTS train_out/train/food_train/generalized_rcnn/model_final.pkl