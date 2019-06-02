# Real-time-face-recognition
结合MTCNN和facenet进行实时人脸识别

facenet模型的下载地址如下所示：
链接：https://pan.baidu.com/s/18XuUlFgai__srxZ8d95Uhw 
提取码：rzf0 
解压后将模型放在test/face_models文件夹下。

本开源代码主要依托MTCNN开源项目和Facenet开源项目，因此关于模型的训练和前期的环境搭建可以参考以下链接。
MTCNN的源码链接https://github.com/AITTSMD/MTCNN-Tensorflow

facenet的源码链接https://github.com/davidsandberg/facenet

我所做的工作仅仅是将两个模型拼接在一起做了一个实时人脸识别，非常简单清晰的过程，关于识别有两种思路。

（1）将facenet提取出的人脸特征做一个人脸库，然后用检测的人脸与人脸库一一比对找最相近的人脸；优点：可以方便的增加新人物人脸，无需再次训练；缺点：精确度不高，且人脸库较大时，速度慢。

（2）将facenet提取出的人脸特征训练一个分类器，然后用检测到的人脸进行分类。优点：精度较高，速度快；缺点：当增加新人物人脸时，需要再次训练分类器。

我采用的是第二种方式进行搭建这个项目。

项目运行文件：test/realtime.py  此文件包含了提取特征、训练svc、实时检测的功能。
