# resnet50-
resnet50迁移学习训练自己的垃圾分类数据集<br>
制作：bilibili:小白菜2233<br>
内容：基于resnet50的垃圾分类系统，对'其他垃圾', '厨余垃圾', '可回收物', '有害垃圾'这四种进行分类<br>
界面：采用PyQt5设计，实现简单的加载图片和识别功能<br>
框架：基于tensorflow2.1 python3.7<br>
模型训练：GPU 英伟达2070max-Q 8G<br>
代码：上传github：https://github.com/qq1308636759/resnet50-<br>
数据集和模型文件太大，上传不了github，数据集可以自己找找。<br>
其中，resnet.py是模型训练文件<br>
UI.py是界面文件<br>
main.py是把代码集合成的函数方便界面调用的主文件。<br>
1.预测精度<br>
  ![Image text]( https://github.com/qq1308636759/resnet50-/blob/main/QQ%E6%88%AA%E5%9B%BE20220126235606.jpg)<br>
2.混淆矩阵<br>
![Image text](https://github.com/qq1308636759/resnet50-/blob/main/QQ%E6%88%AA%E5%9B%BE20220126235618.jpg)<br>
3.效果展示<br>
![Image text](https://github.com/qq1308636759/resnet50-/blob/main/QQ%E6%88%AA%E5%9B%BE20220127000326.jpg)<br>
