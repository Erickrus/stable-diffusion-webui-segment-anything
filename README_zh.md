stable-diffusion-webui 万能图像分割扩展

## 安装

### 安装依赖包
```shell
pip3 install opencv-python matplotlib onnx onnxruntime
pip3 install 'git+https://github.com/facebookresearch/segment-anything.git'
```

### 安装扩展

转到 `Extensions` Tab

点击 `Install from URL`

输入 `https://github.com/Erickrus/stable-diffusion-webui-segment-anything`

点击 `Install`

脚本安装完成后，请重启webui




## 使用

将你的图片上传到左侧图像栏，然后在左侧图像上绘制/单击小圆点，然后单击Segment按钮。

请注意画笔半径请控制在 **5** 左右. 不要使用太大的半径，会影响最终分割点搜索和最终效果。

![](about.png)
