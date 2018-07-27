
## 问题描述

自适应阈值分割 引发的噪声与破裂


采用edge的方案， 粗笔画的难填充满， 虽然噪声小，不是个很好的方法
也存在破裂笔画

由于书写上的千差万别, 手写数字书写中, 浓淡不
均和小内孔问题几乎是不可避免的, 因此解决笔划断
开ö丢失与小内孔问题在手写数字识别的预处理中是
非常必要的。由于二值化算法本身的无能为力, 可采用
事前处理或事后处理的方法解决。

* 浓淡不均
* 笔画断开
* 小内孔
* 笔画丢失

## 解决方案

### Normalization

Moment-based Image Normalization for Handwritten Text Recognition

![Screenshot_20180302_230841.png](./IMG/Screenshot_20180302_230841.png)

### 事前增强
解决方法主要有两种， 一种是事前增强 ， 另外一种是事后增强。
![Screenshot_20180302_224345.png](./IMG/Screenshot_20180302_224345.png)


### 事后处理

![Screenshot_20180302_224551.png](./IMG/Screenshot_20180302_224551.png)