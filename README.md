# my_deep_project

项目地址：[https://github.com/audier/my_deep_project](https://github.com/audier/my_deep_project)
最近准备整理一些使用深度模型实现的项目，作为工作学习的一个整理。

## 0. 深度学习项目规范
 我认为一个完整的深度学习项目应该包含如下所示几个方面：
1. **项目背景**：项目背景是什么，完成什么任务？
2. **项目数据**：项目数据是如何获取的，数据都包含什么，输入什么输出什么？
4. **数据处理**：将获得的数据数字化，处理为能够喂进模型的形式。也包括：数据增强、去除脏数据等
5. **模型选择与建模**：核心，也是深度学习框架要实现的主体
6. **评估准则与效果**：评价模型好坏的标准是什么，如何通过评价标准评估模型效果
7. **模型优化与提升**：哪些方面还可以提升模型的性能？

按照这个结构处理深度学习的任务，会更加规范更易提升，后续我将严格按照这个流程执行自己的项目以及整理回溯。
 
## 1. 深度学习入门项目 
文档地址：https://blog.csdn.net/chinatelecom08/article/details/83413623#_5
项目地址：https://github.com/audier/my_deep_project/tree/master/basic_deep_model
1.	**TensorFlow实现mnist分类**
	- DNN示例
	- CNN示例
	- RNN示例
2.	**keras实现mnist分类**
	- DNN示例
	- CNN示例
	- RNN示例

## 2. 自然语言处理
1. **文章自动生成**
文档地址：https://blog.csdn.net/chinatelecom08/article/details/83654602
项目地址：https://github.com/audier/my_deep_project/tree/master/NLP/1.moyan_novel
	- lstm : tensorflow
	- lstm : keras
2. **翻译系统**
文档地址：https://blog.csdn.net/chinatelecom08/article/details/83860179
项目地址：https://github.com/audier/my_deep_project/tree/master/NLP/2.translation
	- seq2seq (tensorflow)
	- seq2seq + attention (tensorflow)
4. **对话系统**
	- seq2seq +attention (keras)
5. **输入法系统**
	- CBHG (tensorflow)
	- CBHG (keras)

## 3. 语音识别 
文档地址：https://blog.csdn.net/chinatelecom08/article/details/82557715
项目地址：https://github.com/audier/my_ch_speech_recognition
1. CTC + RNN
2. CTC + CNN 
3. seq2seq +attention (keras)

## 4. 图像识别
1. 目标检测
2. 风格迁移
3. 文本生成
## 5. GAN
1. mnist图像生成
	- TensorFlow
	- keras


这是给自己立的一个flag，当这些都完成后，希望能够更好的理解这些深度框架，以及一些细节理论。
当然，也有很大的可能完不成这些任务，图像相关的任务自己也只做过mnist，其他的也不是很了解。但是不管怎么说，还是希望能够把列出来的这些任务都能做一遍，更好的理解深度模型在这些任务中是如何发挥作用的。加油。
