### Vision



classification, object detection, segmentation

ImageNet 的重要贡献是基于WordNet建立的图片语义结构，Deep Learning 是Representation learning, 在图片高维空间里面建立更好的表征，使得这些表征对语义标定有更好的区分和映射。图片的语义标定衍生出很多应用，比如Entry-level recognition 和 Zero-shot learning。

Entry-level recognition主要是分析wordnet上的synset到entry-level description 的关系

Zero-shot learning 解决如果某个类别没有任何训练图片数据，如何去识别这个类别。大致做法是利用当前没有任何图片数据的标定与之前有图片数据的标定的语义相似度，来建立语义标定之间的关联。

### Vision-and-Language

#### 经典任务

Image Captioning, VQA

**Image Captioning** 

给任意一张图，系统可以输出语句来描述这幅画里的内容

1. Encoder-Decoder

Show and Tell: A Neural Image Caption Generator， CVPR 2015

Deep Visual-Semantic Alignments for Generating Image Descriptions，CVPR 2015

2. Attention 

Show, Attend and Tell: Neural Image Caption Generation with Visual Attention， ICML 2015 (Spatial attention)

SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning, CVPR 2017 (channel wise attention)

3. Attributes (跟图片相关的标签或属性)

What Value Do Explicit High Level Concepts Have in Vision to Language Problems ? CVPR 2016

MCB

Modular network

Bottom-up attention

Bottom-Up and Top-Down Attention for Image Captioning, CVPR 2018 (object detection)

Entity Recognition （识别出形容词、名词） 

Dense Caption （多个语句）

Reinforcement Learning

Self-critical Sequence Training for Image Captioning，CVPR 2017

其它（对抗攻击，诗歌）

Attacking Visual Language Grounding with Adversarial Examples: A Case Study on Neural Image Captioning， ACL 2018

Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training

MS COCO Leader Board

**Visual Storying** http://visionandlanguage.net/VIST/

**VQA**  https://visualqa.org/

Definition: An image and a free-form, open-ended question about the image are presented to the method which is required to produce a suitable answer.

VQA 需要更好地理解图像内容并进行一定的推理，有时甚至还需要借助外部的知识库，VQA的评估方法更简单，因为答案往往是客观并简短的，很容易与ground truth对比判断是否准确，不像Captioning需要对长句子做评估。

1. Joint embedding approaches

​    Joint embedding是处理多模态问题时的经典思路

![img](https://pic2.zhimg.com/80/v2-de40ab52f5dc9f86f59aecfb94de91b9_hd.jpg)

2. Attention mechanisms

![img](https://pic2.zhimg.com/80/v2-022fe7e08af04a57ebcb9e6788fbe335_hd.jpg)

  Stack Attention Network

  Visual-Question Co-Attention

  Multi-level Attention

3. Compositional Models

   核心思想时设计一种模块化的模型

   Neural Module Networks，CVPR 2016

   Ask Me Anything: Dynamic Memory Networks for Natural Language Processing, ICML 2016

4. VQA with external knowledge base

   Ask Me Anything: Free-form Visual Question Answering Based on Knowledge from External Sources, CVPR 2016

5. VQA with Reasoning
6. VQA with Explicit Reasoning on Knowledge Base

#### 17年

**Visual grounding（Referring Expression)， Referring Segmentation**

即给出一个句子，在图像上标注出对应区域（标注出mask），或者在视频上定位出对应片段

MAttNet: Modular Attention Network for Referring Expression Comprehension

A Real-Time Cross-modality Correlation Filtering Method for Referring Expression Comprehension

**Visual Dialog**

与 VQA 只有一轮问答不同的是，视觉对话需要机器能够使用自然的，常用的语言和人类维持一个关于图像的，有意义的对话。与 VQA 另外一个不同的地方在于，VQA 的回答普遍都很简短，比如说答案是 yes/no, 数字或者一个名词等等，都偏机器化。而我们希望 visual dialog 能够尽量的生成偏人性化的数据

Visual Dialog with GAN

![img](https://mmbiz.qpic.cn/mmbiz_png/vJe7ErxcLmhiaaH851kia8Nx7sIIXudyw7iapwx8WbPe6KyMYTSXqmKmIbIbzOrXicpkDn5CwoCTvyAwqyT4DibdOaA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

#### 18年 Vision-and-Language 2.0

**Novel Object Captioning**

nocaps novel object captioning at scale

![img](https://mmbiz.qpic.cn/mmbiz_png/vJe7ErxcLmhiaaH851kia8Nx7sIIXudyw7411eLMzql2Ol5UumDCbRQ7ib75ibuBRHaC0UwBZfbJ71vRcPFnk5XpiaQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

2018_CVPR Neural Baby Talk 模板生成，Filling in the slots

2018_NIPS Partially-Supervised Image Captioning constrained beam search

**Visual Reasoning**

CLEVR 第一个专门针对视觉推理任务建立的数据集 2017_CVPR CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning

MAC 提供了一种全可微的模块式的推理结构 2018_ICLR Compositional attention networks for machine reasoning

GQA 可以看作是 CLEVR 的一个真实图像的版本 2019_CVPR GQA: a new dataset for compositional question answering over real world images

**Visual Commonsense Reasoning**, 19年CVPR新出的任务，在选择正确答案的同时，还需要选择出给出这个答案的原因 2019_CVPR From Recognition to Cognition Visual Commonsense Reasoning

**Embodied Vision-and-Language** 将vision-language 和action结合

**embodied VQA** 这个任务是融合多模态信息，通过向放置在一个虚拟环境中的 agent 提出基于文本的问题，需要其在虚拟的空间环境中进行路径规划（Navigation）和探索，以到达目标位置并回答问题。

2018_CVPR Embodied Question Answering

2019_CVPR Multi-Target Embodied Question Answering

**Vision-and-Language Navigation**,18年CVPR提出，这就需要模型对语言和图像同时进行理解，把语言当中描述的位置以及关键点，定位到真实场景图像当中，然后执行相对应的动作。

2018_CVPR Vision-and-Language Navigation Interpreting visually-grounded navigation instructions in real environments

Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation

**Remote Embodied Referring Expressions**，将 navigation 和 referring expression 相结合的一个任务，将 agent 放置于场景中的一个起始点，给定精炼的 navigation guidance，同时包含了两个任务，一个是导航到目的地，一个找到所描述的对应的物品。

2019_CVPR REVERIE Remote Embodied Visual Referring Expression in Real Indoor Environments

![img](https://mmbiz.qpic.cn/mmbiz_png/vJe7ErxcLmhiaaH851kia8Nx7sIIXudyw7pMiaETSfoukEOTVhR9omg5kRqgXf7yJa8w3pgicFYZkAVDzwOoVOaU6A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**Visual/Video BERT**   VQA最近好像被VisualBERT一类的方法刷得很高

VideoBERT: A Joint Model for Video and Language Representation Learning

ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks

VisualBERT: A Simple and Performant Baseline for Vision and Language

Fusion of Detected Objects in Text for Visual Question Answering

Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training

LXMERT: Learning Cross-Modality Encoder Representations from Transformers

VL-BERT: Pre-training of Generic Visual-Linguistic Representations

#### Language to Vision

​    **Language Guided Image Editing，Text-to-Image Synthesis** Language+GAN

#### Lifeifei 

Visual Genome Dataset 让计算机视觉更好地跟自然语言处理里的知识库和语义结构更进一步融合起来

#### Cross modal retrieval

   image-text matching/embedding



2019_ICCV Language Features Matter: Effective Language Representations for Vision-Language Tasks



### Video Understanding

图像（VGG, GoogLeNet, ResNet）， 动态信息（C3D、 IDT, Optical Flow）；帧流，trimmed video clip, 音频流，字符流，video category

video区别于image的本质在于，video有temporal context

video里面含有一些causal signal,就是事件的因果关系和演化过程, 有助于 unsupervised 地 学出比如disentangled representation之类语义层面的东西

#### Classification(action recognition)

C3D和two-stream, C3D 采用 3d kernel导致参数比较多, 模型深度不够; two stream，网络本身并没有对temporal建模，而是利用optical flow

#### Temporal action detection (temporal activity detection, event detection)

Offline and Online

frame-based和proposal-based，frame-based 就是先针对每个frame （或者是snippet，比如6个frame）做分类，然后再在temporal上做grouping；proposal-based的就是先生成proposal（action or background），再针对proposal做detection

spatio-temporal detection 是要生成一个tube（而不是一个bounding box）

#### Video semantic segmentation(video scene pharsing)



#### Video Captioning

Video Captioning with Attributes

Video Captioning with Transferred Semantic Attributes, CVPR 2017

Microsoft Video to Language Challenge

#### Video QA

#### Grounding Actions and Objects by Language in Videos

Temporal Activity Localization by Language，给定一个query（包含对activity的描述），找到对应动作（事件）的起止时间；Spatio-temporal object referring by language， 给定一个query（包含对object/person的描述），在时空中找到连续的bounding box (也就是一个tube)。

#### Video-based Person ReID 

#### tracking

### 高级计算机视觉任务局限的反思

1. 特征缺少足够的属性信息。和传统的hand-crafted的特征提取相比，深度学习的好处是可以学到更多和问题直接相关的feature信息，但是缺点是，如果一旦所需属性和supervise的tag不相关，特征就很可能对一些visual object的属性无关。最明显的例子就是deep cnn良好的旋转不变性，优势的同时也导致了失去物体的姿态信息。关于这部分，我觉得hinton的capsule network就在试图解决这类问题。比如VQA里一个失败样例，问风扇朝上还是朝下，visual attention都attend对了，回答就是错了。
2. 成也attention，败也attention。attention在对各种CV/NLP领域任务有显著提升的同时，也导致了一些high-task的新问题，比如（1）缺少不选择机制，如果遇到不是直接相关的东西，最好的情况是attention会均匀分散到所有地方，最差的情况是会attend到一个很奇怪的东西。（2）造成counting问题，如果图上有两只猫，理想情况attention会分给各自0.5，这样softmax sum了之后，就和一个猫的feature完全一样，导致无法数出有两只猫。
3. 被低估的Relationship。去年刚开始做scene graph/visual relationship的时候不明觉厉，真正做了一段时间尤其是visualize了result之后开始非常嫌弃，最后做VQA的错误样本分析时才开始重新审视它的重要性。relationship是现在很多High-Level task的瓶颈所在，底层的object proposal什么其实已经非常好了（在我看了500多个bottom-up VQA的错误样本后认为，至少相对而言object proposal已经不是瓶颈了）。relationship的重要性在需要visual reasoning的VQA和Visual Grounding上尤为明显。视觉推理最重要的一点就是状态的转移，这种转移往往是通过visual object之间的relation的形式，eg，被人握住的瓶子，那条狗边上的人。但现在Visual Relationship最大的问题有三：（1）现有VG数据库标记太差（2）relationship的种类数量太庞大，不像单纯的物体分类，relationship几乎很难通过标记穷尽一张图上的所有relation，理论上所有object pair有relation，尤其是空间位置关系，任何同一图上一对物体之间必定存在。尝试unsupervised方式？（3）multi-label，两两物体之间的relation不唯一，人可以同时牵着狗/在狗旁边/看着狗/。现在的visual relationship/scene graph框架主流还是单分类。
4. 推理能力。目前computer vision整体来说其实并没有足够的推理能力。Everything is pattern matching。High-Level CV Tasks里定义的最好的，个人觉得是Image Captioning，因为Captioning的本质是pattern matching，不需要额外的推理。个人觉得（1）人类的推理是离散的过程（2）人类的推理不一定是最优路径最少步骤，这样看反而RL可能才是推理的正确学习方式。



### Reference

从 Vision 到 Language 再到 Action，万字漫谈三年跨域信息融合研究

https://mp.weixin.qq.com/s?__biz=MzI5NTIxNTg0OA==&mid=2247490890&idx=1&sn=a04799e6dbeece7af7129af811964975&scene=21#wechat_redirect

一文纵览 Vision-and-Language 领域最新研究与进展

https://mp.weixin.qq.com/s/dyY64QrvPWbjGvJw5H51OA

1. VALSE2017系列之七：视觉与语言领域年度进展概述 https://mp.weixin.qq.com/s/xNz8YUX2XfShPh_Kp0CYEQ?

2. 梅涛：“看图说话”——人类走开，我AI来！|VALSE2017之十二 https://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650327847&idx=1&sn=26cef37ab1331c51f1e62b50b141d655&scene=21#wechat_redirect

4. 让机器“答问如流”：从视觉到语言|VALSE2018之四 https://mp.weixin.qq.com/s/_WOzunt9_I-tDZ1NaCguyA

5. 让机器“察言作画”：从语言到视觉|VALSE2018之二 https://mp.weixin.qq.com/s/d4N-Xz0NImb5khLup5DmpQ

6. 更有智慧的眼睛：图像描述 https://zhuanlan.zhihu.com/p/52499758

7. 最前沿：视觉推理（Visual Reasoning），神经网络也可以有逻辑 https://zhuanlan.zhihu.com/p/28654835

8. 文本+视觉，多篇 Visual/Video BERT 论文介绍 https://zhuanlan.zhihu.com/p/80483517

9. VQA相关方法的简单综述 https://zhuanlan.zhihu.com/p/59530688

10. Visual Question Answering 简介 + 近年文章  https://zhuanlan.zhihu.com/p/57207832

11. 2019·关于高级计算机视觉任务局限的反思 https://zhuanlan.zhihu.com/p/60418025

12. Video Understanding 新方向介绍：Grounding Activities and Objects by Language in Videos

13. Video Online Action Detection & Anticipation 梳理与探讨 https://zhuanlan.zhihu.com/p/35730675

14. Video-based Person ReID的时序建模 https://zhuanlan.zhihu.com/p/36395908

    