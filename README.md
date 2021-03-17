# 76、约束塑性储备作为神经网络频率和权值控制的一种自然方法
- [ ] Constrained plasticity reserve as a natural way to control frequency and weights in spiking neural networks 
时间：2021年03月15日                         第一作者：Oleg Nikitin                        [链接](https://arxiv.org/abs/2103.08143).                     
## 摘要：生物神经元具有自适应性，并进行复杂的计算，包括过滤冗余信息。这种处理通常与贝叶斯推理相联系。然而，最常见的神经细胞模型，包括生物学上合理的模型，如霍奇金-赫胥黎或伊兹克维奇，在单个细胞水平上并不具备预测动力学。现代的突触可塑性或互联适应规则也不能为神经元适应不断变化的输入信号强度的能力提供基础。虽然自然神经元突触生长受到蛋白质供应和循环的精确控制和限制，但广泛使用的STDP等重量校正规则在变化率和范围上是有效的无限制的。在这篇文章中，我们将介绍一种新的机制，通过抽象蛋白质储备限制的STDP生长，并通过细胞内优化算法来控制神经元放电率稳态和重量变化之间的联系。我们将展示，这些细胞动力学如何帮助神经元过滤强烈的信号，帮助神经元保持稳定的放电频率。我们还将检验这种滤波不影响神经元在无监督模式下识别相关输入的能力。这种方法可用于机器学习领域，以提高人工智能系统的鲁棒性。
<details>	<summary>英文摘要</summary>	Biological neurons have adaptive nature and perform complex computations involving the filtering of redundant information. Such processing is often associated with Bayesian inference. Yet most common models of neural cells, including biologically plausible, such as Hodgkin-Huxley or Izhikevich do not possess predictive dynamics on the level of a single cell. The modern rules of synaptic plasticity or interconnections weights adaptation also do not provide grounding for the ability of neurons to adapt to the ever-changing input signal intensity. While natural neuron synaptic growth is precisely controlled and restricted by protein supply and recycling, weight correction rules such as widely used STDP are efficiently unlimited in change rate and scale. In the present article, we will introduce new mechanics of interconnection between neuron firing rate homeostasis and weight change by means of STDP growth bounded by abstract protein reserve, controlled by the intracellular optimization algorithm. We will show, how these cellular dynamics help neurons to filter out the intense signals to help neurons keep a stable firing rate. We will also examine that such filtering does not affect the ability of neurons to recognize the correlated inputs in unsupervised mode. Such an approach might be used in the machine learning domain to improve the robustness of AI systems. </details>
<details>	<summary>注释</summary>	24 pages, 12 figures MSC-class: 68T05 ACM-class: I.2.6 </details>
<details>	<summary>邮件日期</summary>	2021年03月16日</details>

# 75、事件摄影机的时间顺序最近事件（TORE）卷
- [ ] Time-Ordered Recent Event (TORE) Volumes for Event Cameras 
时间：2021年03月10日                         第一作者：R. Wes Baldwin                       [链接](https://arxiv.org/abs/2103.06108).                     
## 摘要：事件摄像机是一种令人兴奋的新型传感器，能够以极低的延迟和宽的动态范围实现高速成像。不幸的是，大多数机器学习体系结构都不是直接处理稀疏数据的，比如从事件摄像机生成的数据。许多最先进的事件摄像机算法依赖于内插事件表示法——模糊了关键的定时信息，增加了数据量，限制了整体网络性能。本文详细介绍了一种称为时间顺序最近事件（TORE）卷的事件表示。存储卷被设计成以最小的信息丢失紧凑地存储原始峰值定时信息。这种仿生设计内存效率高，计算速度快，避免时间阻塞（即固定和预定义的帧速率），并包含来自过去数据的“本地内存”。该设计在一系列具有挑战性的任务（如事件去噪、图像重建、分类和人体姿态估计）上进行了评估，并显示出显著提高了最先进的性能。存储卷是当前使用事件表示的任何算法的易于实现的替代品。
<details>	<summary>英文摘要</summary>	Event cameras are an exciting, new sensor modality enabling high-speed imaging with extremely low-latency and wide dynamic range. Unfortunately, most machine learning architectures are not designed to directly handle sparse data, like that generated from event cameras. Many state-of-the-art algorithms for event cameras rely on interpolated event representations - obscuring crucial timing information, increasing the data volume, and limiting overall network performance. This paper details an event representation called Time-Ordered Recent Event (TORE) volumes. TORE volumes are designed to compactly store raw spike timing information with minimal information loss. This bio-inspired design is memory efficient, computationally fast, avoids time-blocking (i.e. fixed and predefined frame rates), and contains "local memory" from past data. The design is evaluated on a wide range of challenging tasks (e.g. event denoising, image reconstruction, classification, and human pose estimation) and is shown to dramatically improve state-of-the-art performance. TORE volumes are an easy-to-implement replacement for any algorithm currently utilizing event representations. </details>
<details>	<summary>邮件日期</summary>	2021年03月11日</details>

# 74、具有耐久性的脉冲神经网络到神经形态硬件的映射
- [ ] Endurance-Aware Mapping of Spiking Neural Networks to Neuromorphic Hardware 
时间：2021年03月09日                         第一作者：Twisha Titirsha                       [链接](https://arxiv.org/abs/2103.05707).                     
## 摘要：神经形态计算系统正在采用忆阻器来实现高密度和低功耗的突触存储，如硬件中的交叉阵列。这些系统在执行脉冲神经网络（SNNs）时是节能的。我们观察到记忆型纵横制中的长位线和字线是寄生电压降的主要来源，寄生电压降会造成电流不对称。通过电路仿真，我们发现了这种不对称性导致的显著耐久性变化。因此，如果临界忆阻器（耐久性较低的忆阻器）被过度利用，它们可能会导致交叉杆寿命的缩短。我们提出eSpine，这是一种新的技术，通过在映射机器学习工作负载的每个纵横杆中加入耐久性变化来提高寿命，确保具有更高激活的突触总是在具有更高耐久性的忆阻器上实现，反之亦然。eSpine分两步工作。首先，它使用Kernighan-Lin图划分算法将工作负载划分为神经元和突触的簇，每个簇可以放在一个横杆中。第二，利用粒子群优化算法（PSO）的一个实例将簇映射到分片上，通过分析簇在工作负载中的激活情况，将簇的突触放置到十字杆的忆阻器上。我们评估了一个国家的最先进的神经形态硬件模型与相变记忆（PCM）为基础的忆阻器eSpine。使用10个SNN工作负载，我们证明了有效生存期的显著改进。
<details>	<summary>英文摘要</summary>	Neuromorphic computing systems are embracing memristors to implement high density and low power synaptic storage as crossbar arrays in hardware. These systems are energy efficient in executing Spiking Neural Networks (SNNs). We observe that long bitlines and wordlines in a memristive crossbar are a major source of parasitic voltage drops, which create current asymmetry. Through circuit simulations, we show the significant endurance variation that results from this asymmetry. Therefore, if the critical memristors (ones with lower endurance) are overutilized, they may lead to a reduction of the crossbar's lifetime. We propose eSpine, a novel technique to improve lifetime by incorporating the endurance variation within each crossbar in mapping machine learning workloads, ensuring that synapses with higher activation are always implemented on memristors with higher endurance, and vice versa. eSpine works in two steps. First, it uses the Kernighan-Lin Graph Partitioning algorithm to partition a workload into clusters of neurons and synapses, where each cluster can fit in a crossbar. Second, it uses an instance of Particle Swarm Optimization (PSO) to map clusters to tiles, where the placement of synapses of a cluster to memristors of a crossbar is performed by analyzing their activation within the workload. We evaluate eSpine for a state-of-the-art neuromorphic hardware model with phase-change memory (PCM)-based memristors. Using 10 SNN workloads, we demonstrate a significant improvement in the effective lifetime. </details>
<details>	<summary>注释</summary>	Accepted for publication in IEEE Transactions on Parallel and Distributed Systems (TPDS) </details>
<details>	<summary>邮件日期</summary>	2021年03月11日</details>

# 73、一种用于无监督特征学习的高并行度类初启脉冲神经网络
- [ ] High-parallelism Inception-like Spiking Neural Networks for Unsupervised Feature Learning 
时间：2021年03月09日                         第一作者：Mingyuan Meng                       [链接](https://arxiv.org/abs/2001.01680).                     
<details>	<summary>注释</summary>	Published at Neurocomputing DOI: 10.1016/j.neucom.2021.02.027 </details>
<details>	<summary>邮件日期</summary>	2021年03月10日</details>

# 72、基于在线进化脉冲神经网络的流数据无监督异常检测
- [ ] Unsupervised Anomaly Detection in Stream Data with Online Evolving Spiking Neural Networks 
时间：2021年03月08日                         第一作者：Piotr S. Maci\k{a}g (1)                       [链接](https://arxiv.org/abs/1912.08785).                     
<details>	<summary>注释</summary>	52 pages Journal-ref: Neural Networks, Volume 139, 2021, Pages 118-139 DOI: 10.1016/j.neunet.2021.02.017 </details>
<details>	<summary>邮件日期</summary>	2021年03月10日</details>

# 71、一点点能量就有很大的帮助：高效节能，从卷积神经网络到脉冲神经网络的精确转换
- [ ] A Little Energy Goes a Long Way: Energy-Efficient, Accurate Conversion from Convolutional Neural Networks to Spiking Neural Networks 
时间：2021年03月06日                         第一作者：Dengyu Wu                       [链接](https://arxiv.org/abs/2103.00944).                     
<details>	<summary>邮件日期</summary>	2021年03月09日</details>

# 70、神经形态平台上强化学习的双记忆结构
- [ ] A Dual-Memory Architecture for Reinforcement Learning on Neuromorphic Platforms 
时间：2021年03月05日                         第一作者：Wilkie Olin-Ammentorp                       [链接](https://arxiv.org/abs/2103.04780).                     
## 摘要：强化学习（RL）是生物系统中学习的基础，并提供了一个框架来解决现实世界人工智能应用的众多挑战。RL技术的有效实现允许部署在边缘用例中的代理获得新的能力，例如改进的导航、理解复杂情况和关键决策。为了实现这个目标，我们描述了一个灵活的架构来在神经形态平台上进行强化学习。该体系结构是使用Intel神经形态处理器实现的，并演示了如何使用脉冲动力学解决各种任务。我们的研究为现实世界的RL应用提出了一个可用的节能解决方案，并证明了神经形态平台对RL问题的适用性。
<details>	<summary>英文摘要</summary>	Reinforcement learning (RL) is a foundation of learning in biological systems and provides a framework to address numerous challenges with real-world artificial intelligence applications. Efficient implementations of RL techniques could allow for agents deployed in edge-use cases to gain novel abilities, such as improved navigation, understanding complex situations and critical decision making. Towards this goal, we describe a flexible architecture to carry out reinforcement learning on neuromorphic platforms. This architecture was implemented using an Intel neuromorphic processor and demonstrated solving a variety of tasks using spiking dynamics. Our study proposes a usable energy efficient solution for real-world RL applications and demonstrates applicability of the neuromorphic platforms for RL problems. </details>
<details>	<summary>注释</summary>	20 pages, 6 figures ACM-class: I.2 </details>
<details>	<summary>邮件日期</summary>	2021年03月09日</details>

# 69、基于在线元学习的脉冲神经网络快速设备自适应
- [ ] Fast On-Device Adaptation for Spiking Neural Networks via Online-Within-Online Meta-Learning 
时间：2021年02月21日                         第一作者：Bleema Rosenfeld                       [链接](https://arxiv.org/abs/2103.03901).                     
## 摘要：脉冲神经网络（Spiking Neural Networks，SNNs）由于其低功耗特性，近年来在移动医疗管理和自然语言处理等应用中作为边缘智能的机器学习模型得到了广泛的应用。在这种高度个人化的用例中，模型必须能够用最少的训练数据来适应个体的独特特征。元学习被认为是一种训练模型的方法，这种模型能够快速适应新的任务。为数不多的现有snn元学习解决方案离线运行，并且需要某种形式的反向传播，这与当前的神经形态边缘设备不兼容。在这篇论文中，我们提出了一个SNN的在线内在线元学习规则OWOML-SNN，它能够在任务流上实现终身学习，并且依赖于本地的、无后台的、嵌套的更新。
<details>	<summary>英文摘要</summary>	Spiking Neural Networks (SNNs) have recently gained popularity as machine learning models for on-device edge intelligence for applications such as mobile healthcare management and natural language processing due to their low power profile. In such highly personalized use cases, it is important for the model to be able to adapt to the unique features of an individual with only a minimal amount of training data. Meta-learning has been proposed as a way to train models that are geared towards quick adaptation to new tasks. The few existing meta-learning solutions for SNNs operate offline and require some form of backpropagation that is incompatible with the current neuromorphic edge-devices. In this paper, we propose an online-within-online meta-learning rule for SNNs termed OWOML-SNN, that enables lifelong learning on a stream of tasks, and relies on local, backprop-free, nested updates. </details>
<details>	<summary>注释</summary>	Accepted for publication at DSLW 2021 </details>
<details>	<summary>邮件日期</summary>	2021年03月09日</details>

# 68、机器人神经形态感知工具箱
- [ ] A toolbox for neuromorphic sensing in robotics 
时间：2021年03月03日                         第一作者：Julien Dupeyroux                       [链接](https://arxiv.org/abs/2103.02751).                     
## 摘要：由神经形态计算引入的第三代人工智能（AI）正在彻底改变机器人和自主系统感知世界、处理信息以及与环境交互的方式。神经形态系统的高灵活性、能量效率和鲁棒性的承诺得到了模拟脉冲神经网络的软件工具和硬件集成（神经形态处理器）的广泛支持。然而，尽管人们在神经形态视觉（基于事件的摄像机）方面做出了努力，但值得注意的是，大多数机器人可用的传感器与神经形态计算（信息被编码成脉冲）本质上仍然不兼容。为了方便传统传感器的使用，我们需要将输出信号转换成脉冲流，即一系列事件（+1，-1）及其相应的时间戳。在这篇论文中，我们从机器人学的角度对编码算法进行了回顾，并进一步通过一个基准测试来评估它们的性能。我们还引入了ROS（机器人操作系统）工具箱来编码和解码来自机器人上任何类型传感器的输入信号。这项倡议旨在刺激和促进神经形态人工智能的机器人集成，并有机会使传统的现成传感器适应最强大的机器人工具之一ROS中的脉冲神经网络。
<details>	<summary>英文摘要</summary>	The third generation of artificial intelligence (AI) introduced by neuromorphic computing is revolutionizing the way robots and autonomous systems can sense the world, process the information, and interact with their environment. The promises of high flexibility, energy efficiency, and robustness of neuromorphic systems is widely supported by software tools for simulating spiking neural networks, and hardware integration (neuromorphic processors). Yet, while efforts have been made on neuromorphic vision (event-based cameras), it is worth noting that most of the sensors available for robotics remain inherently incompatible with neuromorphic computing, where information is encoded into spikes. To facilitate the use of traditional sensors, we need to convert the output signals into streams of spikes, i.e., a series of events (+1, -1) along with their corresponding timestamps. In this paper, we propose a review of the coding algorithms from a robotics perspective and further supported by a benchmark to assess their performance. We also introduce a ROS (Robot Operating System) toolbox to encode and decode input signals coming from any type of sensor available on a robot. This initiative is meant to stimulate and facilitate robotic integration of neuromorphic AI, with the opportunity to adapt traditional off-the-shelf sensors to spiking neural nets within one of the most powerful robotic tools, ROS. </details>
<details>	<summary>注释</summary>	7 pages, 3 figures, 3 tables, 7 algorithms </details>
<details>	<summary>邮件日期</summary>	2021年03月05日</details>

# 67、基于事件的合成孔径成像
- [ ] Event-based Synthetic Aperture Imaging 
时间：2021年03月03日                         第一作者：Xiang Zhang                       [链接](https://arxiv.org/abs/2103.02376).                     
## 摘要：合成孔径成像（syntheticapertureimaging，SAI）是通过模糊掉离焦的前景遮挡，并从多视点图像中重建出在焦遮挡的目标，从而达到透视效果。然而，非常密集的遮挡和极端的光照条件可能会对基于传统帧相机的SAI带来显著的干扰，导致性能退化。为了解决这些问题，我们提出了一种基于事件摄像机的SAI系统，它可以产生具有极低延迟和高动态范围的异步事件。因此，它可以通过几乎连续的视图来消除密集遮挡的干扰，同时解决过度/不足曝光的问题。为了重建遮挡目标，提出了一种由脉冲神经网络（SNNs）和卷积神经网络（CNNs）组成的混合编解码网络。在混合网络中，首先对采集到的事件的时空信息进行SNN层编码，然后通过样式转换CNN解码器将其转换为遮挡目标的视觉图像。实验结果表明，该方法在处理高密度遮挡和极端光照条件下具有良好的性能，可以用纯事件数据重建高质量的视觉图像。
<details>	<summary>英文摘要</summary>	Synthetic aperture imaging (SAI) is able to achieve the see through effect by blurring out the off-focus foreground occlusions and reconstructing the in-focus occluded targets from multi-view images. However, very dense occlusions and extreme lighting conditions may bring significant disturbances to SAI based on conventional frame-based cameras, leading to performance degeneration. To address these problems, we propose a novel SAI system based on the event camera which can produce asynchronous events with extremely low latency and high dynamic range. Thus, it can eliminate the interference of dense occlusions by measuring with almost continuous views, and simultaneously tackle the over/under exposure problems. To reconstruct the occluded targets, we propose a hybrid encoder-decoder network composed of spiking neural networks (SNNs) and convolutional neural networks (CNNs). In the hybrid network, the spatio-temporal information of the collected events is first encoded by SNN layers, and then transformed to the visual image of the occluded targets by a style-transfer CNN decoder. Through experiments, the proposed method shows remarkable performance in dealing with very dense occlusions and extreme lighting conditions, and high quality visual images can be reconstructed using pure event data. </details>
<details>	<summary>注释</summary>	9 pages, 7 figures </details>
<details>	<summary>邮件日期</summary>	2021年03月04日</details>

# 66、一点点能量就有很大的帮助：高效节能，从卷积神经网络到脉冲神经网络的精确转换
- [ ] A Little Energy Goes a Long Way: Energy-Efficient, Accurate Conversion from Convolutional Neural Networks to Spiking Neural Networks 
时间：2021年03月01日                         第一作者：Dengyu Wu                       [链接](https://arxiv.org/abs/2103.00944).                     
## 摘要：脉冲神经网络（SNNs）提供了一种处理时空数据的固有能力，也就是说，处理现实世界中的感官数据，但是很难训练出高精度的模型。SNN的一个主要研究方向是将预先训练好的卷积神经网络（CNN）转换成具有相同结构的SNN。最先进的转换方法正在接近精度极限，即SNN对原始CNN的精度损失接近于零。然而，我们注意到，只有当处理输入消耗的能量显著增加时，这才有可能实现。在本文中，我们认为这种“能量换精度”的趋势是不必要的——一点点能量可以大大提高精度损失的接近零。具体来说，我们提出了一种新的CNN到SNN转换方法，该方法能够使用合理的短脉冲序列（例如，CIFAR10图像的256个时间步）来实现接近零的精度损失。新的转换方法称为显式电流控制（ECC），包含三种技术（电流归一化、残差消除阈值和批量归一化一致性维护），以便在处理输入时显式控制流经SNN的电流。我们将ECC实现到一个昵称为SpKeras的工具中，该工具可以方便地导入Keras-CNN模型并将其转换为snn。我们使用该工具进行了一系列广泛的实验——使用VGG16和各种数据集，如CIFAR10和CIFAR100——并与最先进的转换方法进行了比较。结果表明，ECC是一种很有前途的方法，它可以同时优化系统的能耗和精度损失。
<details>	<summary>英文摘要</summary>	Spiking neural networks (SNNs) offer an inherent ability to process spatial-temporal data, or in other words, realworld sensory data, but suffer from the difficulty of training high accuracy models. A major thread of research on SNNs is on converting a pre-trained convolutional neural network (CNN) to an SNN of the same structure. State-of-the-art conversion methods are approaching the accuracy limit, i.e., the near-zero accuracy loss of SNN against the original CNN. However, we note that this is made possible only when significantly more energy is consumed to process an input. In this paper, we argue that this trend of ''energy for accuracy'' is not necessary -- a little energy can go a long way to achieve the near-zero accuracy loss. Specifically, we propose a novel CNN-to-SNN conversion method that is able to use a reasonably short spike train (e.g., 256 timesteps for CIFAR10 images) to achieve the near-zero accuracy loss. The new conversion method, named as explicit current control (ECC), contains three techniques (current normalisation, thresholding for residual elimination, and consistency maintenance for batch-normalisation), in order to explicitly control the currents flowing through the SNN when processing inputs. We implement ECC into a tool nicknamed SpKeras, which can conveniently import Keras CNN models and convert them into SNNs. We conduct an extensive set of experiments with the tool -- working with VGG16 and various datasets such as CIFAR10 and CIFAR100 -- and compare with state-of-the-art conversion methods. Results show that ECC is a promising method that can optimise over energy consumption and accuracy loss simultaneously. </details>
<details>	<summary>邮件日期</summary>	2021年03月02日</details>

# 65、SpikeDyn：一种动态环境下具有连续无监督学习能力的节能型Spiking神经网络框架
- [ ] SpikeDyn: A Framework for Energy-Efficient Spiking Neural Networks with Continual and Unsupervised Learning Capabilities in Dynamic Environments 
时间：2021年02月28日                         第一作者：Rachmad Vidya Wicaksana Putra                       [链接](https://arxiv.org/abs/2103.00424).                     
## 摘要：脉冲神经网络（Spiking Neural Networks，SNNs）由于其生物学上的合理性，具有高效无监督和持续学习能力的潜力，但其复杂性仍然是一个严重的研究挑战，以使其能够针对资源受限的场景（如嵌入式系统、物联网边缘等）进行节能设计。我们提出了SpikeDyn，一个在动态环境中具有连续和无监督学习能力的节能snn的综合框架，用于训练和推理阶段。它是通过以下多种不同的机制实现的：1）减少神经元的操作，用直接的侧抑制代替抑制神经元；2）一种记忆和能量受限的SNN模型搜索算法，该算法利用分析模型来估计不同候选SNN的记忆足迹和能量消耗建立并选择一个Pareto最优SNN模型；3）一个轻量级的连续无监督学习算法，采用自适应学习率、自适应膜阈值电位、权值衰减和减少虚假更新。实验结果表明，对于一个由400个兴奋神经元组成的网络，我们的SpikeDyn在训练和推理方面的平均能耗分别比现有的方法降低了51%和37%。由于改进的学习算法，SpikeDyn对最近学习的任务进行分类，平均比最新技术提高了21%，对以前学习的任务平均提高了8%。
<details>	<summary>英文摘要</summary>	Spiking Neural Networks (SNNs) bear the potential of efficient unsupervised and continual learning capabilities because of their biological plausibility, but their complexity still poses a serious research challenge to enable their energy-efficient design for resource-constrained scenarios (like embedded systems, IoT-Edge, etc.). We propose SpikeDyn, a comprehensive framework for energy-efficient SNNs with continual and unsupervised learning capabilities in dynamic environments, for both the training and inference phases. It is achieved through the following multiple diverse mechanisms: 1) reduction of neuronal operations, by replacing the inhibitory neurons with direct lateral inhibitions; 2) a memory- and energy-constrained SNN model search algorithm that employs analytical models to estimate the memory footprint and energy consumption of different candidate SNN models and selects a Pareto-optimal SNN model; and 3) a lightweight continual and unsupervised learning algorithm that employs adaptive learning rates, adaptive membrane threshold potential, weight decay, and reduction of spurious updates. Our experimental results show that, for a network with 400 excitatory neurons, our SpikeDyn reduces the energy consumption on average by 51% for training and by 37% for inference, as compared to the state-of-the-art. Due to the improved learning algorithm, SpikeDyn provides on avg. 21% accuracy improvement over the state-of-the-art, for classifying the most recently learned task, and by 8% on average for the previously learned tasks. </details>
<details>	<summary>注释</summary>	To appear at the 58th IEEE/ACM Design Automation Conference (DAC), December 2021, San Francisco, CA, USA </details>
<details>	<summary>邮件日期</summary>	2021年03月02日</details>

# 64、传统人工神经网络到脉冲神经网络的优化转换
- [ ] Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks 
时间：2021年02月28日                         第一作者：Shikuang Deng                       [链接](https://arxiv.org/abs/2103.00476).                     
## 摘要：脉冲神经网络（SNNs）是一种受生物启发的人工神经网络（ANNs），由脉冲神经元组成，用于处理异步离散信号。由于snn的离散性，使得snn在功耗和推理速度上都有很大的提高，但通常很难直接从零开始训练。另一种方法是通过复制神经网络的权值和调整snn中神经元的峰值阈值电位，将传统的ann转化为snn。研究人员设计了新的SNN结构和转换算法来减小转换误差。然而，一个有效的转换应该解决SNN和ANN体系结构之间的差异，用一个有效的损失函数的近似值，这个函数在这个领域是缺失的。在这项工作中，我们分析了递归归约到分层求和的转换误差，并提出了一种新的策略管道，通过结合阈值平衡和软重置机制将权值转移到目标SNN。这种流水线使得转换后的SNN和传统的ann之间几乎没有精度损失，只有典型SNN模拟时间的$\sim1/10$。我们的方法有望在能量和内存有限的情况下，更好地支持SNNs，并将其移植到嵌入式平台上。
<details>	<summary>英文摘要</summary>	Spiking neural networks (SNNs) are biology-inspired artificial neural networks (ANNs) that comprise of spiking neurons to process asynchronous discrete signals. While more efficient in power consumption and inference speed on the neuromorphic hardware, SNNs are usually difficult to train directly from scratch with spikes due to the discreteness. As an alternative, many efforts have been devoted to converting conventional ANNs into SNNs by copying the weights from ANNs and adjusting the spiking threshold potential of neurons in SNNs. Researchers have designed new SNN architectures and conversion algorithms to diminish the conversion error. However, an effective conversion should address the difference between the SNN and ANN architectures with an efficient approximation \DSK{of} the loss function, which is missing in the field. In this work, we analyze the conversion error by recursive reduction to layer-wise summation and propose a novel strategic pipeline that transfers the weights to the target SNN by combining threshold balance and soft-reset mechanisms. This pipeline enables almost no accuracy loss between the converted SNNs and conventional ANNs with only $\sim1/10$ of the typical SNN simulation time. Our method is promising to get implanted onto embedded platforms with better support of SNNs with limited energy and memory. </details>
<details>	<summary>邮件日期</summary>	2021年03月02日</details>

# 63、SparkXD：一种基于近似DRAM的弹性高效脉冲神经网络推理框架
- [ ] SparkXD: A Framework for Resilient and Energy-Efficient Spiking Neural Network Inference using Approximate DRAM 
时间：2021年02月28日                         第一作者：Rachmad Vidya Wicaksana Putra                       [链接](https://arxiv.org/abs/2103.00421).                     
## 摘要：脉冲神经网络（SNNs）由于其生物稀疏性，具有实现低能耗的潜力。一些研究表明，片外存储器（DRAM）访问是SNN处理中最消耗能量的操作。然而，SNN系统中的最新技术并没有优化每个访问的DRAM能量，因此阻碍了实现高能量效率。为了最大限度地减少每次存取的DRAM能量，一个按键旋钮用于降低DRAM电源电压，但这可能会导致DRAM错误（即所谓的近似DRAM）。针对这一点，我们提出了SparkXD，这是一个新的框架，它提供了一个综合的联合解决方案，用于使用低功耗dram在电压引起的错误下进行弹性和节能SNN推断。SparkXD的关键机制是：（1）通过考虑近似DRAM误码的容错训练来提高SNN的容错性；（2）分析改进的SNN模型的容错性，找到满足目标精度约束的最大可容忍误码率；（3）能量有效的DRAM数据映射对于弹性SNN模型，该模型将权重映射到适当的DRAM位置以最小化DRAM访问能量。通过这些机制，SparkXD减轻了DRAM（近似）错误的负面影响，并提供了所需的精度。实验结果表明，当目标精度在基线设计的1%以内（即SNN没有DRAM错误）时，SparkXD在不同网络规模下平均降低DRAM能量约40%。
<details>	<summary>英文摘要</summary>	Spiking Neural Networks (SNNs) have the potential for achieving low energy consumption due to their biologically sparse computation. Several studies have shown that the off-chip memory (DRAM) accesses are the most energy-consuming operations in SNN processing. However, state-of-the-art in SNN systems do not optimize the DRAM energy-per-access, thereby hindering achieving high energy-efficiency. To substantially minimize the DRAM energy-per-access, a key knob is to reduce the DRAM supply voltage but this may lead to DRAM errors (i.e., the so-called approximate DRAM). Towards this, we propose SparkXD, a novel framework that provides a comprehensive conjoint solution for resilient and energy-efficient SNN inference using low-power DRAMs subjected to voltage-induced errors. The key mechanisms of SparkXD are: (1) improving the SNN error tolerance through fault-aware training that considers bit errors from approximate DRAM, (2) analyzing the error tolerance of the improved SNN model to find the maximum tolerable bit error rate (BER) that meets the targeted accuracy constraint, and (3) energy-efficient DRAM data mapping for the resilient SNN model that maps the weights in the appropriate DRAM location to minimize the DRAM access energy. Through these mechanisms, SparkXD mitigates the negative impact of DRAM (approximation) errors, and provides the required accuracy. The experimental results show that, for a target accuracy within 1% of the baseline design (i.e., SNN without DRAM errors), SparkXD reduces the DRAM energy by ca. 40% on average across different network sizes. </details>
<details>	<summary>注释</summary>	To appear at the 58th IEEE/ACM Design Automation Conference (DAC), December 2021, San Francisco, CA, USA </details>
<details>	<summary>邮件日期</summary>	2021年03月02日</details>

# 62、结合脉冲神经网络和人工神经网络的图像增强分类
- [ ] Combining Spiking Neural Network and Artificial Neural Network for Enhanced Image Classification 
时间：2021年02月28日                         第一作者：Naoya Muramatsu                        [链接](https://arxiv.org/abs/2102.10592).                     
<details>	<summary>注释</summary>	This paper written for DEIM 2021 (https://db-event.jpn.org/deim2021/) has 12 pages, 6 figures and 3 tables MSC-class: 68T05 (Primary) 68T05 (Secondary) ACM-class: I.2.6 </details>
<details>	<summary>邮件日期</summary>	2021年03月02日</details>

# 61、癫痫发作预测的神经形态计算新方法
- [ ] A New Neuromorphic Computing Approach for Epileptic Seizure Prediction 
时间：2021年02月25日                         第一作者：Fengshi Tian                       [链接](https://arxiv.org/abs/2102.12773).                     
## 摘要：报道了几种利用卷积神经网络（CNNs）预测癫痫发作的高特异性和敏感性方法。然而，cnn的计算成本很高，耗电量也很大。这些不便使得基于CNN的方法很难在可穿戴设备上实现。基于能量有效的脉冲神经网络（SNNs），提出了一种用于癫痫发作预测的神经形态计算方法。该方法利用设计的高斯随机离散编码器从脑电样本中产生脉冲序列，并在结合了CNNs和SNNs优点的脉冲卷积神经网络（spiking-convolutional neural network，spiking-CNN）中进行预测。实验结果表明，该方法的灵敏度、特异性和AUC分别为95.1%、99.2%和0.912%，计算复杂度比CNN降低了98.58%，表明该方法硬件友好，精度高。
<details>	<summary>英文摘要</summary>	Several high specificity and sensitivity seizure prediction methods with convolutional neural networks (CNNs) are reported. However, CNNs are computationally expensive and power hungry. These inconveniences make CNN-based methods hard to be implemented on wearable devices. Motivated by the energy-efficient spiking neural networks (SNNs), a neuromorphic computing approach for seizure prediction is proposed in this work. This approach uses a designed gaussian random discrete encoder to generate spike sequences from the EEG samples and make predictions in a spiking convolutional neural network (Spiking-CNN) which combines the advantages of CNNs and SNNs. The experimental results show that the sensitivity, specificity and AUC can remain 95.1%, 99.2% and 0.912 respectively while the computation complexity is reduced by 98.58% compared to CNN, indicating that the proposed Spiking-CNN is hardware friendly and of high precision. </details>
<details>	<summary>注释</summary>	Accepted to 2021 IEEE International Symposium on Circuits and Systems (ISCAS) Journal-ref: 2021 IEEE International Symposium on Circuits and Systems (ISCAS) </details>
<details>	<summary>邮件日期</summary>	2021年02月26日</details>

# 60、皮层振荡在脉冲神经网络中实现了基于采样的计算
- [ ] Cortical oscillations implement a backbone for sampling-based computation in spiking neural networks 
时间：2021年02月23日                         第一作者：Agnes Korcsak-Gorzo                       [链接](https://arxiv.org/abs/2006.11099).                     
<details>	<summary>注释</summary>	28 pages, 11 figures </details>
<details>	<summary>邮件日期</summary>	2021年02月24日</details>

# 59、STDP通过反向传播增强脉冲神经网络的学习
- [ ] STDP enhances learning by backpropagation in a spiking neural network 
时间：2021年02月21日                         第一作者：Kotaro Furuya                        [链接](https://arxiv.org/abs/2102.10530).                     
## 摘要：提出了一种用于脉冲神经网络的半监督学习方法。该方法由反向传播的有监督学习和脉冲时间依赖可塑性（STDP）的无监督学习组成，STDP是一种生物学上合理的学习规则。数值实验表明，在使用少量标记数据的情况下，该方法在不增加标记的情况下提高了精度。现有的判别模型半监督学习方法还没有实现这一特性。对于事件驱动系统，可以实现所提出的学习方法。因此，如果在神经形态硬件上实现，它在实时性问题上会非常高效。结果表明，STDP在监督学习后的应用中除了自组织外，还起着重要的作用，这不同于以往将STDP作为预训练解释为自组织的方法。
<details>	<summary>英文摘要</summary>	A semi-supervised learning method for spiking neural networks is proposed. The proposed method consists of supervised learning by backpropagation and subsequent unsupervised learning by spike-timing-dependent plasticity (STDP), which is a biologically plausible learning rule. Numerical experiments show that the proposed method improves the accuracy without additional labeling when a small amount of labeled data is used. This feature has not been achieved by existing semi-supervised learning methods of discriminative models. It is possible to implement the proposed learning method for event-driven systems. Hence, it would be highly efficient in real-time problems if it were implemented on neuromorphic hardware. The results suggest that STDP plays an important role other than self-organization when applied after supervised learning, which differs from the previous method of using STDP as pre-training interpreted as self-organization. </details>
<details>	<summary>注释</summary>	9 pages, 12 figures </details>
<details>	<summary>邮件日期</summary>	2021年02月23日</details>

# 58、结合脉冲神经网络和人工神经网络的图像增强分类
- [ ] Combining Spiking Neural Network and Artificial Neural Network for Enhanced Image Classification 
时间：2021年02月21日                         第一作者：Naoya Muramatsu                        [链接](https://arxiv.org/abs/2102.10592).                     
## 摘要：随着深度神经网络的不断创新，更接近生物脑突触的脉冲神经网络（spiking neural networks，SNNs）因其低功耗而备受关注。然而，对于连续数据值，它们必须采用编码过程将值转换为峰值序列。因此，它们还没有超过直接处理这些值的人工神经网络（ANNs）的性能。为此，我们将人工神经网络和神经网络相结合，建立了多功能混合神经网络（HNNs），以提高相关性能。
<details>	<summary>英文摘要</summary>	With the continued innovations of deep neural networks, spiking neural networks (SNNs) that more closely resemble biological brain synapses have attracted attention owing to their low power consumption. However, for continuous data values, they must employ a coding process to convert the values to spike trains. Thus, they have not yet exceeded the performance of artificial neural networks (ANNs), which handle such values directly. To this end, we combine an ANN and an SNN to build versatile hybrid neural networks (HNNs) that improve the concerned performance. </details>
<details>	<summary>注释</summary>	This paper written for DEIM 2021 (https://db-event.jpn.org/deim2021/) has 12 pages, 6 figures and 3 tables MSC-class: 68T05 (Primary) 68T05 (Secondary) ACM-class: I.2.6 </details>
<details>	<summary>邮件日期</summary>	2021年02月23日</details>

# 57、脉冲神经形态芯片学习纠缠量子态
- [ ] Spiking neuromorphic chip learns entangled quantum states 
时间：2021年02月18日                         第一作者：Stefanie Czischek                       [链接](https://arxiv.org/abs/2008.01039).                     
<details>	<summary>注释</summary>	21 pages, 6 figures Submission to SciPost </details>
<details>	<summary>邮件日期</summary>	2021年02月22日</details>

# 56、方程：神经形态实现的脉冲驱动平衡传播
- [ ] EqSpike: Spike-driven Equilibrium Propagation for Neuromorphic Implementations 
时间：2021年02月17日                         第一作者：Erwann Martin                       [链接](https://arxiv.org/abs/2010.07859).                     
<details>	<summary>邮件日期</summary>	2021年02月18日</details>

# 55、阴阳数据集
- [ ] The Yin-Yang dataset 
时间：2021年02月16日                         第一作者：Laura Kriener                       [链接](https://arxiv.org/abs/2102.08211).                     
## 摘要：阴阳数据集是为研究生物似然误差反向传播和脉冲神经网络的深度学习而开发的。它提供了一些优势，可以替代经典的深度学习数据集，特别是在算法和模型原型场景中。首先，它体积更小，因此学习速度更快，因此更适合部署在网络规模有限的神经形态芯片上。第二，与深度神经网络相比，它在使用浅层神经网络所能达到的精度之间存在着非常明显的差距。
<details>	<summary>英文摘要</summary>	The Yin-Yang dataset was developed for research on biologically plausible error backpropagation and deep learning in spiking neural networks. It serves as an alternative to classic deep learning datasets, especially in algorithm- and model-prototyping scenarios, by providing several advantages. First, it is smaller and therefore faster to learn, thereby being better suited for the deployment on neuromorphic chips with limited network sizes. Second, it exhibits a very clear gap between the accuracies achievable using shallow as compared to deep neural networks. </details>
<details>	<summary>注释</summary>	3 pages, 3 figures, 2 tables </details>
<details>	<summary>邮件日期</summary>	2021年02月17日</details>

# 54、用GANs归化神经形态视觉事件流
- [ ] Naturalizing Neuromorphic Vision Event Streams Using GANs 
时间：2021年02月14日                         第一作者：Dennis Robey                       [链接](https://arxiv.org/abs/2102.07243).                     
## 摘要：动态视觉传感器能够在资源受限的环境中以高时间分辨率工作，但代价是捕获静态内容。事件流的稀疏特性使得下游处理任务更高效，因为它们适合于功率高效的脉冲神经网络。与神经形态视觉相关的挑战之一是缺乏事件流的可解释性。虽然大多数应用用例并不打算让事件流被除分类网络之外的任何东西直观地解释，但是在传统高速CMOS传感器无法到达的空间中集成这些传感器的机会将丢失。例如，像内窥镜这样的生物入侵传感器必须符合严格的电源预算，这就不允许以兆赫的速度进行图像集成。虽然动态视觉传感可以填补这一空白，解释的挑战仍然存在，并将降低临床诊断的信心。产生式对抗网络的使用为克服和补偿视觉芯片空间分辨率差和缺乏可解释性提供了一种可能的解决方案。本文系统地应用Pix2Pix网络对CIFAR-10和linnaeus5数据集的事件流进行自然化处理。通过对归化的事件流进行图像分类（其收敛到等效原始图像的2.81%以内），并对CIFAR-10和Linnaeus 5数据集的未处理事件流进行13.19%的相关改进，对网络的质量进行了基准测试。
<details>	<summary>英文摘要</summary>	Dynamic vision sensors are able to operate at high temporal resolutions within resource constrained environments, though at the expense of capturing static content. The sparse nature of event streams enables efficient downstream processing tasks as they are suited for power-efficient spiking neural networks. One of the challenges associated with neuromorphic vision is the lack of interpretability of event streams. While most application use-cases do not intend for the event stream to be visually interpreted by anything other than a classification network, there is a lost opportunity to integrating these sensors in spaces that conventional high-speed CMOS sensors cannot go. For example, biologically invasive sensors such as endoscopes must fit within stringent power budgets, which do not allow MHz-speeds of image integration. While dynamic vision sensing can fill this void, the interpretation challenge remains and will degrade confidence in clinical diagnostics. The use of generative adversarial networks presents a possible solution to overcoming and compensating for a vision chip's poor spatial resolution and lack of interpretability. In this paper, we methodically apply the Pix2Pix network to naturalize the event stream from spike-converted CIFAR-10 and Linnaeus 5 datasets. The quality of the network is benchmarked by performing image classification of naturalized event streams, which converges to within 2.81% of equivalent raw images, and an associated improvement over unprocessed event streams by 13.19% for the CIFAR-10 and Linnaeus 5 datasets. </details>
<details>	<summary>注释</summary>	5 pages, 7 figures </details>
<details>	<summary>邮件日期</summary>	2021年02月16日</details>

# 53、利用混合信号脉冲学习电路实现高效均衡网络
- [ ] Implementing efficient balanced networks with mixed-signal spike-based learning circuits 
时间：2021年02月12日                         第一作者：Julian B\"uchel                       [链接](https://arxiv.org/abs/2010.14353).                     
<details>	<summary>注释</summary>	5 pages, 6 figures. Accepted at IEEE International Symposium on Circuits and Systems 2021 </details>
<details>	<summary>邮件日期</summary>	2021年02月15日</details>

# 52、基于广义期望最大化的脉冲神经网络多样本在线学习
- [ ] Multi-Sample Online Learning for Spiking Neural Networks based on Generalized Expectation Maximization 
时间：2021年02月05日                         第一作者：Hyeryung Jang                        [链接](https://arxiv.org/abs/2102.03280).                     
## 摘要：脉冲神经网络（SNNs）提供了一种新的计算范式，它通过二元神经动态激活来获取生物大脑的一些效率。概率SNN模型通常通过使用对数似然梯度的无偏估计来训练以最大化期望输出的可能性。虽然先前的工作使用单样本估计器从一次运行的网络，本文提出利用多个隔间采样独立的脉冲信号，同时共享突触权重。其关键思想是利用这些信号来获得对数似然训练准则及其梯度的更精确的统计估计。该方法基于广义期望最大化（GEM），利用重要性抽样优化了对数似然的近似。导出的在线学习算法实现了一个具有全局每隔室学习信号的三因素规则。在神经形态MNIST-DVS数据集上的分类任务的实验结果表明，当增加用于训练和推理的隔室数量时，在对数似然性、准确性和校准方面有显著的改进。
<details>	<summary>英文摘要</summary>	Spiking Neural Networks (SNNs) offer a novel computational paradigm that captures some of the efficiency of biological brains by processing through binary neural dynamic activations. Probabilistic SNN models are typically trained to maximize the likelihood of the desired outputs by using unbiased estimates of the log-likelihood gradients. While prior work used single-sample estimators obtained from a single run of the network, this paper proposes to leverage multiple compartments that sample independent spiking signals while sharing synaptic weights. The key idea is to use these signals to obtain more accurate statistical estimates of the log-likelihood training criterion, as well as of its gradient. The approach is based on generalized expectation-maximization (GEM), which optimizes a tighter approximation of the log-likelihood using importance sampling. The derived online learning algorithm implements a three-factor rule with global per-compartment learning signals. Experimental results on a classification task on the neuromorphic MNIST-DVS data set demonstrate significant improvements in terms of log-likelihood, accuracy, and calibration when increasing the number of compartments used for training and inference. </details>
<details>	<summary>注释</summary>	To be presented at ICASSP 2021. Author's Accepted Manuscript. (A longer version can be found at arXiv:2007.11894), Author's Accepted Manuscript. arXiv admin note: text overlap with arXiv:2007.11894 </details>
<details>	<summary>邮件日期</summary>	2021年02月08日</details>

# 51、优化的脉冲神经元通过双峰时间编码对图像进行高精度分类
- [ ] Optimized spiking neurons classify images with high accuracy through temporal coding with two spikes 
时间：2021年01月26日                         第一作者：Christoph St\"ockl                        [链接](https://arxiv.org/abs/2002.00860).                     
<details>	<summary>注释</summary>	23 pages, 5 figures, 1 tables </details>
<details>	<summary>邮件日期</summary>	2021年01月27日</details>

# 50、一种基于Loihi神经形态处理器的DVS摄像机手势识别算法
- [ ] An Efficient Spiking Neural Network for Recognizing Gestures with a DVS Camera on the Loihi Neuromorphic Processor 
时间：2021年01月25日                         第一作者：Riccardo Massa                       [链接](https://arxiv.org/abs/2006.09985).                     
<details>	<summary>注释</summary>	Accepted for publication at the 2020 International Joint Conference on Neural Networks (IJCNN) </details>
<details>	<summary>邮件日期</summary>	2021年01月26日</details>

# 49、腿型机器人仿生运动的神经形态自适应脉冲CPG
- [ ] Neuromorphic adaptive spiking CPG towards bio-inspired locomotion of legged robots 
时间：2021年01月24日                         第一作者：Pablo Lopez-Osorio                       [链接](https://arxiv.org/abs/2101.09709).                     
## 摘要：近年来，脊椎动物的运动机制为机器人系统性能的提高提供了灵感。这些机制包括它们的运动对通过生物传感器记录的环境变化的适应性。在这方面，我们的目标是复制这种适应性的腿机器人通过一个脉冲中心模式发生器。这种脉冲中心模式发生器产生不同的运动（节奏）模式，这些模式由外部刺激驱动，即连接到机器人的力敏感电阻器的输出，以提供反馈。脉冲中枢模式发生器由五个漏神经元群组成，这些漏神经元群具有特定的拓扑结构，使得节律模式可以由上述外部刺激产生和驱动。因此，末端机器人平台（任意腿机器人）的运动可以通过使用任意传感器作为输入来适应地形。采用brian2模拟器和SpiNNaker神经形态平台，对具有自适应学习的脉冲中心模式发生器进行了软硬件仿真验证。特别是，我们的实验清楚地表明，当输入刺激不同时，脉冲中央模式发生器群体中产生的脉冲之间的振荡频率发生了适应性变化。为了验证脉冲中心模式发生器的鲁棒性和适应性，我们通过改变传感器的输出进行了多次测试。这些实验在brian2和SpiNNaker中进行；两个实现都显示出相似的行为，Pearson相关系数为0.905。
<details>	<summary>英文摘要</summary>	In recent years, locomotion mechanisms exhibited by vertebrate animals have been the inspiration for the improvement in the performance of robotic systems. These mechanisms include the adaptability of their locomotion to any change registered in the environment through their biological sensors. In this regard, we aim to replicate such kind of adaptability in legged robots through a Spiking Central Pattern Generator. This Spiking Central Pattern Generator generates different locomotion (rhythmic) patterns which are driven by an external stimulus, that is, the output of a Force Sensitive Resistor connected to the robot to provide feedback. The Spiking Central Pattern Generator consists of a network of five populations of Leaky Integrate-and-Fire neurons designed with a specific topology in such a way that the rhythmic patterns can be generated and driven by the aforementioned external stimulus. Therefore, the locomotion of the end robotic platform (any-legged robot) can be adapted to the terrain by using any sensor as input. The Spiking Central Pattern Generator with adaptive learning has been numerically validated at software and hardware level, using the Brian 2 simulator and the SpiNNaker neuromorphic platform for the latest. In particular, our experiments clearly show an adaptation in the oscillation frequencies between the spikes produced in the populations of the Spiking Central Pattern Generator while the input stimulus varies. To validate the robustness and adaptability of the Spiking Central Pattern Generator, we have performed several tests by variating the output of the sensor. These experiments were carried out in Brian 2 and SpiNNaker; both implementations showed a similar behavior with a Pearson correlation coefficient of 0.905. </details>
<details>	<summary>注释</summary>	23 pages, 12 figures </details>
<details>	<summary>邮件日期</summary>	2021年01月26日</details>

# 48、事件驱动目标识别的脉冲学习系统
- [ ] A Spike Learning System for Event-driven Object Recognition 
时间：2021年01月21日                         第一作者：Shibo Zhou                       [链接](https://arxiv.org/abs/2101.08850).                     
## 摘要：事件驱动传感器，如激光雷达和动态视觉传感器（DVS）在高分辨率和高速应用中受到越来越多的关注。为了提高识别精度，人们做了大量的工作。然而，对于识别延迟或时间效率这一基本问题的研究还远远不够。在本文中，我们提出了一个脉冲学习系统，该系统使用脉冲神经网络（SNN）和一种新的时态编码来实现精确快速的目标识别。提出的时态编码方案将每个事件的到达时间和数据映射到SNN脉冲时间，使得异步到达的事件立即得到处理而没有延迟。该方案很好地结合了SNN的异步处理能力，提高了时间效率。与现有系统相比的一个关键优势是，每个识别任务的事件累积时间由系统自动确定，而不是由用户预先设置。系统可以在不等待所有输入事件的情况下提前完成识别。在7个激光雷达和DVS数据集上进行了广泛的实验。结果表明，该系统在取得显著时间效率的同时，具有最先进的识别精度。实验结果表明，在不同的实验条件下，在KITTI数据集上，识别延迟降低了56.3%-91.7%。
<details>	<summary>英文摘要</summary>	Event-driven sensors such as LiDAR and dynamic vision sensor (DVS) have found increased attention in high-resolution and high-speed applications. A lot of work has been conducted to enhance recognition accuracy. However, the essential topic of recognition delay or time efficiency is largely under-explored. In this paper, we present a spiking learning system that uses the spiking neural network (SNN) with a novel temporal coding for accurate and fast object recognition. The proposed temporal coding scheme maps each event's arrival time and data into SNN spike time so that asynchronously-arrived events are processed immediately without delay. The scheme is integrated nicely with the SNN's asynchronous processing capability to enhance time efficiency. A key advantage over existing systems is that the event accumulation time for each recognition task is determined automatically by the system rather than pre-set by the user. The system can finish recognition early without waiting for all the input events. Extensive experiments were conducted over a list of 7 LiDAR and DVS datasets. The results demonstrated that the proposed system had state-of-the-art recognition accuracy while achieving remarkable time efficiency. Recognition delay was shown to reduce by 56.3% to 91.7% in various experiment settings over the popular KITTI dataset. </details>
<details>	<summary>注释</summary>	Shibo Zhou and Wei Wang contributed equally to this work ACM-class: I.5.1; I.5.4; I.2.6; I.2.10 </details>
<details>	<summary>邮件日期</summary>	2021年01月25日</details>

# 47、亚稳材料的自主合成
- [ ] Autonomous synthesis of metastable materials 
时间：2021年01月19日                         第一作者：Sebastian Ament                       [链接](https://arxiv.org/abs/2101.07385).                     
## 摘要：人工智能的自主实验为加速科学发现提供了新的范例。非平衡材料合成是复杂的、资源密集型实验的标志，其加速将是材料发现和发展的分水岭。非平衡合成相图的绘制最近通过高通量实验得到了加速，但由于参数空间太大而无法进行详尽的探索，因此仍然限制了材料的研究。我们演示了加速合成和探索亚稳材料通过分层自主实验所管辖的科学自主推理代理（SARA）。SARA集成了机器人材料的合成和表征以及一系列人工智能方法，有效地揭示了加工相图的结构。SARA设计了用于平行材料合成的横向梯度激光脉冲退火（lg-LSA）实验，并利用光谱技术快速识别相变。多维参数空间的有效探索是通过嵌套的主动学习（AL）循环来实现的，该循环建立在先进的机器学习模型之上，该模型结合了实验的基本物理以及端到端的不确定性量化。有了这一点，以及在多个尺度上的协调，SARA体现了人工智能对复杂科学任务的利用。我们通过自主绘制Bi$\u2$O$\u3$系统的合成相边界来证明它的性能，从而在建立一个合成相图时产生数量级的加速，该合成相图包括在室温下动力学稳定$\delta$-Bi$\u2$O$\u3$的条件，固体氧化物燃料电池等电化学技术的关键发展。
<details>	<summary>英文摘要</summary>	Autonomous experimentation enabled by artificial intelligence (AI) offers a new paradigm for accelerating scientific discovery. Non-equilibrium materials synthesis is emblematic of complex, resource-intensive experimentation whose acceleration would be a watershed for materials discovery and development. The mapping of non-equilibrium synthesis phase diagrams has recently been accelerated via high throughput experimentation but still limits materials research because the parameter space is too vast to be exhaustively explored. We demonstrate accelerated synthesis and exploration of metastable materials through hierarchical autonomous experimentation governed by the Scientific Autonomous Reasoning Agent (SARA). SARA integrates robotic materials synthesis and characterization along with a hierarchy of AI methods that efficiently reveal the structure of processing phase diagrams. SARA designs lateral gradient laser spike annealing (lg-LSA) experiments for parallel materials synthesis and employs optical spectroscopy to rapidly identify phase transitions. Efficient exploration of the multi-dimensional parameter space is achieved with nested active learning (AL) cycles built upon advanced machine learning models that incorporate the underlying physics of the experiments as well as end-to-end uncertainty quantification. With this, and the coordination of AL at multiple scales, SARA embodies AI harnessing of complex scientific tasks. We demonstrate its performance by autonomously mapping synthesis phase boundaries for the Bi$_2$O$_3$ system, leading to orders-of-magnitude acceleration in establishment of a synthesis phase diagram that includes conditions for kinetically stabilizing $\delta$-Bi$_2$O$_3$ at room temperature, a critical development for electrochemical technologies such as solid oxide fuel cells. </details>
<details>	<summary>邮件日期</summary>	2021年01月20日</details>

# 46、模拟七鳃鳗机器人在SpiNNaker和Loihi神经形态板上运行的脉冲中心模式发生器
- [ ] A Spiking Central Pattern Generator for the control of a simulated lamprey robot running on SpiNNaker and Loihi neuromorphic boards 
时间：2021年01月18日                         第一作者：Emmanouil Angelidis                       [链接](https://arxiv.org/abs/2101.07001).                     
## 摘要：中枢模式发生器（CPGs）模型长期以来被用来研究动物运动的神经机制，也被用作机器人研究的工具。在这项工作中，我们提出了一个脉冲CPG神经网络及其在神经形态硬件上的实现作为一种手段来控制一个模拟七鳃鳗模型。为了建立我们的CPG模型，我们采用了自然出现的动态系统，这些系统是通过在神经工程框架（NEF）中使用递归神经种群而产生的。我们定义了我们模型背后的数学公式，它由一个由高电平信号调制的耦合抽象振荡器系统组成，能够产生各种输出步态。我们证明，利用这种中央模式发生器模型的数学公式，可以将该模型转化为一个脉冲神经网络（SNN），该网络可以很容易地用SNN模拟器Nengo进行模拟。然后利用脉冲CPG模型生成不同场景下模拟七鳃鳗机器人模型的游动步态。我们证明，通过修改网络的输入（可以由感官信息提供），机器人可以在方向和速度上进行动态控制。该方法可推广应用于工程应用和科学研究中的其它类型的cpg。我们在两个神经形态平台上测试我们的系统，SpiNNaker和Loihi。最后，我们证明了这类脉冲算法在能量效率和计算速度方面显示了利用神经形态硬件理论优势的潜力。
<details>	<summary>英文摘要</summary>	Central Pattern Generators (CPGs) models have been long used to investigate both the neural mechanisms that underlie animal locomotion as well as a tool for robotic research. In this work we propose a spiking CPG neural network and its implementation on neuromorphic hardware as a means to control a simulated lamprey model. To construct our CPG model, we employ the naturally emerging dynamical systems that arise through the use of recurrent neural populations in the Neural Engineering Framework (NEF). We define the mathematical formulation behind our model, which consists of a system of coupled abstract oscillators modulated by high-level signals, capable of producing a variety of output gaits. We show that with this mathematical formulation of the Central Pattern Generator model, the model can be turned into a Spiking Neural Network (SNN) that can be easily simulated with Nengo, an SNN simulator. The spiking CPG model is then used to produce the swimming gaits of a simulated lamprey robot model in various scenarios. We show that by modifying the input to the network, which can be provided by sensory information, the robot can be controlled dynamically in direction and pace. The proposed methodology can be generalized to other types of CPGs suitable for both engineering applications and scientific research. We test our system on two neuromorphic platforms, SpiNNaker and Loihi. Finally, we show that this category of spiking algorithms shows a promising potential to exploit the theoretical advantages of neuromorphic hardware in terms of energy efficiency and computational speed. </details>
<details>	<summary>注释</summary>	25 pages, 15 figures </details>
<details>	<summary>邮件日期</summary>	2021年01月19日</details>

# 45、用于时空特征提取的卷积脉冲神经网络
- [ ] Convolutional Spiking Neural Networks for Spatio-Temporal Feature Extraction 
时间：2021年01月18日                         第一作者：Ali Samadzadeh                       [链接](https://arxiv.org/abs/2003.12346).                     
<details>	<summary>注释</summary>	10 pages, 7 figures, 2 tables </details>
<details>	<summary>邮件日期</summary>	2021年01月20日</details>

# 44、方程：神经形态实现的脉冲驱动平衡传播
- [ ] EqSpike: Spike-driven Equilibrium Propagation for Neuromorphic Implementations 
时间：2021年01月15日                         第一作者：Erwann Martin                       [链接](https://arxiv.org/abs/2010.07859).                     
<details>	<summary>邮件日期</summary>	2021年01月18日</details>

# 43、概率脉冲神经网络的多样本在线学习
- [ ] Multi-Sample Online Learning for Probabilistic Spiking Neural Networks 
时间：2021年01月05日                         第一作者：Hyeryung Jang                        [链接](https://arxiv.org/abs/2007.11894).                     
<details>	<summary>注释</summary>	Submitted </details>
<details>	<summary>邮件日期</summary>	2021年01月06日</details>

# 42、人工脉冲量子神经元
- [ ] An Artificial Spiking Quantum Neuron 
时间：2020年12月30日                         第一作者：Lasse Bj{\o}rn Kristensen                       [链接](https://arxiv.org/abs/1907.06269).                     
<details>	<summary>邮件日期</summary>	2021年01月01日</details>

# 41、深Q网络向事件驱动脉冲神经网络转化的策略与基准
- [ ] Strategy and Benchmark for Converting Deep Q-Networks to Event-Driven Spiking Neural Networks 
时间：2020年12月23日                         第一作者：Weihao Tan                       [链接](https://arxiv.org/abs/2009.14456).                     
<details>	<summary>注释</summary>	Accepted by AAAI2021 </details>
<details>	<summary>邮件日期</summary>	2020年12月24日</details>

# 40、生成模型的进化变分优化
- [ ] Evolutionary Variational Optimization of Generative Models 
时间：2020年12月22日                         第一作者：Jakob Drefs                       [链接](https://arxiv.org/abs/2012.12294).                     
## 摘要：我们结合两种流行的优化方法来推导生成模型的学习算法：变分优化和进化算法。利用截断后验概率作为变分分布族，实现了离散时滞生成模型的组合。截断后验概率的变分参数是一组潜在状态。通过将这些状态解释为个体的基因组，并利用变分下界来定义适应度，我们可以应用进化算法来实现变分循环。所使用的变分分布是非常灵活的，我们证明了进化算法可以有效地优化变分界。此外，变分回路通常适用（“黑盒”），无需分析推导。为了说明该方法的普遍适用性，我们将该方法应用于三种生成模型（使用噪声或贝叶斯网、二进制稀疏编码以及脉冲和板稀疏编码）。为了证明新的变分方法的有效性和效率，我们使用了图像去噪和修复的标准竞争基准。这些基准允许对各种方法进行定量比较，包括概率方法、深层确定性和生成性网络以及非局部图像处理方法。在“零镜头”学习（当只使用损坏的图像进行训练时）的范畴中，我们观察到进化变分算法在许多基准设置中显著改善了最新的状态。对于一个著名的修复基准，我们还观察到了各种算法的最新性能，尽管我们只对损坏的图像进行训练。总的来说，我们的研究强调了研究生成模型的优化方法以提高性能的重要性。
<details>	<summary>英文摘要</summary>	We combine two popular optimization approaches to derive learning algorithms for generative models: variational optimization and evolutionary algorithms. The combination is realized for generative models with discrete latents by using truncated posteriors as the family of variational distributions. The variational parameters of truncated posteriors are sets of latent states. By interpreting these states as genomes of individuals and by using the variational lower bound to define a fitness, we can apply evolutionary algorithms to realize the variational loop. The used variational distributions are very flexible and we show that evolutionary algorithms can effectively and efficiently optimize the variational bound. Furthermore, the variational loop is generally applicable ("black box") with no analytical derivations required. To show general applicability, we apply the approach to three generative models (we use noisy-OR Bayes Nets, Binary Sparse Coding, and Spike-and-Slab Sparse Coding). To demonstrate effectiveness and efficiency of the novel variational approach, we use the standard competitive benchmarks of image denoising and inpainting. The benchmarks allow quantitative comparisons to a wide range of methods including probabilistic approaches, deep deterministic and generative networks, and non-local image processing methods. In the category of "zero-shot" learning (when only the corrupted image is used for training), we observed the evolutionary variational algorithm to significantly improve the state-of-the-art in many benchmark settings. For one well-known inpainting benchmark, we also observed state-of-the-art performance across all categories of algorithms although we only train on the corrupted image. In general, our investigations highlight the importance of research on optimization methods for generative models to achieve performance improvements. </details>
<details>	<summary>邮件日期</summary>	2020年12月24日</details>

# 39、用直接训练的更大的脉冲神经网络进行更深入的研究
- [ ] Going Deeper With Directly-Trained Larger Spiking Neural Networks 
时间：2020年12月18日                         第一作者：Hanle Zheng                       [链接](https://arxiv.org/abs/2011.05280).                     
<details>	<summary>注释</summary>	12 pages, 6 figures, conference or other essential info </details>
<details>	<summary>邮件日期</summary>	2020年12月21日</details>

# 38、脉冲神经网络到神经形态硬件的热感知编译
- [ ] Thermal-Aware Compilation of Spiking Neural Networks to Neuromorphic Hardware 
时间：2020年12月17日                         第一作者：Twisha Titirsha                        [链接](https://arxiv.org/abs/2010.04773).                     
<details>	<summary>注释</summary>	Accepted for publication at LCPC 2020 </details>
<details>	<summary>邮件日期</summary>	2020年12月21日</details>

# 37、基于贝叶斯学习的二元权值脉冲神经网络训练
- [ ] BiSNN: Training Spiking Neural Networks with Binary Weights via Bayesian Learning 
时间：2020年12月15日                         第一作者：Hyeryung Jang                        [链接](https://arxiv.org/abs/2012.08300).                     
## 摘要：基于人工神经网络（ANN）的电池供电设备的推理可以通过限制突触权值为二进制，从而消除执行乘法的需要，从而提高能量效率。另一种新兴的方法依赖于使用脉冲神经网络（SNNs），这是一种受生物启发的动态事件驱动模型，通过使用二进制稀疏激活来提高能源效率。本文介绍了一种SNN模型，它结合了时间稀疏二进制激活和二进制权值的优点。推导了两种学习规则，第一种基于直通和代理梯度技术的组合，第二种基于贝叶斯范式。实验验证了全精度实现的性能损失，并证明了贝叶斯范式在精度和校准方面的优势。
<details>	<summary>英文摘要</summary>	Artificial Neural Network (ANN)-based inference on battery-powered devices can be made more energy-efficient by restricting the synaptic weights to be binary, hence eliminating the need to perform multiplications. An alternative, emerging, approach relies on the use of Spiking Neural Networks (SNNs), biologically inspired, dynamic, event-driven models that enhance energy efficiency via the use of binary, sparse, activations. In this paper, an SNN model is introduced that combines the benefits of temporally sparse binary activations and of binary weights. Two learning rules are derived, the first based on the combination of straight-through and surrogate gradient techniques, and the second based on a Bayesian paradigm. Experiments validate the performance loss with respect to full-precision implementations, and demonstrate the advantage of the Bayesian paradigm in terms of accuracy and calibration. </details>
<details>	<summary>注释</summary>	Submitted </details>
<details>	<summary>邮件日期</summary>	2020年12月16日</details>

# 36、脉冲神经元Hebbian和STDP学习权值的约束
- [ ] Constraints on Hebbian and STDP learned weights of a spiking neuron 
时间：2020年12月14日                         第一作者：Dominique Chu                        [链接](https://arxiv.org/abs/2012.07664).                     
## 摘要：我们从数学上分析了Hebbian和STDP学习规则对权值的限制，这些规则应用于权值归一化的脉冲神经元。在纯Hebbian学习的情况下，我们发现标准化的权值等于权值的提升概率，直到依赖于学习率的修正项，并且通常很小。对于STDP算法，可以导出类似的关系，其中标准化的权重值反映了权重的提升和降级概率之间的差异。这些关系实际上很有用，因为它们允许检查Hebbian和STDP算法的收敛性。另一个应用是新颖性检测。我们使用MNIST数据集演示了这一点。
<details>	<summary>英文摘要</summary>	We analyse mathematically the constraints on weights resulting from Hebbian and STDP learning rules applied to a spiking neuron with weight normalisation. In the case of pure Hebbian learning, we find that the normalised weights equal the promotion probabilities of weights up to correction terms that depend on the learning rate and are usually small. A similar relation can be derived for STDP algorithms, where the normalised weight values reflect a difference between the promotion and demotion probabilities of the weight. These relations are practically useful in that they allow checking for convergence of Hebbian and STDP algorithms. Another application is novelty detection. We demonstrate this using the MNIST dataset. </details>
<details>	<summary>邮件日期</summary>	2020年12月15日</details>

# 35、生物神经网络的低阶模型
- [ ] Low-Order Model of Biological Neural Networks 
时间：2020年12月12日                         第一作者：Huachuan Wang                        [链接](https://arxiv.org/abs/2012.06720).                     
## 摘要：生物神经网络的生物似真低阶模型（LOM）是由树突节点/树、脉冲/非脉冲神经元、无监督/有监督协方差/累积学习机制、反馈连接和最大泛化方案组成的递归层次网络。这些组件模型的动机和必要性在于使LOM易于学习和检索，而无需区分、优化或迭代，以及聚类、检测和识别多个/层次损坏、扭曲和闭塞的时间和空间模式。
<details>	<summary>英文摘要</summary>	A biologically plausible low-order model (LOM) of biological neural networks is a recurrent hierarchical network of dendritic nodes/trees, spiking/nonspiking neurons, unsupervised/ supervised covariance/accumulative learning mechanisms, feedback connections, and a scheme for maximal generalization. These component models are motivated and necessitated by making LOM learn and retrieve easily without differentiation, optimization, or iteration, and cluster, detect and recognize multiple/hierarchical corrupted, distorted, and occluded temporal and spatial patterns. </details>
<details>	<summary>邮件日期</summary>	2020年12月15日</details>

# 34、脉冲神经网络第一部分：空间模式检测
- [ ] Spiking Neural Networks -- Part I: Detecting Spatial Patterns 
时间：2020年12月09日                         第一作者：Hyeryung Jang                       [链接](https://arxiv.org/abs/2010.14208).                     
<details>	<summary>注释</summary>	Submitted </details>
<details>	<summary>邮件日期</summary>	2020年12月10日</details>

# 33、脉冲神经网络第二部分：时空模式检测
- [ ] Spiking Neural Networks -- Part II: Detecting Spatio-Temporal Patterns 
时间：2020年12月09日                         第一作者：Nicolas Skatchkovsky                       [链接](https://arxiv.org/abs/2010.14217).                     
<details>	<summary>注释</summary>	Submitted. The first two authors have equally contributed to this work </details>
<details>	<summary>邮件日期</summary>	2020年12月10日</details>

# 32、脉冲神经网络第三部分：神经形态通信
- [ ] Spiking Neural Networks -- Part III: Neuromorphic Communications 
时间：2020年12月09日                         第一作者：Nicolas Skatchkovsky                       [链接](https://arxiv.org/abs/2010.14220).                     
<details>	<summary>注释</summary>	Submitted </details>
<details>	<summary>邮件日期</summary>	2020年12月10日</details>

# 31、一种训练脉冲神经网络的多智能体进化机器人框架
- [ ] A multi-agent evolutionary robotics framework to train spiking neural networks 
时间：2020年12月07日                         第一作者：Souvik Das                       [链接](https://arxiv.org/abs/2012.03485).                     
## 摘要：提出了一种基于多智能体进化机器人（ER）的训练脉冲神经网络（SNN）的新框架。snn群体的权重以及它们在ER环境中控制的机器人的形态参数被视为表型。该框架的规则根据某些机器人在竞争环境中捕获食物的效率，选择它们及其snn进行繁殖，而选择其他snn进行淘汰。虽然机器人和它们的snn没有通过任何损失函数获得生存或繁衍的明确奖励，但当它们进化到捕猎食物并在这些规则下生存时，这些驱动力隐而不露。它们捕获食物的效率随着世代的变化而呈现出间断平衡的进化特征。给出了两种表型遗传算法：变异遗传算法和带变异交叉遗传算法。通过对每种算法进行100个实验，比较了这些算法的性能。我们发现，在SNN中，带突变的交叉比仅带统计显著性差异的突变能提高40%的学习速度。
<details>	<summary>英文摘要</summary>	A novel multi-agent evolutionary robotics (ER) based framework, inspired by competitive evolutionary environments in nature, is demonstrated for training Spiking Neural Networks (SNN). The weights of a population of SNNs along with morphological parameters of bots they control in the ER environment are treated as phenotypes. Rules of the framework select certain bots and their SNNs for reproduction and others for elimination based on their efficacy in capturing food in a competitive environment. While the bots and their SNNs are given no explicit reward to survive or reproduce via any loss function, these drives emerge implicitly as they evolve to hunt food and survive within these rules. Their efficiency in capturing food as a function of generations exhibit the evolutionary signature of punctuated equilibria. Two evolutionary inheritance algorithms on the phenotypes, Mutation and Crossover with Mutation, are demonstrated. Performances of these algorithms are compared using ensembles of 100 experiments for each algorithm. We find that Crossover with Mutation promotes 40% faster learning in the SNN than mere Mutation with a statistically significant margin. </details>
<details>	<summary>注释</summary>	9 pages, 11 figures </details>
<details>	<summary>邮件日期</summary>	2020年12月08日</details>

# 30、大脑的功能是否像一台使用相位三值计算的量子相位计算机？
- [ ] Does the brain function as a quantum phase computer using phase ternary computation? 
时间：2020年12月04日                         第一作者：Andrew Simon Johnson                        [链接](https://arxiv.org/abs/2012.06537).                     
## 摘要：在这里，我们提供的证据表明，神经通信的基本基础来自于压力脉冲/孤子，它能够以足够的时间精度进行计算，以克服任何处理错误。神经系统内的信号传递和计算是复杂而不同的现象。动作电位是塑性的，这使得动作电位峰值对于神经计算来说是一个不合适的固定点，但是动作电位阈值适合于这个目的。此外，由脉冲神经元计时的神经模型的运算速率低于克服加工误差所需的速率。以视网膜处理为例，我们证明了基于电缆理论的当代神经传导理论不适合解释视网膜和大脑其他部分的完整功能所需的短计算时间。此外，电缆理论不能帮助传播的行动电位，因为在激活阈值没有足够的电荷在激活地点连续离子通道静电开放。对大脑神经网络的解构表明它是一组量子相位计算机中的一员，其中图灵机是最简单的：大脑是另一个基于相位三值计算的计算机。然而，使用图灵机制的尝试无法解决视网膜的编码或智能的计算，因为基于图灵的计算机的技术是根本不同的。我们证明了大脑神经网络中的编码是基于量子的，其中量子有一个时间变量和一个相位基变量，这使得相位三值计算成为可能，正如之前在视网膜中所证明的那样。
<details>	<summary>英文摘要</summary>	Here we provide evidence that the fundamental basis of nervous communication is derived from a pressure pulse/soliton capable of computation with sufficient temporal precision to overcome any processing errors. Signalling and computing within the nervous system are complex and different phenomena. Action potentials are plastic and this makes the action potential peak an inappropriate fixed point for neural computation, but the action potential threshold is suitable for this purpose. Furthermore, neural models timed by spiking neurons operate below the rate necessary to overcome processing error. Using retinal processing as our example, we demonstrate that the contemporary theory of nerve conduction based on cable theory is inappropriate to account for the short computational time necessary for the full functioning of the retina and by implication the rest of the brain. Moreover, cable theory cannot be instrumental in the propagation of the action potential because at the activation-threshold there is insufficient charge at the activation site for successive ion channels to be electrostatically opened. Deconstruction of the brain neural network suggests that it is a member of a group of Quantum phase computers of which the Turing machine is the simplest: the brain is another based upon phase ternary computation. However, attempts to use Turing based mechanisms cannot resolve the coding of the retina or the computation of intelligence, as the technology of Turing based computers is fundamentally different. We demonstrate that that coding in the brain neural network is quantum based, where the quanta have a temporal variable and a phase-base variable enabling phase ternary computation as previously demonstrated in the retina. </details>
<details>	<summary>注释</summary>	16 pages, 7 figures. Key Words: Plasticity; Action potential; Timing; Error redaction; Synchronization; Quantum phase computation; Phase ternary computation; Retinal model ACM-class: I.2; J.3 </details>
<details>	<summary>邮件日期</summary>	2020年12月14日</details>

# 29、DIET-SNN：深脉冲神经网络中带泄漏和阈值优化的直接输入编码
- [ ] DIET-SNN: Direct Input Encoding With Leakage and Threshold Optimization in Deep Spiking Neural Networks 
时间：2020年12月02日                         第一作者：Nitin Rathi                       [链接](https://arxiv.org/abs/2008.03658).                     
<details>	<summary>邮件日期</summary>	2020年12月03日</details>

# 28、从头开始训练低潜伏期深脉冲神经网络的批标准化研究
- [ ] Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch 
时间：2020年11月30日                         第一作者：Youngeun Kim                       [链接](https://arxiv.org/abs/2010.01729).                     
<details>	<summary>邮件日期</summary>	2020年12月01日</details>

# 27、DIET-SNN：深脉冲神经网络中带泄漏和阈值优化的直接输入编码
- [ ] DIET-SNN: Direct Input Encoding With Leakage and Threshold Optimization in Deep Spiking Neural Networks 
时间：2020年11月29日                         第一作者：Nitin Rathi                       [链接](https://arxiv.org/abs/2008.03658).                     
<details>	<summary>邮件日期</summary>	2020年12月01日</details>

# 26、编译脉冲神经网络以减轻神经形态的硬件约束
- [ ] Compiling Spiking Neural Networks to Mitigate Neuromorphic Hardware Constraints 
时间：2020年11月27日                         第一作者：Adarsha Balaji                        [链接](https://arxiv.org/abs/2011.13965).                     
## 摘要：脉冲神经网络（SNNs）是在{resource}和{power}约束平台上进行时空模式识别的有效计算模型。在神经形态硬件上执行snn可以进一步降低这些平台的能耗。随着模型尺寸和复杂性的增加，将基于SNN的应用程序映射到基于tile的神经形态硬件变得越来越具有挑战性。这归因于神经突触核心的局限性，即。一种横杆，每个突触后神经元只能容纳固定数量的突触前连接。对于具有许多神经元和每个神经元的突触前连接的基于SNN的复杂模型，（1）在训练后可能需要修剪连接以适应交叉资源，导致模型质量的损失，例如准确性，（2）神经元和突触需要分块并放置在硬件的神经系统核心上，这可能导致延迟和能量消耗增加。在这项工作中，我们提出（1）一种新的展开技术，将具有许多突触前连接的神经元功能分解为一系列同质的神经单元，以显著提高交叉杆的利用率并保留所有突触前连接，（2）SpiNeMap，提出了一种在神经形态硬件上映射snn的新方法，旨在最小化能量消耗和峰值潜伏期。
<details>	<summary>英文摘要</summary>	Spiking Neural Networks (SNNs) are efficient computation models to perform spatio-temporal pattern recognition on {resource}- and {power}-constrained platforms. SNNs executed on neuromorphic hardware can further reduce energy consumption of these platforms. With increasing model size and complexity, mapping SNN-based applications to tile-based neuromorphic hardware is becoming increasingly challenging. This is attributed to the limitations of neuro-synaptic cores, viz. a crossbar, to accommodate only a fixed number of pre-synaptic connections per post-synaptic neuron. For complex SNN-based models that have many neurons and pre-synaptic connections per neuron, (1) connections may need to be pruned after training to fit onto the crossbar resources, leading to a loss in model quality, e.g., accuracy, and (2) the neurons and synapses need to be partitioned and placed on the neuro-sypatic cores of the hardware, which could lead to increased latency and energy consumption. In this work, we propose (1) a novel unrolling technique that decomposes a neuron function with many pre-synaptic connections into a sequence of homogeneous neural units to significantly improve the crossbar utilization and retain all pre-synaptic connections, and (2) SpiNeMap, a novel methodology to map SNNs on neuromorphic hardware with an aim to minimize energy consumption and spike latency. </details>
<details>	<summary>邮件日期</summary>	2020年12月01日</details>

# 25、一种用于在线学习的时态神经网络结构
- [ ] A Temporal Neural Network Architecture for Online Learning 
时间：2020年11月27日                         第一作者：James E. Smith                       [链接](https://arxiv.org/abs/2011.13844).                     
## 摘要：一个长期存在的观点是，通过模拟大脑新皮质的运作，脉冲神经网络（SNN）可以实现类似的理想特性：灵活的学习、速度和效率。时态神经网络（TNNs）是一种snn，用来传递和处理编码为相对峰值时间的信息（与峰值速率相反）。提出了一种TNN体系结构，并在在线监督分类的大背景下证明了TNN的操作。首先，通过无监督学习，TNN根据相似性将输入模式划分为多个簇。然后TNN将一个簇标识符传递给一个简单的在线监督解码器，解码器完成分类任务。TNN学习过程只使用每个突触的局部信号来调整突触的权重，聚类行为在全局范围内出现。系统架构是在抽象层描述的，类似于传统数字设计中的门和寄存器传输层。除了整体架构的特性之外，一些TNN组件对于这项工作来说是新的。虽然没有直接解决，但总体研究目标是TNNs的直接硬件实现。因此，所有的架构元素都很简单，处理的精度很低。重要的是，低精度导致学习时间非常快。使用历史悠久的MNIST数据集的仿真结果表明，学习时间比其他在线方法至少快一个数量级，同时提供了类似的错误率。
<details>	<summary>英文摘要</summary>	A long-standing proposition is that by emulating the operation of the brain's neocortex, a spiking neural network (SNN) can achieve similar desirable features: flexible learning, speed, and efficiency. Temporal neural networks (TNNs) are SNNs that communicate and process information encoded as relative spike times (in contrast to spike rates). A TNN architecture is proposed, and, as a proof-of-concept, TNN operation is demonstrated within the larger context of online supervised classification. First, through unsupervised learning, a TNN partitions input patterns into clusters based on similarity. The TNN then passes a cluster identifier to a simple online supervised decoder which finishes the classification task. The TNN learning process adjusts synaptic weights by using only signals local to each synapse, and clustering behavior emerges globally. The system architecture is described at an abstraction level analogous to the gate and register transfer levels in conventional digital design. Besides features of the overall architecture, several TNN components are new to this work. Although not addressed directly, the overall research objective is a direct hardware implementation of TNNs. Consequently, all the architecture elements are simple, and processing is done at very low precision. Importantly, low precision leads to very fast learning times. Simulation results using the time-honored MNIST dataset demonstrate learning times at least an order of magnitude faster than other online approaches while providing similar error rates. </details>
<details>	<summary>注释</summary>	13 pages, 10 figures ACM-class: C.3; I.2.6; I.5.3 </details>
<details>	<summary>邮件日期</summary>	2020年11月30日</details>

# 24、结合可学习膜时间常数提高脉冲神经网络的学习能力
- [ ] Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks 
时间：2020年11月27日                         第一作者：Wei Fang                       [链接](https://arxiv.org/abs/2007.05785).                     
<details>	<summary>邮件日期</summary>	2020年11月30日</details>

# 23、PeleNet：Loihi的储层计算框架
- [ ] PeleNet: A Reservoir Computing Framework for Loihi 
时间：2020年11月24日                         第一作者：Carlo Michaelis                       [链接](https://arxiv.org/abs/2011.12338).                     
## 摘要：脉冲神经网络的高级框架是快速原型化和复杂算法高效开发的关键因素。在过去的几年里，这种框架已经出现在传统的计算机上，但是编程神经形态的硬件仍然是一个挑战。通常需要具备神经形态芯片硬件知识的低级编程。PeleNet框架旨在简化神经形态硬件Loihi的储层计算。它是在英特尔的NxSDK之上构建的，是用Python编写的。该框架管理权重矩阵、参数和探测。特别是，它提供了一个自动和有效的网络分布在几个核心和芯片。这样，用户就不用面对技术细节，可以集中精力进行实验。
<details>	<summary>英文摘要</summary>	High-level frameworks for spiking neural networks are a key factor for fast prototyping and efficient development of complex algorithms. Such frameworks have emerged in the last years for traditional computers, but programming neuromorphic hardware is still a challenge. Often low level programming with knowledge about the hardware of the neuromorphic chip is required. The PeleNet framework aims to simplify reservoir computing for the neuromorphic hardware Loihi. It is build on top of the NxSDK from Intel and is written in Python. The framework manages weight matrices, parameters and probes. In particular, it provides an automatic and efficient distribution of networks over several cores and chips. With this, the user is not confronted with technical details and can concentrate on experiments. </details>
<details>	<summary>邮件日期</summary>	2020年11月26日</details>

# 22、面向零镜头跨语言图像检索
- [ ] Towards Zero-shot Cross-lingual Image Retrieval 
时间：2020年11月24日                         第一作者：Pranav Aggarwal                       [链接](https://arxiv.org/abs/2012.05107).                     
## 摘要：最近人们对多模态语言和视觉问题的兴趣激增。在语言方面，由于大多数多模态数据集都是单语的，所以这些模型主要关注英语。我们试图通过在文本方面进行跨语言预训练的零镜头方法来弥补这一差距。我们提出了一个简单而实用的方法来建立一个跨语言图像检索模型，该模型在单语训练数据集上进行训练，但可以在推理过程中以零镜头的跨语言方式使用。我们还引入了一个新的目标函数，通过相互推送不同的文本来收紧文本嵌入簇。最后，我们介绍了一个新的1K多语种MSCOCO2014字幕测试数据集（XTD10），该数据集采用7种语言，我们使用众包平台收集。我们使用它作为跨语言评估零炮模型性能的测试集。XTD10数据集在以下位置公开：https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10
<details>	<summary>英文摘要</summary>	There has been a recent spike in interest in multi-modal Language and Vision problems. On the language side, most of these models primarily focus on English since most multi-modal datasets are monolingual. We try to bridge this gap with a zero-shot approach for learning multi-modal representations using cross-lingual pre-training on the text side. We present a simple yet practical approach for building a cross-lingual image retrieval model which trains on a monolingual training dataset but can be used in a zero-shot cross-lingual fashion during inference. We also introduce a new objective function which tightens the text embedding clusters by pushing dissimilar texts from each other. Finally, we introduce a new 1K multi-lingual MSCOCO2014 caption test dataset (XTD10) in 7 languages that we collected using a crowdsourcing platform. We use this as the test set for evaluating zero-shot model performance across languages. XTD10 dataset is made publicly available here: https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10 </details>
<details>	<summary>邮件日期</summary>	2020年12月10日</details>

# 21、一种更具生物学意义的人工神经网络局部学习规则
- [ ] A More Biologically Plausible Local Learning Rule for ANNs 
时间：2020年11月24日                         第一作者：Shashi Kant Gupta                       [链接](https://arxiv.org/abs/2011.12012).                     
## 摘要：反向传播算法因其生物学合理性而经常引起争论。然而，为了寻求更具生物学意义的学习，人们提出了各种神经结构的学习方法。他们中的大多数人都试图解决“重量传输问题”，并试图通过一些替代方法在体系结构中向后传播错误。在这项工作中，我们研究了一种稍有不同的方法，它只使用局部信息来捕获脉冲定时信息，而不会传播错误。所提出的学习规则来自于脉冲时间依赖的可塑性和神经元联系的概念。对具有两个隐藏层的MNIST和IRIS数据集的二元分类进行的初步评估表明，其性能与反向传播相当。与通过交叉熵损失反向传播学习的模型相比，使用该方法学习的模型对FGSM攻击具有更好的鲁棒性。学习的局部性为网络中大规模的分布式并行学习提供了可能。最后，提出的方法是一个更符合生物学的方法，可能有助于理解生物神经元如何学习不同的抽象。
<details>	<summary>英文摘要</summary>	The backpropagation algorithm is often debated for its biological plausibility. However, various learning methods for neural architecture have been proposed in search of more biologically plausible learning. Most of them have tried to solve the "weight transport problem" and try to propagate errors backward in the architecture via some alternative methods. In this work, we investigated a slightly different approach that uses only the local information which captures spike timing information with no propagation of errors. The proposed learning rule is derived from the concepts of spike timing dependant plasticity and neuronal association. A preliminary evaluation done on the binary classification of MNIST and IRIS datasets with two hidden layers shows comparable performance with backpropagation. The model learned using this method also shows a possibility of better adversarial robustness against the FGSM attack compared to the model learned through backpropagation of cross-entropy loss. The local nature of learning gives a possibility of large scale distributed and parallel learning in the network. And finally, the proposed method is a more biologically sound method that can probably help in understanding how biological neurons learn different abstractions. </details>
<details>	<summary>注释</summary>	8 pages (4 main + 1 reference + 3 supplementary) </details>
<details>	<summary>邮件日期</summary>	2020年11月25日</details>

# 20、从头开始训练低潜伏期深脉冲神经网络的批标准化研究
- [ ] Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch 
时间：2020年11月24日                         第一作者：Youngeun Kim                       [链接](https://arxiv.org/abs/2010.01729).                     
<details>	<summary>邮件日期</summary>	2020年11月25日</details>

# 19、结合可学习膜时间常数提高脉冲神经网络的学习能力
- [ ] Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks 
时间：2020年11月24日                         第一作者：Wei Fang                       [链接](https://arxiv.org/abs/2007.05785).                     
<details>	<summary>邮件日期</summary>	2020年11月25日</details>

# 18、脉冲神经元的自然梯度学习
- [ ] Natural-gradient learning for spiking neurons 
时间：2020年11月23日                         第一作者：Elena Kreutzer                       [链接](https://arxiv.org/abs/2011.11710).                     
## 摘要：在许多突触可塑性的规范理论中，权重的更新隐含地依赖于所选择的权重参数化。例如，这个问题与神经元形态有关：在功能上对躯体放电的影响相当的突触，由于其在树突树上的位置不同，其脊柱大小可能有很大差异。基于欧氏梯度下降的经典理论很容易由于这种参数化依赖而导致不一致。这些问题是在黎曼几何的框架下解决的，在黎曼几何中，我们提出塑性应遵循自然梯度下降。在这一假设下，我们推导出了一个突触学习规则，该规则将功能效率与树突状民主、乘法标度和异突触可塑性等生物学现象的解释结合起来。因此，我们认为，在寻找功能性突触可塑性的过程中，进化可能产生了自己版本的自然梯度下降。
<details>	<summary>英文摘要</summary>	In many normative theories of synaptic plasticity, weight updates implicitly depend on the chosen parametrization of the weights. This problem relates, for example, to neuronal morphology: synapses which are functionally equivalent in terms of their impact on somatic firing can differ substantially in spine size due to their different positions along the dendritic tree. Classical theories based on Euclidean gradient descent can easily lead to inconsistencies due to such parametrization dependence. The issues are solved in the framework of Riemannian geometry, in which we propose that plasticity instead follows natural gradient descent. Under this hypothesis, we derive a synaptic learning rule for spiking neurons that couples functional efficiency with the explanation of several well-documented biological phenomena such as dendritic democracy, multiplicative scaling and heterosynaptic plasticity. We therefore suggest that in its search for functional synaptic plasticity, evolution might have come up with its own version of natural gradient descent. </details>
<details>	<summary>注释</summary>	Joint senior authorship: Walter M. Senn and Mihai A. Petrovici </details>
<details>	<summary>邮件日期</summary>	2020年11月25日</details>

# 17、多层记忆脉冲神经网络的片上错误触发学习
- [ ] On-Chip Error-triggered Learning of Multi-layer Memristive Spiking Neural Networks 
时间：2020年11月21日                         第一作者：Melika Payv                       [链接](https://arxiv.org/abs/2011.10852).                     
## 摘要：神经形态计算的最新突破表明，梯度下降学习的局部形式与脉冲神经网络（SNNs）和突触可塑性是相容的。虽然SNNs可以用神经形态的VLSI可伸缩地实现，但是仍然缺少一种可以在原地使用梯度下降进行学习的体系结构。在本文中，我们提出了一个局部的，梯度为基础的，错误触发学习算法与在线三元权值更新。所提出的算法可以在线训练多层snn与记忆神经形态的硬件表现出小损失的性能相比，国家的最新技术。我们还提出了一种基于忆阻纵横制阵列的硬件结构来执行所需的向量矩阵乘法。采用标准180nmcmos工艺，在亚阈值范围内设计了在线训练所需的外围电路，包括突触前、突触后和写电路。
<details>	<summary>英文摘要</summary>	Recent breakthroughs in neuromorphic computing show that local forms of gradient descent learning are compatible with Spiking Neural Networks (SNNs) and synaptic plasticity. Although SNNs can be scalably implemented using neuromorphic VLSI, an architecture that can learn using gradient-descent in situ is still missing. In this paper, we propose a local, gradient-based, error-triggered learning algorithm with online ternary weight updates. The proposed algorithm enables online training of multi-layer SNNs with memristive neuromorphic hardware showing a small loss in the performance compared with the state of the art. We also propose a hardware architecture based on memristive crossbar arrays to perform the required vector-matrix multiplications. The necessary peripheral circuitry including pre-synaptic, post-synaptic and write circuits required for online training, have been designed in the sub-threshold regime for power saving with a standard 180 nm CMOS process. </details>
<details>	<summary>注释</summary>	15 pages, 11 figures, Journal of Emerging Technology in Circuits and Systems (JETCAS) </details>
<details>	<summary>邮件日期</summary>	2020年11月24日</details>

# 16、快速而深入：具有第一脉冲时间的节能神经形态学习
- [ ] Fast and deep: energy-efficient neuromorphic learning with first-spike times 
时间：2020年11月19日                         第一作者：Julian G\"oltz                       [链接](https://arxiv.org/abs/1912.11443).                     
<details>	<summary>注释</summary>	20 pages, 8 figures </details>
<details>	<summary>邮件日期</summary>	2020年11月20日</details>

# 15、基于生物似然无监督延迟学习的脉冲神经网络时间特征提取
- [ ] Bio-plausible Unsupervised Delay Learning for Extracting Temporal Features in Spiking Neural Networks 
时间：2020年11月18日                         第一作者：Alireza Nadafian                       [链接](https://arxiv.org/abs/2011.09380).                     
## 摘要：神经元间传导延迟的可塑性在学习中起着基础性作用。然而，大脑中这种调节的确切机制仍然是一个开放的问题。了解突触延迟的精确调节可以帮助我们开发有效的大脑启发计算模型，提供与实验证据一致的见解。在这篇论文中，我们提出一个无监督的生物学上合理的学习规则来调整神经网络中的突触延迟。然后，我们提供了一些数学证明来证明我们的学习规则赋予神经元学习重复时空模式的能力。此外，将基于STDP的脉冲神经网络与我们提出的延迟学习规则相结合，应用于随机点运动图的实验结果表明了所提出的延迟学习规则在提取时间特征方面的有效性。
<details>	<summary>英文摘要</summary>	The plasticity of the conduction delay between neurons plays a fundamental role in learning. However, the exact underlying mechanisms in the brain for this modulation is still an open problem. Understanding the precise adjustment of synaptic delays could help us in developing effective brain-inspired computational models in providing aligned insights with the experimental evidence. In this paper, we propose an unsupervised biologically plausible learning rule for adjusting the synaptic delays in spiking neural networks. Then, we provided some mathematical proofs to show that our learning rule gives a neuron the ability to learn repeating spatio-temporal patterns. Furthermore, the experimental results of applying an STDP-based spiking neural network equipped with our proposed delay learning rule on Random Dot Kinematogram indicate the efficacy of the proposed delay learning rule in extracting temporal features. </details>
<details>	<summary>邮件日期</summary>	2020年11月19日</details>

# 14、脉冲神经网络的时间代理反向传播算法
- [ ] Temporal Surrogate Back-propagation for Spiking Neural Networks 
时间：2020年11月18日                         第一作者：Yukun Yang                       [链接](https://arxiv.org/abs/2011.09964).                     
## 摘要：脉冲神经网络（SNN）通常比人工神经网络（ANN）更节能，其工作方式与我们的大脑有很大的相似性。近年来，BP算法在神经网络训练中显示出了强大的能力。然而，由于脉冲行为是不可微的，BP不能直接应用于SNN。虽然已有的工作证明了在空间和时间方向上通过替代梯度或随机性来逼近BP梯度的几种方法，但是它们忽略了每一步之间重置机制引入的时间依赖性。本文以理论完善为目标，深入研究了缺失项的影响。通过增加重置机制的时间依赖性，新算法对玩具数据集的学习率调整更具鲁棒性，但对CIFAR-10等较大的学习任务没有太大的改进。从经验上讲，缺失项的好处不值得额外的计算开销。在许多情况下，可以忽略缺少的项。
<details>	<summary>英文摘要</summary>	Spiking neural networks (SNN) are usually more energy-efficient as compared to Artificial neural networks (ANN), and the way they work has a great similarity with our brain. Back-propagation (BP) has shown its strong power in training ANN in recent years. However, since spike behavior is non-differentiable, BP cannot be applied to SNN directly. Although prior works demonstrated several ways to approximate the BP-gradient in both spatial and temporal directions either through surrogate gradient or randomness, they omitted the temporal dependency introduced by the reset mechanism between each step. In this article, we target on theoretical completion and investigate the effect of the missing term thoroughly. By adding the temporal dependency of the reset mechanism, the new algorithm is more robust to learning-rate adjustments on a toy dataset but does not show much improvement on larger learning tasks like CIFAR-10. Empirically speaking, the benefits of the missing term are not worth the additional computational overhead. In many cases, the missing term can be ignored. </details>
<details>	<summary>注释</summary>	4 pases, 3 figures, 3 tables, 10 eqs </details>
<details>	<summary>邮件日期</summary>	2020年11月20日</details>

# 13、一种用于术中心电图高频振荡检测的脉冲神经网络（SNN）
- [ ] A Spiking Neural Network (SNN) for detecting High Frequency Oscillations (HFOs) in the intraoperative ECoG 
时间：2020年11月17日                         第一作者：Karla Burelo                        [链接](https://arxiv.org/abs/2011.08783).                     
## 摘要：癫痫手术需要彻底切除致痫脑组织，才能实现癫痫发作的自由。在术中的ECoG记录中，由致痫组织产生的高频振荡（HFOs）可以用来调整切除边缘。然而，实时自动检测HFOs仍然是一个开放的挑战。在这里，我们提出了一个脉冲神经网络（SNN）的自动HFO检测，是最适合神经形态的硬件实现。我们训练SNN来检测术中ECoG在线测量的HFO信号，使用一个独立标记的数据集。我们针对快速纹波频率范围（250-500hz）的HFO检测，并将网络结果与标记的HFO数据进行比较。我们赋予SNN一种新的伪影抑制机制来抑制突变，并在ECoG数据集上验证了其有效性。该SNN检测到的HFO率（术前记录中位数为6.6 HFO/min）与数据集中公布的HFO率（58 min，16次记录）相当。所有8例患者术后癫痫发作结果的“预测”准确率均为100%。这些结果为建立一个可在癫痫手术中用于指导致痫区切除的实时便携式电池供电HFO检测系统提供了进一步的进展。
<details>	<summary>英文摘要</summary>	To achieve seizure freedom, epilepsy surgery requires the complete resection of the epileptogenic brain tissue. In intraoperative ECoG recordings, high frequency oscillations (HFOs) generated by epileptogenic tissue can be used to tailor the resection margin. However, automatic detection of HFOs in real-time remains an open challenge. Here we present a spiking neural network (SNN) for automatic HFO detection that is optimally suited for neuromorphic hardware implementation. We trained the SNN to detect HFO signals measured from intraoperative ECoG on-line, using an independently labeled dataset. We targeted the detection of HFOs in the fast ripple frequency range (250-500 Hz) and compared the network results with the labeled HFO data. We endowed the SNN with a novel artifact rejection mechanism to suppress sharp transients and demonstrate its effectiveness on the ECoG dataset. The HFO rates (median 6.6 HFO/min in pre-resection recordings) detected by this SNN are comparable to those published in the dataset (58 min, 16 recordings). The postsurgical seizure outcome was "predicted" with 100% accuracy for all 8 patients. These results provide a further step towards the construction of a real-time portable battery-operated HFO detection system that can be used during epilepsy surgery to guide the resection of the epileptogenic zone. </details>
<details>	<summary>注释</summary>	11 pages, 3 figures, 2 tables. The results of this publication were obtained by simulating our hardware platform, built for online processing of biological signals. This hardware combines neural recording headstages with a multi-core neuromorphic processor arxiv.org/abs/2009.11245 </details>
<details>	<summary>邮件日期</summary>	2020年11月18日</details>

# 12、结合可学习膜时间常数提高脉冲神经网络的学习能力
- [ ] Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks 
时间：2020年11月16日                         第一作者：Wei Fang                       [链接](https://arxiv.org/abs/2007.05785).                     
<details>	<summary>邮件日期</summary>	2020年11月17日</details>

# 11、具有Alpha突触功能的脉冲神经网络中的时间编码：反向传播学习
- [ ] Temporal Coding in Spiking Neural Networks with Alpha Synaptic Function: Learning with Backpropagation 
时间：2020年11月16日                         第一作者：Iulia M. Comsa                       [链接](https://arxiv.org/abs/1907.13223).                     
<details>	<summary>注释</summary>	Open-source code related to this paper is available at https://github.com/google/ihmehimmeli v2: Added references and added some clarifications for the methods </details>
<details>	<summary>邮件日期</summary>	2020年11月18日</details>

# 10、LIAF-Net：轻量级高效时空信息处理的漏泄集成模拟消防网络
- [ ] LIAF-Net: Leaky Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information Processing 
时间：2020年11月12日                         第一作者：Zhenzhi Wu                       [链接](https://arxiv.org/abs/2011.06176).                     
## 摘要：基于漏积分火灾（LIF）模型的脉冲神经网络（SNNs）已被应用于节能的时空处理任务中。由于生物似有理的神经元动力学和简单性，LIF-SNN受益于事件驱动处理，然而，通常面临性能下降的尴尬。这可能是因为在LIF-SNN中，神经元通过脉冲传递信息。为了解决这一问题，本文提出了一种漏积分模拟火灾（LIAF）神经元模型，使得模拟值可以在神经元之间传输，并在此基础上建立了一个称为LIAF网络的深层网络，以实现高效的时空处理。在时域上，LIAF遵循传统的LIF动态机制来保持其时间处理能力。在空间域中，LIAF能够通过卷积积分或全连通积分对空间信息进行集成。作为一个时空层，LIAF也可以与传统的人工神经网络（ANN）层联合使用。实验结果表明，在bAbI问答（QA）任务中，LIAF网络的性能与选通递归单元（GRU）和长短时记忆（LSTM）相当，在时空动态视觉传感器（DVS）数据集（包括MNIST-DVS、CIFAR10-DVS和DVS128手势）上，LIAF网络的性能达到了最先进的水平，但数量要少得多与传统的LSTM、GRU、卷积LSTM（ConvLSTM）或3D卷积（Conv3D）构建的网络相比，突触权值和计算开销都有较大的提高。与传统的LIF-SNN相比，LIAF网络在所有这些实验中也显示出显著的精度提高。总之，LIAF-Net提供了一个结合ANNs和SNNs优点的轻量级高效时空信息处理框架。
<details>	<summary>英文摘要</summary>	Spiking neural networks (SNNs) based on Leaky Integrate and Fire (LIF) model have been applied to energy-efficient temporal and spatiotemporal processing tasks. Thanks to the bio-plausible neuronal dynamics and simplicity, LIF-SNN benefits from event-driven processing, however, usually faces the embarrassment of reduced performance. This may because in LIF-SNN the neurons transmit information via spikes. To address this issue, in this work, we propose a Leaky Integrate and Analog Fire (LIAF) neuron model, so that analog values can be transmitted among neurons, and a deep network termed as LIAF-Net is built on it for efficient spatiotemporal processing. In the temporal domain, LIAF follows the traditional LIF dynamics to maintain its temporal processing capability. In the spatial domain, LIAF is able to integrate spatial information through convolutional integration or fully-connected integration. As a spatiotemporal layer, LIAF can also be used with traditional artificial neural network (ANN) layers jointly. Experiment results indicate that LIAF-Net achieves comparable performance to Gated Recurrent Unit (GRU) and Long short-term memory (LSTM) on bAbI Question Answering (QA) tasks, and achieves state-of-the-art performance on spatiotemporal Dynamic Vision Sensor (DVS) datasets, including MNIST-DVS, CIFAR10-DVS and DVS128 Gesture, with much less number of synaptic weights and computational overhead compared with traditional networks built by LSTM, GRU, Convolutional LSTM (ConvLSTM) or 3D convolution (Conv3D). Compared with traditional LIF-SNN, LIAF-Net also shows dramatic accuracy gain on all these experiments. In conclusion, LIAF-Net provides a framework combining the advantages of both ANNs and SNNs for lightweight and efficient spatiotemporal information processing. </details>
<details>	<summary>注释</summary>	14 pages, 9 figures, submitted to IEEE Transactions on Neural Networks and Learning Systems ACM-class: I.2.6 </details>
<details>	<summary>邮件日期</summary>	2020年11月13日</details>

# 9、利用生物似然奖赏传播调整卷积脉冲神经网络
- [x] Tuning Convolutional Spiking Neural Network with Biologically-plausible Reward Propagation 
时间：2020年11月12日                         第一作者：Tielin Zhang                        [链接](https://arxiv.org/abs/2010.04434).                     
<details>	<summary>邮件日期</summary>	2020年11月13日</details>

# 8、基于VCSEL神经元的全光神经形态二值卷积算法
- [ ] All-optical neuromorphic binary convolution with a spiking VCSEL neuron for image gradient magnitudes 
时间：2020年11月09日                         第一作者：Yahui Zhang                       [链接](https://arxiv.org/abs/2011.04438).                     
## 摘要：首次提出了一种基于光子脉冲垂直腔面发射激光器（VCSEL）神经元的全光二值卷积方法，并进行了实验验证。从数字图像中提取并使用矩形脉冲进行时间编码的光输入被注入VCSEL神经元中，VCSEL神经元提供快速（<100ps长）脉冲发射数的卷积结果。实验和数值结果表明，采用单脉冲VCSEL神经元实现了二值卷积，全光二值卷积可用于计算图像梯度大小，检测边缘特征，分离源图像中的垂直分量和水平分量。我们还证明了这种全光脉冲二值卷积系统对噪声具有很强的鲁棒性，并且可以处理高分辨率的图像。此外，该系统还具有速度快、能量效率高、硬件实现简单等优点，突出了脉冲光子VCSEL神经元在高速神经图像处理系统和未来光子脉冲卷积神经网络中的应用潜力。
<details>	<summary>英文摘要</summary>	All-optical binary convolution with a photonic spiking vertical-cavity surface-emitting laser (VCSEL) neuron is proposed and demonstrated experimentally for the first time. Optical inputs, extracted from digital images and temporally encoded using rectangular pulses, are injected in the VCSEL neuron which delivers the convolution result in the number of fast (<100 ps long) spikes fired. Experimental and numerical results show that binary convolution is achieved successfully with a single spiking VCSEL neuron and that all-optical binary convolution can be used to calculate image gradient magnitudes to detect edge features and separate vertical and horizontal components in source images. We also show that this all-optical spiking binary convolution system is robust to noise and can operate with high-resolution images. Additionally, the proposed system offers important advantages such as ultrafast speed, high energy efficiency and simple hardware implementation, highlighting the potentials of spiking photonic VCSEL neurons for high-speed neuromorphic image processing systems and future photonic spiking convolutional neural networks. </details>
<details>	<summary>注释</summary>	jxxsy@126.com; antonio.hurtado@strath.ac.uk </details>
<details>	<summary>邮件日期</summary>	2020年11月10日</details>

# 7、用gpu快速模拟高度连接的棘波皮层模型
- [ ] Fast simulations of highly-connected spiking cortical models using GPUs 
时间：2020年11月09日                         第一作者：Bruno Golosio                       [链接](https://arxiv.org/abs/2007.14236).                     
<details>	<summary>邮件日期</summary>	2020年11月10日</details>

# 6、你只刺一次：提高能源效率神经形态推理到神经网络水平的准确性
- [ ] You Only Spike Once: Improving Energy-Efficient Neuromorphic Inference to ANN-Level Accuracy 
时间：2020年11月08日                         第一作者：Srivatsa P                        [链接](https://arxiv.org/abs/2006.09982).                     
<details>	<summary>注释</summary>	10 pages, 4 figures. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. This work is an extended version of the paper accepted to the 2nd Workshop on Accelerated Machine Learning (AccML 2020) </details>
<details>	<summary>邮件日期</summary>	2020年11月10日</details>

# 5、深脉冲神经网络反向传播的校正线性突触后电位函数
- [x] Rectified Linear Postsynaptic Potential Function for Backpropagation in Deep Spiking Neural Networks 
时间：2020年11月04日                         第一作者：Malu Zhang                       [链接](https://arxiv.org/abs/2003.11837).                     
<details>	<summary>注释</summary>	This work has been submitted to the IEEE for possible publication. Copyrightmay be transferred without notice, after which this version may no longer beaccessible </details>
<details>	<summary>邮件日期</summary>	2020年11月05日</details>

# 4、脉冲耦合振荡器网络中的受控微扰诱导开关
- [x] Controlled Perturbation-Induced Switching in Pulse-Coupled Oscillator Networks 
时间：2020年11月02日                         第一作者：Fabio Schittler Neves                        [链接](https://arxiv.org/abs/2011.00888).                     
## 摘要：脉冲耦合系统，如脉冲神经网络，表现出非平凡不变集的形式吸引但不稳定的鞍周期轨道的单位同步成组。这些轨道之间的异宿连接原则上可以支持这些网络中的切换过程，并支持新的神经计算。对于耦合振子的小网络，我们在此研究在何种条件下以及系统对称性如何强制或禁止某些可能由扰动引起的开关跃迁。对于由五个振子组成的网络，我们导出了两个团簇对称性的显式跃迁规则，这些规则偏离了已知的连续耦合振子的跃迁规则。第三种对称产生异宿网络，它由所有具有这种对称性的不稳定吸引子以及它们之间的连接组成。我们的结果表明，脉冲耦合系统能够可靠地产生符合特定转换规则的复杂时空模式。我们简要地讨论了脉冲神经系统计算的可能含义。
<details>	<summary>英文摘要</summary>	Pulse-coupled systems such as spiking neural networks exhibit nontrivial invariant sets in the form of attracting yet unstable saddle periodic orbits where units are synchronized into groups. Heteroclinic connections between such orbits may in principle support switching processes in those networks and enable novel kinds of neural computations. For small networks of coupled oscillators we here investigate under which conditions and how system symmetry enforces or forbids certain switching transitions that may be induced by perturbations. For networks of five oscillators we derive explicit transition rules that for two cluster symmetries deviate from those known from oscillators coupled continuously in time. A third symmetry yields heteroclinic networks that consist of sets of all unstable attractors with that symmetry and the connections between them. Our results indicate that pulse-coupled systems can reliably generate well-defined sets of complex spatiotemporal patterns that conform to specific transition rules. We briefly discuss possible implications for computation with spiking neural systems. </details>
<details>	<summary>邮件日期</summary>	2020年11月03日</details>

# 3、RANC：可重构的神经形态计算体系结构
- [ ] RANC: Reconfigurable Architecture for Neuromorphic Computing 
时间：2020年11月01日                         第一作者：Joshua Mack                       [链接](https://arxiv.org/abs/2011.00624).                     
## 摘要：神经形态结构已经被引入作为能量有效的脉冲神经网络执行的平台。这些体系结构所提供的大规模并行性也引起了非机器学习应用领域的兴趣。为了提升硬件设计者和应用开发者的进入壁垒，我们提出了RANC：一种可重构的神经形态计算体系结构，一个开源的高度灵活的生态系统，通过C++仿真和硬件通过FPGA仿真，能够快速地在软件中对神经形态结构进行实验。我们展示了RANC生态系统的实用性，通过展示其重现IBM的TrueNorth行为的能力，并通过与IBM的Compass模拟环境和已发表文献的直接比较进行验证。RANC允许基于应用程序洞察优化架构，以及原型化可以完全支持新类应用程序的未来神经形态架构。通过基于Alveo U250 FPGA的定量分析，研究了体系结构变化对提高应用程序映射效率的影响，证明了RANC的高度参数化和可配置性。本文介绍了合成孔径雷达分类和矢量矩阵乘法应用的路由后资源使用和吞吐量分析，并展示了一个可扩展到模拟259K个不同神经元和733m个不同突触的神经形态结构。
<details>	<summary>英文摘要</summary>	Neuromorphic architectures have been introduced as platforms for energy efficient spiking neural network execution. The massive parallelism offered by these architectures has also triggered interest from non-machine learning application domains. In order to lift the barriers to entry for hardware designers and application developers we present RANC: a Reconfigurable Architecture for Neuromorphic Computing, an open-source highly flexible ecosystem that enables rapid experimentation with neuromorphic architectures in both software via C++ simulation and hardware via FPGA emulation. We present the utility of the RANC ecosystem by showing its ability to recreate behavior of the IBM's TrueNorth and validate with direct comparison to IBM's Compass simulation environment and published literature. RANC allows optimizing architectures based on application insights as well as prototyping future neuromorphic architectures that can support new classes of applications entirely. We demonstrate the highly parameterized and configurable nature of RANC by studying the impact of architectural changes on improving application mapping efficiency with quantitative analysis based on Alveo U250 FPGA. We present post routing resource usage and throughput analysis across implementations of Synthetic Aperture Radar classification and Vector Matrix Multiplication applications, and demonstrate a neuromorphic architecture that scales to emulating 259K distinct neurons and 73.3M distinct synapses. </details>
<details>	<summary>注释</summary>	18 pages, 12 figures, accepted for publication in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems. For associated source files see https://github.com/UA-RCL/RANC </details>
<details>	<summary>邮件日期</summary>	2020年11月03日</details>

# 2、基于Loihi处理器的mav光流着陆神经形态控制
- [x] Neuromorphic control for optic-flow-based landings of MAVs using the Loihi processor 
时间：2020年11月01日                         第一作者：Julien Dupeyroux                       [链接](https://arxiv.org/abs/2011.00534).                     
## 摘要：像Loihi这样的神经形态处理器为微型飞行器（mav）这样的受限系统提供了一个有希望的替代传统计算模块，使其具有强大、高效和自主的技能，如起飞和着陆、避障和追踪。然而，在机器人平台上使用这种处理器的一个主要挑战是模拟和真实世界之间的现实差距。在这项研究中，我们首次提出了一个完全嵌入式应用的Loihi神经芯片原型在飞行机器人。为了实现自主着陆，提出了一种基于腹部光流场发散的脉冲神经网络（SNN）来计算推力指令。进化是使用PySNN库在基于Python的模拟器中执行的。该网络结构仅由分布在3层中的35个神经元组成。仿真和Loihi之间的定量分析表明，推力设定值的均方根误差低至0.005 g，同时，隐藏层的脉冲序列匹配率为99.8%，输出层的脉冲序列匹配率为99.7%。所提出的方法成功地填补了现实差距，为未来机器人中的神经形态应用提供了重要的见解。补充材料可在https://mavlab.tudelft.nl/loihi/。
<details>	<summary>英文摘要</summary>	Neuromorphic processors like Loihi offer a promising alternative to conventional computing modules for endowing constrained systems like micro air vehicles (MAVs) with robust, efficient and autonomous skills such as take-off and landing, obstacle avoidance, and pursuit. However, a major challenge for using such processors on robotic platforms is the reality gap between simulation and the real world. In this study, we present for the very first time a fully embedded application of the Loihi neuromorphic chip prototype in a flying robot. A spiking neural network (SNN) was evolved to compute the thrust command based on the divergence of the ventral optic flow field to perform autonomous landing. Evolution was performed in a Python-based simulator using the PySNN library. The resulting network architecture consists of only 35 neurons distributed among 3 layers. Quantitative analysis between simulation and Loihi reveals a root-mean-square error of the thrust setpoint as low as 0.005 g, along with a 99.8% matching of the spike sequences in the hidden layer, and 99.7% in the output layer. The proposed approach successfully bridges the reality gap, offering important insights for future neuromorphic applications in robotics. Supplementary material is available at https://mavlab.tudelft.nl/loihi/. </details>
<details>	<summary>邮件日期</summary>	2020年11月03日</details>

# 1、用直接训练的更大的脉冲神经网络进行更深入的研究
- [x] Going Deeper With Directly-Trained Larger Spiking Neural Networks 
时间：2020年10月29日                         第一作者：Hanle Zheng                       [链接](https://arxiv.org/abs/2011.05280).                     
## 摘要：脉冲神经网络（Spiking neural networks，SNNs）在时空信息和事件驱动信号处理的生物似然编码方面有着广阔的应用前景，非常适合于神经形态硬件的节能实现。然而，SNNs独特的工作模式使其比传统网络更难训练。目前，探索高性能深层snn的培养主要有两条途径。第一种方法是将预先训练好的神经网络模型转换为SNN模型，SNN模型通常需要较长的编码窗口才能收敛，并且在训练过程中不能利用时空特征来求解时间任务。另一种是直接在时空域训练snn。但是由于触发函数的二元脉冲活动和梯度消失或爆炸的问题，目前的方法局限于浅层结构，因此难以利用大规模数据集（如ImageNet）。为此，我们提出了一种基于时空反向传播的阈值相关批处理归一化（tdBN）方法，称为STBP-tdBN，它能够直接训练非常深的SNN并在神经形态硬件上有效地实现其推理。通过提出的方法和详细的快捷连接，我们将直接训练的snn从浅层（<10层）扩展到非常深的结构（50层）。在此基础上，从理论上分析了基于块动态等距理论的方法的有效性。最后，我们报告了更高的准确率结果，包括93.15%的CIFAR-10，67.8%的DVS-CIFAR10和67.05%的ImageNet与很少的时间步长。据我们所知，这是第一次在ImageNet上探索直接训练的高性能深度snn。
<details>	<summary>英文摘要</summary>	Spiking neural networks (SNNs) are promising in a bio-plausible coding for spatio-temporal information and event-driven signal processing, which is very suited for energy-efficient implementation in neuromorphic hardware. However, the unique working mode of SNNs makes them more difficult to train than traditional networks. Currently, there are two main routes to explore the training of deep SNNs with high performance. The first is to convert a pre-trained ANN model to its SNN version, which usually requires a long coding window for convergence and cannot exploit the spatio-temporal features during training for solving temporal tasks. The other is to directly train SNNs in the spatio-temporal domain. But due to the binary spike activity of the firing function and the problem of gradient vanishing or explosion, current methods are restricted to shallow architectures and thereby difficult in harnessing large-scale datasets (e.g. ImageNet). To this end, we propose a threshold-dependent batch normalization (tdBN) method based on the emerging spatio-temporal backpropagation, termed "STBP-tdBN", enabling direct training of a very deep SNN and the efficient implementation of its inference on neuromorphic hardware. With the proposed method and elaborated shortcut connection, we significantly extend directly-trained SNNs from a shallow structure ( < 10 layer) to a very deep structure (50 layers). Furthermore, we theoretically analyze the effectiveness of our method based on "Block Dynamical Isometry" theory. Finally, we report superior accuracy results including 93.15 % on CIFAR-10, 67.8 % on DVS-CIFAR10, and 67.05% on ImageNet with very few timesteps. To our best knowledge, it's the first time to explore the directly-trained deep SNNs with high performance on ImageNet. </details>
<details>	<summary>注释</summary>	12 pages, 6 figures, conference or other essential info </details>
<details>	<summary>邮件日期</summary>	2020年11月11日</details>

