摘要： 遥感图像语义分割是遥感领域的核心技术之一，在城市规划、环境监测、灾害评估等领域具有重要的应用价值。随着遥感传感器分辨率的不断提高，高分辨率遥感图像中包含的地物类型日益多样、尺度差异显著、背景环境愈加复杂，对语义分割算法的特征提取能力和计算效率提出了更高的要求。近年来，状态空间模型（State Space Model, SSM）凭借其线性计算复杂度和全局建模能力，为遥感图像分割提供了新的技术路径。然而，现有基于SSM的分割方法仍然存在全局建模与局部细节增强相互耦合、注意力机制冗余堆叠导致计算开销增大等问题。同时，单一模态数据往往难以全面表征复杂地物的特征信息，光学图像与合成孔径雷达（SAR）图像在成像机理上的本质差异也给跨模态特征融合带来了显著挑战。

为此，本文围绕遥感图像语义分割中的上述问题开展了以下研究：

（1）针对现有基于SSM的分割方法中全局上下文建模与局部细节增强在单一路径中耦合处理、注意力机制冗余堆叠等问题，本文提出了基于双路径解耦的语义分割网络DP-UNet。该网络在解码器中设计了双路径解耦VSS模块（DVSS），采用"共享基座、分别增强"的策略：以二维选择性扫描模块（SS2D）作为共享基座生成基础特征，全局路径直接保留该基础特征以维持完整的上下文语义信息，局部路径则在此基础上通过高效通道注意力（ECA）进行通道重标定，并利用参数域可调卷积（PMC）在参数空间引入可学习的中心抑制先验以增强对边缘与纹理细节的感知能力，最终通过自适应路径融合门控（APFG）实现两条路径特征的自适应整合。在多尺度特征融合阶段，设计了轻量化多尺度空间核模块（MSK），在压缩通道空间内通过多尺度卷积提取空间模式并以纯空间注意力进行特征增强，有效避免了通道注意力在网络中的冗余施加。实验结果表明，DP-UNet在ISPRS Potsdam数据集上的平均交并比达到86.71%、总体精度达到91.57%，在ISPRS Vaihingen数据集上的平均交并比达到83.84%、总体精度达到91.43%，在LoveDA数据集上的平均交并比达到53.21%，模型参数量为11.30M，计算量为44.26G FLOPs，能够有效解决全局与局部特征耦合及注意力冗余问题，在分割精度与计算效率之间取得了良好的平衡。

（2）针对单一模态遥感数据特征表征能力有限、光学图像与SAR图像模态差异显著导致特征融合不充分等问题，本文在DP-UNet的基础上提出了基于跨模态融合的多模态语义分割网络。该网络设计了双分支编码器分别提取光学图像和SAR图像的模态专属特征，并提出跨模态多尺度融合模块（CrossModalMSK），通过模态内多尺度空间特征提取捕获各模态不同尺度的空间信息，通过模态间交叉注意力机制建模两种模态特征之间的关联性并提取互补信息，实现不同模态、不同尺度特征的有效对齐与协同融合。解码器端复用DVSS模块对融合后的多模态特征进行渐进式上采样与精细化重建。实验结果表明，该网络在WHU-OPT-SAR数据集上的[占位：平均交并比达到XX%、总体精度达到XX%]，能够有效融合光学与SAR图像的互补信息，显著提升了复杂场景下的地物分割性能。

关键词： 遥感图像语义分割；状态空间模型；双路径解耦；参数域可调卷积；多模态融合；跨模态注意力

第一章 绪论
1.1 研究背景与意义
随着遥感技术的飞速发展，尤其是卫星遥感和无人机遥感平台的日益普及，遥感图像的空间分辨率不断提高，所获取的图像数据量也在急剧增长[1]。遥感图像能够为地理环境研究提供广阔的观测视角，已被广泛应用于土地覆盖分类[2-3]、环境监测[4]、灾害评估[5]、城市规划[6]和精准农业[7]等领域，为国民经济建设和社会发展提供了重要的数据支撑。因此，如何从海量遥感图像中高效、准确地提取地物信息，已成为遥感领域亟待解决的关键问题。

语义分割是计算机视觉领域中的一项核心技术，旨在对图像中的每一个像素进行类别标注，在遥感图像分析中扮演着至关重要的角色[8]。该技术能够实现对建筑物、道路、植被、水体等不同地物类型的自动识别与精细划分，为上述应用场景提供高效的决策支持。近年来，基于深度学习的语义分割方法取得了显著进展，从早期的全卷积网络（Fully Convolutional Network, FCN）[9]到编码器-解码器架构的U-Net[10]，再到融合多尺度上下文信息的DeepLabV3+[11]，分割精度和效率均得到了大幅提升。

然而，高分辨率遥感图像的语义分割仍然面临诸多挑战。首先，遥感图像中地物类型多样、尺度差异显著，同一场景中既包含大面积的农田、水体等宏观地物，也存在车辆、小型建筑等细小目标，要求模型同时具备全局上下文建模能力和局部细节捕获能力。其次，遥感图像背景复杂，地物边界模糊，不同类别之间存在"同物异谱"和"异物同谱"现象，进一步加大了精确分割的难度。此外，高分辨率图像的数据量庞大，对算法的计算效率提出了更高的要求，如何在精度与效率之间取得平衡是一个重要的研究课题。

在模型架构方面，基于Transformer[12]的方法通过自注意力机制有效捕获了长距离依赖关系，在语义分割任务中展现出优越的性能。然而，自注意力机制的计算复杂度与输入序列长度呈二次方关系，当应用于高分辨率遥感图像时，巨大的计算开销严重制约了其实际部署。近年来，状态空间模型（State Space Model, SSM）[13]凭借其线性计算复杂度和全局序列建模能力，为解决这一矛盾提供了新的技术路径。以Mamba[14]为代表的选择性状态空间模型通过输入依赖的选择机制，在保持线性复杂度的同时实现了对长序列的高效建模，已在自然语言处理和计算机视觉等领域展现出巨大潜力。在遥感图像分割中，基于Mamba的方法[15]已初步验证了其在处理大规模场景时兼顾精度与效率的可行性。然而，现有方法仍存在全局上下文建模与局部细节增强在单一处理路径中相互耦合、注意力机制在网络不同阶段冗余堆叠等问题，限制了模型性能的进一步提升。

此外，在实际遥感应用中，单一模态的数据往往难以全面表征复杂地物的特征信息。光学图像虽然具有丰富的光谱和纹理信息，能够直观反映地表覆盖物的属性和类型，但易受云层遮挡、光照变化等环境因素的影响，在恶劣天气条件下成像质量显著下降。合成孔径雷达（Synthetic Aperture Radar, SAR）作为一种主动微波遥感技术，具备全天候、全天时的成像能力，能够穿透云层和部分地表覆盖物，提供稳定可靠的地表观测信息[16]。然而，SAR图像受相干斑噪声干扰严重，且缺乏光谱信息，对地物类型的区分能力有限。通过融合光学图像与SAR图像，可以充分发挥两种模态的互补优势，弥补单一数据源的不足，从而全面、准确地获取地表特征信息。然而，光学图像与SAR图像在成像机理、数据特性和信息表达形式上存在本质差异，如何有效弥合这种模态鸿沟、实现跨模态特征的充分融合，仍然是一个具有挑战性的研究问题。

综上所述，针对遥感图像语义分割中存在的全局与局部特征建模耦合、计算效率受限以及多模态特征融合困难等问题，开展基于状态空间模型的高效分割方法研究，并探索光学与SAR图像的跨模态融合策略，对于推动遥感图像智能解译技术的发展具有重要的理论意义和应用价值。

1.2 国内外研究现状
遥感图像语义分割是遥感与计算机视觉交叉领域的重要研究方向。近年来，随着深度学习技术的快速发展，该领域的研究方法经历了从传统方法到深度学习方法、从卷积神经网络到Transformer和状态空间模型、从单模态到多模态的演进过程。本节将从基于卷积神经网络的方法、基于Transformer的方法、基于状态空间模型的方法以及多模态融合方法四个方面，对国内外研究现状进行综述。

1.2.1 基于卷积神经网络的语义分割方法
卷积神经网络（Convolutional Neural Network, CNN）是深度学习在语义分割领域最早也是最广泛应用的基础架构。2015年，Shelhamer等人[9]提出的全卷积网络（FCN）首次将分类网络中的全连接层替换为卷积层，实现了端到端的像素级预测，奠定了深度学习语义分割的基础范式。然而，FCN通过连续的池化操作导致特征图分辨率大幅降低，分割结果的边界较为粗糙，空间细节信息损失严重。

为解决空间信息丢失的问题，Ronneberger等人[10]提出了U-Net架构。U-Net采用对称的编码器-解码器结构，并通过跳跃连接将编码器各层的高分辨率特征传递至对应的解码器层，有效融合了深层语义信息与浅层空间细节，在医学图像分割中取得了突破性成果，并被广泛应用于遥感图像分割领域。在此基础上，Zhou等人[17]提出了UNet++，通过密集的嵌套跳跃连接进一步增强了不同层级特征之间的融合效果。

在多尺度特征提取方面，Chen等人[18]提出了DeepLab系列方法。其中，DeepLabV2引入了空洞空间金字塔池化（Atrous Spatial Pyramid Pooling, ASPP）模块，通过并行应用不同膨胀率的空洞卷积，在不增加参数量的前提下有效扩大了感受野，增强了模型对不同尺度目标的建模能力。DeepLabV3+[11]在此基础上结合了编码器-解码器结构，进一步提升了分割边界的精细度。Zhao等人[19]提出的金字塔场景解析网络（PSPNet）通过金字塔池化模块整合不同尺度的全局上下文信息，有效缓解了对大尺度物体和复杂背景的误判问题。

在模型轻量化方面，Badrinarayanan等人[20]提出了SegNet，通过利用编码阶段的池化索引进行非参数化上采样，显著减少了模型参数量。Yu等人[21]提出了双边分割网络（BiSeNet），采用双分支设计分别提取空间细节特征和语义上下文特征，在分割精度与推理速度之间取得了良好的平衡。Howard等人[22]提出的MobileNetV3通过引入深度可分离卷积和反转残差结构，大幅降低了骨干网络的计算开销，为轻量化语义分割模型提供了高效的特征提取基础。

在遥感图像分割领域，基于CNN的方法同样取得了丰富成果。Wang等人[23]提出了UNetFormer，在U-Net的解码器中引入了高效的全局-局部注意力机制，有效提升了遥感图像中多尺度地物的分割精度。Li等人[24]提出的ABCNet通过注意力增强的双边网络结构，实现了对遥感图像中复杂场景的精细分割。

尽管基于CNN的方法在遥感图像分割中取得了显著成效，但卷积操作固有的局部感受野限制了其对长距离依赖关系的建模能力。虽然可以通过堆叠多层卷积或使用空洞卷积来扩大感受野，但这种扩展方式效率较低，且难以有效捕获图像中远距离像素之间的语义关联，在处理大面积均质区域或需要全局上下文理解的场景时表现受限。

1.2.2 基于Transformer的语义分割方法
Transformer[12]最初为自然语言处理任务设计，其核心的自注意力机制能够直接建模序列中任意两个位置之间的依赖关系，从根本上克服了CNN局部感受野的限制。Dosovitskiy等人[25]首次将Transformer应用于计算机视觉领域，提出了Vision Transformer（ViT），通过将图像划分为固定大小的图像块并进行序列化处理，在图像分类任务中取得了优异的性能。

在语义分割领域，Zheng等人[26]提出了SETR（Segmentation Transformer），将ViT作为编码器，首次验证了纯Transformer架构在语义分割任务中的可行性。Xie等人[27]提出了SegFormer，设计了层次化的Transformer编码器和轻量级的全多层感知机（All-MLP）解码器，在多个分割基准上取得了精度与效率的良好平衡。Liu等人[28]提出了Swin Transformer，通过滑动窗口机制将自注意力的计算限制在局部窗口内，并通过窗口间的移位实现跨窗口信息交互，将计算复杂度从二次方降低至线性，极大地提升了Transformer处理高分辨率图像的效率。

在遥感图像分割中，基于Transformer的方法也得到了广泛应用。Wang等人[29]提出了BANet，利用Transformer捕获遥感图像中的全局上下文信息，有效提升了对复杂场景的理解能力。Strudel等人[30]提出了Segmenter，采用纯Transformer架构进行语义分割，通过全局注意力机制增强了对大范围地物的识别能力。

然而，尽管Transformer在建模长距离依赖方面具有显著优势，但自注意力机制的计算复杂度与输入分辨率呈二次方关系。即使采用Swin Transformer等局部窗口策略进行优化，当应用于高分辨率遥感图像时，其计算开销仍然较大，在资源受限的实际应用场景中面临部署困难。此外，Transformer对局部细节的建模能力相对较弱，在处理遥感图像中精细的地物边界和小目标时仍存在不足。

1.2.3 基于状态空间模型的语义分割方法
状态空间模型（State Space Model, SSM）源于控制理论和信号处理领域，通过隐状态的递推更新实现对序列数据的建模[13]。与Transformer的自注意力机制不同，SSM通过线性递推关系处理序列，其计算复杂度与序列长度呈线性关系，在处理长序列时具有显著的效率优势。

Gu等人[31]提出了结构化状态空间序列模型（S4），通过对SSM进行结构化参数化，首次在长距离序列建模任务中取得了优异的性能。在此基础上，Gu和Dao[14]进一步提出了Mamba模型，引入了选择性状态空间机制（Selective State Space），通过输入依赖的参数化方式使模型能够根据输入内容自适应地选择性保留或遗忘信息，同时设计了硬件感知的并行扫描算法，在保持线性计算复杂度的同时显著提升了模型的表达能力和计算效率。

Mamba在自然语言处理任务中展现出了与Transformer相当甚至超越的性能后，迅速被引入计算机视觉领域。Liu等人[32]提出了VMamba，设计了二维选择性扫描模块（SS2D），通过四个方向的交叉扫描路径将二维图像特征展开为一维序列进行处理，有效地将Mamba的序列建模能力扩展到二维视觉任务中。Zhu等人[33]提出了Vision Mamba（Vim），采用双向状态空间模型处理图像序列，在图像分类任务中实现了优异的性能。

在语义分割领域，基于Mamba的方法已展现出巨大潜力。Xing等人[34]提出了SegMamba，将Mamba引入三维医学图像分割任务，通过长距离序列建模有效处理了体数据中的空间依赖关系。Ruan和Xiang[35]提出了VM-UNet，构建了基于Vision Mamba的纯SSM架构，在医学图像分割中验证了SSM作为编码器-解码器骨干的有效性。

在遥感图像分割领域，基于Mamba的方法同样受到了广泛关注。Chen等人[36]提出了RS3Mamba，将Mamba与CNN相结合用于遥感图像语义分割，通过辅助的选择性状态空间模块增强了多尺度特征的提取能力。Shi等人[15]提出了CM-UNet，将基于CNN的编码器与基于Mamba的解码器相结合，在多个遥感分割数据集上取得了优异的性能。

尽管现有基于Mamba的遥感分割方法已展现出良好的精度与效率平衡，但仍存在一些值得关注的问题。一方面，现有方法的解码器通常采用单一路径的序列化设计，全局上下文建模与局部细节增强在同一处理流程中耦合执行，两种本质不同的子任务共享参数空间，可能相互制约，限制了各自的优化潜力。另一方面，部分方法在网络的多个阶段重复施加功能相似的注意力操作，导致计算资源的冗余消耗，模型复杂度和推理开销随之增大，影响了实际部署效率。如何在保持全局建模能力的同时高效增强局部细节，以及如何合理配置注意力机制以避免冗余，是当前基于SSM的遥感分割方法需要进一步解决的问题。

1.2.4 多模态遥感图像融合分割方法
在遥感应用中，不同传感器获取的多模态数据能够提供互补的地表信息，多模态数据融合已成为提升语义分割性能的重要手段[37]。根据融合阶段的不同，多模态融合方法通常可分为早期融合、中期融合和晚期融合三种策略[38]。早期融合在输入层直接拼接不同模态的数据，方法简单但难以处理模态间的异质性。晚期融合在决策层对各模态独立预测的结果进行集成，能够避免模态间的相互干扰，但缺乏特征层面的交互。中期融合在特征提取阶段对不同模态的特征进行交互与整合，能够在保持模态特异性的同时实现特征互补，是目前研究最为广泛的融合策略。

在光学与高程数据的融合方面，Hazirbas等人[39]提出了FuseNet，采用双分支编码器分别处理RGB图像和深度图像，并将深度分支的多层特征逐级融合至RGB分支的编码器中。然而，这种单向融合方式未能充分利用双向的模态交互信息。Audebert等人[40]采用两个独立的CNN网络分别处理光学图像和数字表面模型（DSM）图像，并通过逐元素相加的方式进行模态特征融合，方法简单但融合深度有限。Ma等人[41]提出了AMM-FuseNet，采用通道注意力机制和密集连接的空间金字塔池化增强多模态特征的表征能力，在一定程度上缓解了简单融合策略的局限性。

在光学与SAR图像的融合分割方面，由于两种模态在成像机理上存在本质差异——光学图像反映地物的光谱反射特性，而SAR图像通过后向散射系数表征地表结构特征——跨模态特征融合面临更大的挑战。Li等人[42]在WHU-OPT-SAR数据集[43]上提出了MCANet，通过多模态交叉注意力机制提取光学图像和SAR图像中的互补特征，并设计低高层特征融合模块优化分割结果，验证了多模态融合对分割性能的显著提升。Feng等人[44]提出了CMGFNet，通过跨模态门控融合机制自适应地调节不同模态特征的贡献权重，有效应对了模态间语义不一致的问题。Zhang等人[45]提出了CMX，设计了交叉模态特征校正模块，通过双向特征交互实现不同模态之间的有效融合，其通用的跨模态融合范式对光学与SAR图像的融合也具有重要的借鉴意义。

尽管上述多模态融合方法在一定程度上提升了分割性能，但仍存在一些需要解决的问题。首先，多数方法在融合过程中采用简单的特征拼接或逐元素相加策略，难以充分建模不同模态特征之间的关联性和互补性。其次，光学与SAR图像在分辨率、噪声水平和语义表达上的差异，使得不同尺度特征的跨模态对齐和融合仍然具有挑战性。此外，如何在多模态融合框架中高效地整合先进的序列建模技术，以同时实现全局上下文的跨模态交互和局部细节的精细增强，是一个值得深入探索的研究方向。

1.3 主要研究内容
针对上述研究现状中存在的问题，本文围绕基于状态空间模型的遥感图像语义分割方法展开研究，致力于解决现有方法中全局与局部特征建模耦合、注意力机制冗余以及多模态特征融合不充分等问题。具体研究内容如下：

（1）针对现有基于SSM的遥感分割方法中全局上下文建模与局部细节增强在单一路径中耦合处理、注意力机制在不同网络阶段冗余堆叠等问题，本文提出了基于双路径解耦的语义分割网络DP-UNet。在解码器中设计了双路径解耦VSS模块（DVSS），采用"共享基座、分别增强"的策略，以SS2D模块作为共享基座生成基础特征，全局路径保留完整的上下文语义信息，局部路径通过高效通道注意力（ECA）和参数域可调卷积（PMC）增强边缘与纹理等细节特征，并通过自适应路径融合门控（APFG）实现两路径特征的自适应整合。同时，设计了轻量化多尺度空间核模块（MSK），在压缩通道空间内以纯空间注意力完成多尺度特征融合，避免通道注意力的冗余施加。

（2）针对单一模态遥感数据特征表征能力有限、光学与SAR图像模态差异显著导致特征融合不充分等问题，本文在DP-UNet的基础上进一步扩展至多模态融合场景，提出了基于跨模态融合的语义分割网络。设计了双分支编码器分别提取光学图像和SAR图像的模态专属特征，提出跨模态多尺度融合模块（CrossModalMSK），通过模态内多尺度空间特征提取和模态间交叉注意力机制，实现不同模态、不同尺度特征之间的有效对齐与互补融合。解码器端复用DVSS模块对融合后的多模态特征进行渐进式上采样与精细化重建。

（3）为验证所提方法的有效性，在ISPRS Potsdam、ISPRS Vaihingen、LoveDA和WHU-OPT-SAR四个公开遥感数据集上开展了系统性的实验评估，包括与多种代表性方法的对比实验、关键模块的消融实验以及模型复杂度分析，全面验证了所提方法在分割精度和计算效率方面的优势。

1.4 本文组织结构
本文聚焦于基于状态空间模型的遥感图像语义分割方法研究，全文共分为五章，各章主要内容安排如下：

第一章为绪论。首先介绍了遥感图像语义分割的研究背景与意义；其次分别从基于卷积神经网络的方法、基于Transformer的方法、基于状态空间模型的方法以及多模态融合方法四个方面综述了国内外研究现状；接着根据当前研究中存在的问题，确定了本文的主要研究内容；最后介绍了论文的组织结构。

第二章为相关理论与技术基础。首先介绍了语义分割任务的基本概念和编码器-解码器框架；其次阐述了状态空间模型的基本原理，包括从S4到Mamba的演进过程以及二维选择性扫描机制；然后介绍了本文涉及的注意力机制和多模态遥感数据的基本特性；接着介绍了本文实验所用的四个公开数据集和语义分割任务的评价指标；最后对本章内容进行小结。

第三章为基于双路径解耦的单模态遥感图像语义分割方法。首先分析了现有基于SSM的分割方法中存在的全局-局部耦合和注意力冗余问题；然后详细介绍了DP-UNet的整体架构以及DVSS模块、PMC模块和MSK模块的设计原理；接着在ISPRS Potsdam、ISPRS Vaihingen和LoveDA三个数据集上进行实验分析；最后对本章内容进行小结。

第四章为基于跨模态融合的多模态遥感图像语义分割方法。首先分析了单模态分割方法的局限性以及光学与SAR图像跨模态融合的挑战；然后详细介绍了多模态融合网络的整体架构以及跨模态多尺度融合模块的设计；接着在WHU-OPT-SAR数据集上进行实验分析；最后对本章内容进行小结。

第五章为总结与展望。总结了本文的主要研究工作和创新点，分析了当前工作的局限性，并对未来的研究方向进行了展望。

参考文献
[1] Zhu X X, Tuia D, Mou L, et al. Deep learning in remote sensing: A comprehensive review and list of resources[J]. IEEE Geoscience and Remote Sensing Magazine, 2017, 5(4): 8-36.

[2] Kussul N, Lavreniuk M, Skakun S, et al. Deep learning classification of land cover and crop types using remote sensing data[J]. IEEE Geoscience and Remote Sensing Letters, 2017, 14(5): 778-782.

[3] Zhang C, Sargent I, Pan X, et al. An object-based convolutional neural network (OCNN) for urban land use classification[J]. Remote Sensing of Environment, 2018, 216: 57-70.

[4] Sublime J, Kalinicheva E. Automatic post-disaster damage mapping using deep-learning techniques for change detection: Case study of the Tohoku tsunami[J]. Remote Sensing, 2019, 11(9): 1123.

[5] Gupta R, Hosfelt R, Saber S, et al. xBD: A dataset for assessing building damage from satellite imagery[J]. arXiv preprint arXiv:1911.09296, 2019.

[6] Wurm M, Stark T, Zhu X X, et al. Semantic segmentation of slums in satellite images using transfer learning on fully convolutional neural networks[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2019, 150: 59-69.

[7] Kamilaris A, Prenafeta-Boldú F X. Deep learning in agriculture: A survey[J]. Computers and Electronics in Agriculture, 2018, 147: 70-90.

[8] Yuan X, Shi J, Gu L. A review of deep learning methods for semantic segmentation of remote sensing imagery[J]. Expert Systems with Applications, 2021, 169: 114417.

[9] Shelhamer E, Long J, Darrell T. Fully convolutional networks for semantic segmentation[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(4): 640-651.

[10] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation[C]//Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2015: 234-241.

[11] Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European Conference on Computer Vision (ECCV). Springer, 2018: 801-818.

[12] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems (NeurIPS), 2017: 5998-6008.

[13] Gu A, Dao T, Ermon S, et al. HiPPO: Recurrent memory with optimal polynomial projections[C]//Advances in Neural Information Processing Systems (NeurIPS), 2020: 1474-1487.

[14] Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces[J]. arXiv preprint arXiv:2312.00752, 2023.

[15] Shi C, Chen Y, Wang G. CM-UNet: Hybrid CNN-Mamba UNet for remote sensing image semantic segmentation[J]. arXiv preprint arXiv:2405.10530, 2024.

[16] Moreira A, Prats-Iraola P, Younis M, et al. A tutorial on synthetic aperture radar[J]. IEEE Geoscience and Remote Sensing Magazine, 2013, 1(1): 6-43.

[17] Zhou Z, Siddiquee M M R, Tajbakhsh N, et al. UNet++: A nested U-Net architecture for medical image segmentation[C]//Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. Springer, 2018: 3-11.

[18] Chen L C, Papandreou G, Kokkinos I, et al. DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018, 40(4): 834-848.

[19] Zhao H, Shi J, Qi X, et al. Pyramid scene parsing network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017: 2881-2890.

[20] Badrinarayanan V, Kendall A, Cipolla R. SegNet: A deep convolutional encoder-decoder architecture for image segmentation[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(12): 2481-2495.

[21] Yu C, Wang J, Peng C, et al. BiSeNet: Bilateral segmentation network for real-time semantic segmentation[C]//Proceedings of the European Conference on Computer Vision (ECCV). Springer, 2018: 325-341.

[22] Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019: 1314-1324.

[23] Wang L, Li R, Zhang C, et al. UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2022, 190: 196-214.

[24] Li R, Zheng S, Zhang C, et al. ABCNet: Attentive bilateral contextual network for efficient semantic segmentation of fine-resolution remotely sensed imagery[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2021, 181: 84-98.

[25] Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[C]//Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[26] Zheng S, Lu J, Zhao H, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021: 6881-6890.

[27] Xie E, Wang W, Yu Z, et al. SegFormer: Simple and efficient design for semantic segmentation with transformers[C]//Advances in Neural Information Processing Systems (NeurIPS), 2021: 12077-12090.

[28] Liu Z, Lin Y, Cao Y, et al. Swin Transformer: Hierarchical vision transformer using shifted windows[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021: 10012-10022.

[29] Wang L, Li R, Duan C, et al. A novel transformer based semantic segmentation scheme for fine-resolution remote sensing images[J]. IEEE Geoscience and Remote Sensing Letters, 2022, 19: 1-5.

[30] Strudel R, Garcia R, Laptev I, et al. Segmenter: Transformer for semantic segmentation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021: 7262-7272.

[31] Gu A, Goel K, Ré C. Efficiently modeling long sequences with structured state spaces[C]//Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[32] Liu Y, Tian Y, Zhao Y, et al. VMamba: Visual state space model[J]. arXiv preprint arXiv:2401.10166, 2024.

[33] Zhu L, Liao B, Zhang Q, et al. Vision Mamba: Efficient visual representation learning with bidirectional state space model[J]. arXiv preprint arXiv:2401.09417, 2024.

[34] Xing Z, Ye T, Yang Y, et al. SegMamba: Long-range sequential modeling Mamba for 3D medical image segmentation[J]. arXiv preprint arXiv:2401.13560, 2024.

[35] Ruan J, Xiang S. VM-UNet: Vision Mamba UNet for medical image segmentation[J]. arXiv preprint arXiv:2402.02491, 2024.

[36] Chen T, Zhu L, Niu B, et al. RS3Mamba: Visual state space model for remote sensing image semantic segmentation[J]. IEEE Geoscience and Remote Sensing Letters, 2024.

[37] Gao L, Hong D, Yao J, et al. Spectral superresolution of multispectral imagery with joint sparse and low-rank learning[J]. IEEE Transactions on Geoscience and Remote Sensing, 2021, 59(3): 2269-2280.

[38] Zhang J. Multi-source remote sensing data fusion: Status and trends[J]. International Journal of Image and Data Fusion, 2010, 1(1): 5-24.

[39] Hazirbas C, Ma L, Domokos C, et al. FuseNet: Incorporating depth into semantic segmentation via fusion-based CNN architecture[C]//Proceedings of the Asian Conference on Computer Vision (ACCV). Springer, 2016: 213-228.

[40] Audebert N, Le Saux B, Lefèvre S. Beyond RGB: Very high resolution urban remote sensing with multimodal deep networks[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2018, 140: 20-32.

[41] Ma X, Mao Z, Li X, et al. AMM-FuseNet: Attention-based multi-modal image fusion network for land cover mapping[J]. Remote Sensing, 2022, 14(18): 4458.

[42] Li X, Lei L, Sun Y, et al. Multimodal bilinear fusion network with second-order attention-based channel selection for land cover classification[J]. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2020, 13: 1064-1077.

[43] Li X, Zhang G, Cui H, et al. MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification[J]. International Journal of Applied Earth Observation and Geoinformation, 2022, 106: 102638.

[44] Feng J, Wang L, Yu H, et al. CMGFNet: A cross-modal gated fusion network for building extraction from very high-resolution remote sensing images and GIS data[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2022, 188: 61-76.

[45] Zhang J, Liu H, Yang K, et al. CMX: Cross-modal fusion for RGB-X semantic segmentation with transformers[J]. IEEE Transactions on Intelligent Transportation Systems, 2023, 24(12): 14679-14694.

