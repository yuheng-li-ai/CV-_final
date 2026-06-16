# Phase8 Method Comparison: Multiview vs Text-to-3D vs Single-Image-to-3D

本项目中三个插入物体分别对应三种资产生成方式：Object A 使用真实多视角视频经过 COLMAP 与 2DGS 重建；Object B 使用 threestudio/DreamFusion-SDS 由文本 prompt 生成 3D；Object C 使用单张图像经去背景后输入 Magic123 生成 3D。三种方法最终都能得到可融合资产，但失真来源不同，因此不能只用“是否像”来评价，而需要同时比较几何准确度、纹理细节和计算耗时。

Object A 的优势是相机约束最强。dense COLMAP 注册 148/148 张图像，得到 24125 个稀疏点，平均重投影误差 1.1296 px；dense-half 2DGS 在 19 个评估视角上达到 PSNR 24.7654、SSIM 0.8228、LPIPS 0.3000。因此 A 的图像级 novel-view rendering 是三者中最有真实观测支撑的。但是 A 的问题出现在 2DGS 到 mesh 的导出阶段：mesh_res=1024 的 TSDF 后处理网格达到 12275319 vertices / 23782653 faces / 610.9 MB，包含大量桌面和背景壳层；降低到 mesh_res=256 后仍需 connected-component filtering，最终只选出 component_02 作为可用物体。也就是说，A 的主要失真不是 COLMAP 或 2DGS 渲染失败，而是 2DGS 表示转换为 Blender mesh 时，低纹理表面、桌面接触区域和背景壳层被混在一起。

Object B 的优势是输入成本最低，只需要 prompt，但它的几何真实性依赖 SDS 先验。三种 prompt ablation 中，simple prompt 训练 49:41，detailed prompt 训练 50:03，style-constrained prompt 训练 47:39。最终选择 style-constrained，因为其轮廓和杯柄/杯身稳定性最好，vertices/faces 为 39314/78628，texture resolution 为 1024x1024，geometry_score=4、texture_score=3、fusion_readiness_score=4。B 的主要问题是 SDS 常见的 hallucination、表面粗糙、Janus/ghosting 风险和初始坐标轴任意性。它不像 A 那样有真实多视角约束，但经过 prompt 约束和后续 Blender 位姿枚举后，作为融合物体的可用性较高。

Object C 的优势是保留了单张输入图像的可见外观，计算时间也低于 B。auto-mask 方案 fine training 约 16.724 min，raw/refined 分别约 16.86 和 17.87 min。最终选择 auto_mask，因为它在 raw image、auto background removal、refined mask 三者中取得最好平衡：25874 vertices、50000 faces、2048x2048 albedo，geometry_score=4、texture_score=4、fusion_readiness_score=4。C 的核心限制是单图 3D 的不可观测区域必须由模型推断，导致侧面、背面、厚度和结构容易失真；raw image 会带来背景污染，refined mask 虽然更干净但反而出现更明显的膨胀和碎片化。最终 C 只能以较小尺度放入场景，并在报告中作为单图生成局限性的证据。

总体比较上，多视角重建在真实纹理和相机几何约束上最强，但对输入采集质量和后续 mesh 导出最敏感；文本生成最灵活、最不依赖实物输入，但几何准确性和纹理细节主要由扩散先验决定；单图生成处于二者之间，能保留正面纹理和轮廓，但不可见区域的几何最不可靠。最终融合中，三种方法都出现了不同程度失真：A 是 mesh extraction 与低纹理面失真，B 是 SDS 生成伪影，C 是单图重建的侧/背面推断误差。这一差异正是报告中对比三种资产生成路线的核心结论。
