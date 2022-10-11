<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Single Image Haze Removal Using Dark Channel Prior](#single-image-haze-removal-using-dark-channel-prior)
- [Guided Image Filtering](#guided-image-filtering)
- [实验结果](#实验结果)

<!-- /code_chunk_output -->

## Single Image Haze Removal Using Dark Channel Prior

https://blog.csdn.net/qq_27590277/article/details/106264043

https://www.cnblogs.com/Imageshop/p/3281703.html

__暗通道先验：__ 在绝大多数非天空的局部区域里，某一些像素总会有至少一个颜色通道具有很低的值。

$$J^{dark}(\boldsymbol{x})= \min_{\boldsymbol{y}\in \Omega(\boldsymbol{x})} (\min_{c\in \{r,g,b\}} J^c (\boldsymbol{y})) \to 0$$

__但是__ 有雾图的暗通道不趋于 0。

|原始图像|暗通道图|
|---|---|
|<img src="/img/fog0.png" width=100%>|<img src="/img/fog0_dark.png" width=100%>|

__雾图形成模型：__ I 是有雾图（3N），J 是待恢复的无雾图（3N），A 是环境光成分（3），t 是透射率（N）。共 4N+3 个未知量，N 为图像大小。

$$\boldsymbol{I}(\boldsymbol{x})= \boldsymbol{J}(\boldsymbol{x})t(\boldsymbol{x}) + \boldsymbol{A}(1-t(\boldsymbol{x}))$$

__假设 A 已知，先求 t。__ 应用暗通道先验，等式右边第一项趋于 0。

$$\min_{\boldsymbol{y}\in \Omega(\boldsymbol{x})} (\min_{c} \frac{I^c (\boldsymbol{y})}{A^c}) = \tilde{t}(\boldsymbol{x}) \min_{\boldsymbol{y}\in \Omega(\boldsymbol{x})} (\min_{c} \frac{J^c (\boldsymbol{y})}{A^c}) + 1 - \tilde{t}(\boldsymbol{x})$$

__引入修正因子：__ $\omega = 0.95$。

$$\Rightarrow \tilde{t}(\boldsymbol{x}) = 1 - \omega \min_{\boldsymbol{y}\in \Omega(\boldsymbol{x})} (\min_{c} \frac{I^c (\boldsymbol{y})}{A^c})$$

__再获取 A。具体步骤如下：__
1. 从暗通道图中按照亮度的大小取前 0.1% 的像素。
2. 在 I 的这些像素中，寻找亮度最大值，作为 A。

__最终恢复 J。__ $t_0$ = 0.1，用于防止 J 过大。

$$\boldsymbol{J}(\boldsymbol{x}) = \frac{\boldsymbol{I}(\boldsymbol{x})-\boldsymbol{A}}{\max(t(\boldsymbol{x}),t_0)} + \boldsymbol{A}$$

__为了提升效果，对估计的 $\tilde{t}$ 使用引导图像滤波。__

|滤波前|滤波后|
|---|---|
|<img src="/img/fog0_t.png" width=100%>|<img src="/img/fog0_t2.png" width=100%>|

## Guided Image Filtering

__名词定义：__
|名称|符号|解释|
|---|---|---|
|引导图|I|我们令有雾图的灰度图为 I。|
|滤波输入|p|我们令投射率图 $\tilde{t}$ 为 p。|
|输出|q|得到边缘细节更加准确的 $\tilde{t}_{new}$。|

__基本假设:__ q 与 I 在以 k 为中心的窗口 $\omega_k$ 内局部线性相关。（这保证了 q 与 I 的边缘特征一致。）

$$q_i = a_k I_i + b_k, \forall i\in\omega_k$$

__定义损失函数：__ $\epsilon$ 用于惩罚过大的 $a_k$。

$$E(a_k,b_k) = \sum_{i\in\omega_k}((a_k I_i + b_k - p_i)^2+\epsilon a_k^2)$$

__分别对 $a_k, b_k$ 求偏导，令偏导数为 0。__ $\mu_k,\sigma^2_k$ 代表 I 在 $\omega_k$ 中的均值与方差。$|\omega|$ 代表窗口 $\omega_k$ 的大小。

$$\Rightarrow 
\left\{
\begin{aligned}
a_k&=\frac{\frac{1}{|\omega|}\sum_{i\in\omega_k}I_ip_i - \mu_k\bar{p}_k}{\sigma_k^2 + \epsilon}\\
b_k&=\bar{p}_k - a_k \mu_k\\
\end{aligned}
\right.$$

__最终对 q 取均值：__

$$q_i = \frac{1}{\omega} \sum_{k|i\in\omega_k}(a_kI_i+b_k)$$

## 实验结果

对于 $840\times 560$ 的输入：
去雾耗时 __10.6 ms__（窗口大小 $9\times 9$）；
引导图像滤波耗时 __11.2 ms__（窗口大小 $33\times 33$）。

|有雾图|去雾图|
|---|---|
|<img src="/img/fog0.png" width=100%>|<img src="/img/fog0_out.png" width=100%>|
|<img src="/img/fog1.png" width=100%>|<img src="/img/fog1_out.png" width=100%>|
|<img src="/img/fog2.png" width=100%>|<img src="/img/fog2_out.png" width=100%>|
|<img src="/img/endo.png" width=100%>|<img src="/img/endo_out.png" width=100%>|

