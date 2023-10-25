# FastRecon:Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction

Paper
---------------

[Paper](https://paperswithcode.com/paper/fastrecon-few-shot-industrial-anomaly)

In industrial anomaly detection, data efficiency and the ability for fast migration across products become the main concerns when developing detection algorithms. Existing methods tend to be data-hungry and work in the one-model-one-category way, which hinders their effectiveness in real-world industrial scenarios. In this paper, we propose a few-shot anomaly detection strategy that works in a low-data regime and can generalize across products at no cost. Given a defective query sample, we propose to utilize a few normal samples as a reference to reconstruct its normal version, where the final anomaly detection can be achieved by sample alignment. Specifically, we introduce a novel regression with distribution regularization to obtain the optimal transformation from support to query features, which guarantees the reconstruction result shares visual similarity with the query sample and meanwhile maintains the property of normal samples. Experimental results show that our method significantly outperforms previous state-of-the-art at both image and pixel-level AUROC performances from 2 to 8-shot scenarios. Besides, with only a limited number of training samples (less than 8 samples), our method reaches competitive performance with vanilla AD methods which are trained with extensive normal samples.

The code will be available soon.

![](captures/main_00.png)

Overview of our method. Feature maps of each query sample and support samples are exacted by a pre-trained encoder. Features from support images are aggregated into a support feature pool. This pool is down-sampled through greedy coreset selection as $S$ to reduce data redundancy and improve inference speed. The coreset $S$ and the original query feature map $Q$ are then input to our proposed regression with distribution regularization as shown in the grey region. An optimal transformation $\bar{W}$ between $S$ and $Q$ is obtained by the regression to make sure the reconstructed sample $\bar{W} S$, denoted as $\bar{Q}$, to share similarity with $Q$ but keeps all the property of normal samples. Finally, we align $\bar{Q}$ and Q for direct comparison to obtaining the anomaly estimation.

![](captures/result_00.png)

Qualitative results of anomaly localization for both MVTec and MPDD datasets. The first row in the red box presents the support sample for each category while the second row indicates the query samples. The results show that our method can provide accurate localization of defect regions even for more complicated patterns in MPDD.
