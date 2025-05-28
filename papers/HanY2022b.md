# Identifying outliers in astronomical images with unsupervised machine learning

Yang Han1? , Zhiqiang Zou1,2? , Nan Li3,4??, Yanli Chen1,2

- 1 School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, Jiangsu, People's Republic of China
- 2 Jiangsu Key Laboratory of Big Data Security and Intelligent Processing, Nanjing, Jiangsu, People's Republic of China
- 3 Key Laboratory of Optical Astronomy, National Astronomical Observatories, Chinese Academy of Sciences, Beijing, People's Republic of China
- 4 University of Chinese Academy of Sciences, Beijing, People's Republic of China

Received 20XX Month Day; accepted 20XX Month Day

Abstract Astronomical outliers, such as unusual, rare or unknown types of astronomical objects or phenomena, constantly lead to the discovery of genuinely unforeseen knowledge in astronomy. More unpredictable outliers will be uncovered in principle with the increment of the coverage and quality of upcoming survey data. However, it is a severe challenge to mine rare and unexpected targets from enormous data with human inspection due to a significant workload. Supervised learning is also unsuitable for this purpose since designing proper training sets for unanticipated signals is unworkable. Motivated by these challenges, we adopt unsupervised machine learning approaches to identify outliers in the data of galaxy images to explore the paths for detecting astronomical outliers. For comparison, we construct three methods, which are built upon the k-nearest neighbors (KNN), Convolutional Auto-Encoder (CAE) + KNN, and CAE + KNN + Attention Mechanism (attCAE KNN) separately. Testing sets are created based on the Galaxy Zoo image data published online to evaluate the performance of the above methods. Results show that attCAE KNN achieves the best recall (78%), which is 53% higher than the classical KNN method and 22% higher than CAE+KNN. The efficiency

of attCAE KNN (10 minutes) is also superior to KNN (4 hours) and equal to CAE+KNN (10 minutes) for accomplishing the same task. Thus, we believe it is feasible to detect astronomical outliers in the data of galaxy images in an unsupervised manner. Next, we will apply attCAE KNN to available survey datasets to assess its applicability and reliability.

Key words: Outlier Detection, Unsupervised Learning, Auto-Encoder, Galaxy Zoo, KNN

# 1 INTRODUCTION

Astronomy is stepping into the big data era with the upcoming large-scale surveys (Lochner & Bassett 2021), e.g., Euclid 1 , LSST 2 and CSST 3 . Mining knowledge from enormous astronomical datasets has become critical for astrophysical and cosmological investigations . Typically, data mining in astronomy includes object classification, dependency detection, class description, and anomalies/outlier detection. The first three categories of tasks are problem-driven, i.e., once the goals are well-defined, the tasks can be handled in a supervised manner by involving well-designed training sets. These tasks help improve the accuracy and precision of the models for describing mainstream objects, and relevant approaches are relatively mature and widely applied in astronomy (Lukic et al. 2019; Zhu et al. 2019; Cheng et al. 2020; Gupta et al. 2022; Chen et al. 2022; Zhang et al. 2022). On the other hand, astronomical anomalies/outliers constantly lead to unforeseen knowledge in astronomy, which may trigger revolutionary discoveries. Expectedly, more unpredictable outliers should be uncovered in principle with the increment of the coverage and quality of upcoming survey data. Therefore, developing approaches for outlier detection are as important as those for the first three tasks (Reyes & Estevez 2020; Ishida et al. 2021; Webb et al. 2020). ´

Outliers are defined in various papers (Hawkins 1980; Beckman & Cook 1983; Barnett & Lewis 1984; Pearsons et al. 1995), generally, it is described as: an outlier is an observation that deviates significantly from primary observations so that it aroused suspicions that a different mechanism generates it. In daily life, outlier detection has numerous applications, including credit card fraud detection, the discovery of criminal activities in E-commerce, video surveillance, pharmaceutical research, weather prediction and the analysis of performance statistics of professional athletes. Most of them are relevant to troubles. Nevertheless, the detection of astronomical outliers always leads to the discovery of surprising unforeseen facts and expands

<sup>?</sup> First author

<sup>??</sup> Corresponding author, nan.li@nao.cas.cn

<sup>1</sup> https://www.euclid-ec.org/

<sup>2</sup> https://www.lsst.org

<sup>3</sup> http://www.bao.ac.cn/csst/

the boundaries of human knowledge of the universe (Pruzhinskaya et al. 2019; Sharma et al. 2019). Hence, it is necessary to develop efficient and automated approaches for detecting astronomical outliers and understand their feasibility and reliability thoroughly, particularly in the era of big data (Zhang et al. 2004; Margalef-Bentabol et al. 2020).

Early outlier detection methods (Edgeworth 1888; Zhang et al. 2004; Dutta et al. 2007; Solarz et al. 2017; Giles & Walkowicz 2019; Fustes et al. 2013; Baron & Poznanski 2017) are generally based on traditional unsupervised learning algorithms. For instance, Giles & Walkowicz (2019) employed a variant of the DBSCAN clustering algorithm to detect outliers in derived light curve features. Baron & Poznanski (2017) extracted the feature of galaxy spectra manually firstly and then adopted an unsupervised Random Forest to detect the most outlying galaxy spectra within the Sloan Digital Sky Survey 4 (SDSS). Moreover, as a well-known clustering algorithm, k-Nearest Neighbor (Dasarathy 1991) becomes popular for detecting outliers since it operates without assumption about the data distribution. However, these traditional methods become unsuitable when the volume and quality of astronomical images increase greatly. One reason is that the feature extraction routines in traditional methods are too coarse and inflexible to retain details and untypical features of the high-quality astronomical images; another reason is that the efficiency of CPU-based traditional methods is too slow to handle the tremendous volume of future survey data.

Recently, beyond traditional machine learning, deep learning is utilized to construct programs for detecting outliers (Chalapathy & Chawla 2019; Nadeem et al. 2016; Hendrycks et al. 2018; D'Addona et al. 2021), such as Auto-Encoder (AE Vincent et al. 2010) and Convolutional Auto-Encoder (CAE Masci et al. 2011; Storey-Fisher et al. 2020). AE and CAE represent input images with a feature vector which can be used to reconstruct the input images with the most likelihoods. This feature extraction procedure is automated and speedy. Besides, Bayesian Gaussian Mixture is utilized to implement the clustering process and then identify the galaxy images' outlier according to the distribution of the feature vectors in latent space (Cheng et al. 2021). Combining the above two modules, one can classify galaxy images without labels (Cheng et al. 2020), as well as to detect outliers. However, the performance of such unsupervised approaches based on deep learning is above 10%˜20% worse than that of supervised approaches due to noisy data (Zhou & Paffenroth 2017).

In this work, we adopt the attention mechanism (Vaswani et al. 2017) to further improve the performance of the unsupervised methods as it can make the CAE pay more attention to the critical features and suppress background noise. To understand the differences from traditional outlier detection methods to state-of-art attention-improved ones systematically, we construct three programs, which are built upon the KNN, CAE +

<sup>4</sup> https://www.sdss.org/

KNN and CAE + KNN + Attention mechanism, separately. We organize two types of datasets based on the galaxy images data published by the Galaxy Zoo Challenge Project on Kaggle 5 to evaluate the performance of various approaches in different cases. The first datasets of galaxy images for testing the above approaches include inliers containing a single type of galaxy morphology plus outliers containing a single type of galaxy morphology; the second dataset is similar to the first ones but with multiple types of galaxy morphology in the outliers. After conducting extensive experiments, we find that CAE boosts the clustering process significantly and improves the accuracy of detecting outliers; the attention mechanism increases the accuracy further since it guarantees CAE to extract valuable features only, avoiding noise. It is the first time involving the attention mechanism in the outlier detection of astronomical images, which is worth being included in the program for similar purposes in the future. For the convenience of other researchers, we published the code and data used in this project onine 6 .

This paper is structured as follows. We introduce the datasets used in this work in Sect. 2. Sect. 3 describes the methods we constructed. Details about the experiments, including data processing and the implementation of outliers detection with the above approaches, are shown in Sect. 4. We summarize and analyze the results in Sect. 5. Finally, the discussion and conclusions are delivered in Sect. 6.

## 2 DATA

The galaxy morphology data used in this study is collected from the Galaxy Zoo project (Willett et al. 2013; Ventura & D'Antona 2011). In this section, we first introduce the origin and composition of the dataset and then present the filtering methods and how to divide the original data in Sect. 2.1. Sect. 2.2 describes how to construct experimental data subsets to evaluate the performance of outlier detection with the approaches mentioned in Sect. 3.

## 2.1 The Galaxy Zoo Dataset

The SDSS captured around one million galaxy images. To classify these galaxies morphologically, the Galaxy Zoo Project was launched (Lintott et al. 2008), a crowd-sourced astronomy project inviting volunteers to assist in the morphological classification of large numbers of galaxies. The dataset we adopted is one of the legacies of the galaxy zoo project, and it is publicly available online with the Galaxy-zoo Data Challenge Project on Kaggle7 .

<sup>5</sup> https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

<sup>6</sup> https://github.com/hanlaomao/hanlaomao.git

<sup>7</sup> https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data

| category, and the last column is the number of galaxy images in each category. |  |  |  |
| --- | --- | --- | --- |
| Category | Category name | Thresholds | Number |
| 0 | Completely round smooth | fsmooth ≥ 0.469 | 8436 |
| fcompletelyround ≥ 0.50 |  |  |  |
| 1 | In-between smooth | fsmooth ≥ 0.469 | 8069 |

2 Cigar-shaped smooth fsmooth ≥ 0.469 579

3 Edge-on ffeatures/disk ≥ 0.430 3903

4 Spiral fedge−on,no ≥ 0.715 7806

Total 28793

fin−between ≥ 0.50

fcigar−shaped ≥ 0.50

fedge−on,yes ≥ 0.602

fSpiral,yes ≥ 0.619

Table 1: The five galaxy morphology categories from 0˜4 with 28793 samples. The first column is the category id, the second column is the name of category, the third column is the thresholds corresponding to each

The dataset provides 28793 galaxy morphology images with middle filters available in SDSS (g, r, and i) and a truth table including 37 parameters for describing the morphology of each galaxy. The 37 parameters are between 0 and 1 to represent the probability distribution of galaxy morphology in 11 tasks and 37 responses (Willett et al. 2013). Higher response values indicate that more people recognize the corresponding features in the images of given galaxies. The catalog is further debiased to match a more consistent question tree of galaxy morphology classification (Hart et al. 2016).

To make the problem of outlier detection representative, we reorganize 28793 images into five categories: Completely round smooth, In-between smooth, Cigar-shaped smooth,Edge-on and Spiral according to the 37 parameters in the truth table. The filtering method refers to the threshold discrimination criteria in Zhu et al. (2019). For instance, when selecting the Completely round smooth, values are chosen as follows: fsmooth more than 0.469, fcomplete,round more than 0.50, as shown in Table 1. The testing sets are constructed by choosing images from the above categories, and details are presented in Sect. 2.2.

## 2.2 Experimental Data Subsets

To mimic different cases of outlier detection, we construct two sorts of experimental data subsets by selecting images from the categories of galaxy images described in Sect. 2.1. One group of testing sets includes inliers containing a single type of galaxy morphology plus outliers containing a single type of galaxy morphology; Table 2: Number of samples in five different experimental data subsets. The first column is the category id and the last five columns are five date subsets. For example, the second column represents the first experimental date subset, it just contains 16000 samples of category 0 as inliers and 1778 samples of category 2 as outliers. The last column represents the fifth data subset which contains 16000 samples of category 0 as inliers and contains total 1778 samples from catefory 1˜4.

| Category | Subset1 | Subset2 | Subset3 | Subset4 | Subset5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 16000 | 16000 | 16000 | 16000 | 16000 |
| 1 | 0 | 0 | 1778 | 0 | 445 |
| 2 | 1778 | 0 | 0 | 0 | 445 |
| 3 | 0 | 1778 | 0 | 0 | 444 |
| 4 | 0 | 0 | 0 | 1778 | 444 |
| Total | 17778 | 17778 | 17778 | 17778 | 17778 |

the other group of testing sets is similar to the first ones for inliers but containing multiple types of galaxy morphology in the outliers.

Implicitly, the first group contains four data subsets, the inliers are all Completely round smooth galaxies, and the outliers are selected from other categories of galaxies separately. The fraction of outliers is 10% in each subset. The second group contains one data subset, the inliers are also Completely round smooth galaxies, but the outliers consist of galaxy images from other categories of galaxies. The total fraction of outliers is 10% as well, and the four types of galaxy images are equally constituted in the outliers.

Table 2 shows an overview of the above five testing sets, and columns denote the structure of each testing set. For instance, the first testing set (subsect1) consists of Completely round smooth galaxies (category 0) as inliers and Cigar-shaped smooth galaxies (category 2) as outliers. There are 16000 inliers and 1778 outliers. Note that when lacking galaxy images of some categories, we expand the insufficient number of galaxy images by using data augmentation (see Sect. 4.1).

## 3 METHODOLOGY

For comparing the traditional methods and our state of art method, we build three approaches for outlier detection. The simplest one is based on KNN only, a classic clustering algorithm grounded on distance metrics. The second one involves CAE for feature extraction but still utilizes KNN for the clustering procedure. At last, we employ the attention mechanism to improve the stability of the feature extraction with CAE. The following subsections demonstrate details of the construction of these approaches.

## 3.1 The KNN-based Approach

The KNN algorithm is one of the non-parametric classifying algorithms (Dasarathy 1991), whose core idea is to assume that data X has K nearest neighbors in the feature space. If most K neighbors belong to a certain category, the X could also be determined to belong to this category. As shown in Fig. 1 (a), the yellow rectangle is the data X needs to be predicted. Assuming K=3, as shown in Fig. 1 (b), then the KNN algorithm will find the three neighbors closest to X (here enclosed in a circle) and select a category with the most elements. For example, in Fig. 1 (b), there are more elements described by red triangles, so the X is classified to the category containing elements described by red triangles. As shown in Fig. 1 (c) and Fig. 1 (d), when K=5, the X is classified to the category containing elements described by blue circles. Hu (2019) used KNN-based algorithms to perform classification experiments on a variety of datasets and achieved good results without any assumptions about the data. However, the KNN-based algorithm would cost considerable computing time due to the high data dimension in the case of astronomical images as input data.

## 3.2 The CAE-KNN-based Approach

CAE (Masci et al. 2011) is an optimized AE by adopting a convolution operation, which could extract principal features of astronomical image with high dimension. CAE KNN makes full use of the CAE advantage in reducing the dimension to improve above KNN-based algorithm. We first present the architecture and components of CAE as shown in Fig. 2, and then describe the joint of CAE and KNN.

CAE consists of two components: the encoder and the decoder. The first component is the encoder, which is responsible for extracting the representative features from input images. For an input image x, the j th representative feature map h j is expressed as Eq. 1.

$h^{j}=f(x*W^{j}+b^{j})$,

where Wj is the j th filter, ∗ denotes the convolution operation, b j is the corresponding bias of the feature map and *f* is an activation function. The activation function f(z), where the input denotes by z used in the convolutional layers, is the Rectified Linear Unit (ReLu) (Bengio et al. 2007), as described in Eq. 2.

$$f(z)=\begin{cases}0&if\ z\ <\ 0\\ z&if\ z\ \geq\ 0.\end{cases}\tag{2}$$

The encoder in this study is built with four convolutional layers (filter size: 64, 32, 16, and 8) and three dense layers (unit size: 128, 64, 32). A pooling layer follows each convolutional layer with 2 by 2 pixels. The

Fig. 1: The classifying results based on classical KNN. Panel (a) and panel (b) present the procedure of an element with yellow rectangle is classified to the category with red triangle when K=3. Panel (c) and panel (d) describe the procedure of data X with yellow rectangle is classified to the category with blue circle when K=5.

pooling layer is also considered a down-sampling layer, aiming to reduce the volume of parameters involved in the encoder.

The second component of the CAE is the decoder, and its function is to reconstruct the input image according to the extracted feature map obtained by the encoder. The decoder structure is symmetrical with the structure of the encoder. In other words, its structure is just the opposite of the encoder structure. As for the detail of reconstructing procedure, please refer to (Masci et al. 2011; Cheng et al. 2020). The decoder has three dense layers (unit size: 32, 64, and 128), four convolutional layers (filter size: 8, 16, 32 and 64) using

Fig. 2: The architecture and components of CAE. CAE consists of two components, the encoder and the decoder. Green cuboids in encoder and orange cuboids in decoder denote convolution layer; blue cuboids indicate maxpooling layer; purple cuboids present uppooling layer.

the ReLu activation function (Bengio et al. 2007), and an extra convolutional layer (filter size: 3) using the softmax (Ren et al. 2017) function as the output of the decoder. Except for the last output layer, there is an upsampling layer behind each convolutional layer, whose function is to gradually restore the feature maps to the same shape as input images. The layer between the encoder and the decoder is the embedding layer used to reconstruct the input galaxy images.

The loss function L between the two components is given by Eq. 3 (Cheng et al. 2020).

$$L=-\frac{1}{N}\left[t^{n}logy^{n}+(1-t^{n})\log\left(1-y^{n}\right)\right]\,\tag{3}$$

where N is the number of samples, t n is the target data, and y n is the reconstructed data. The goal of CAE is to minimize the reconstruction error by using loss function L.

As so far, we could get the low dimension features from galaxy images by using the embedding layer vectors in CAE. And then, these features are fed into the KNN algorithm avoiding the time-consuming problem of KNN outlier detection. However, the CAE KNN has the disadvantage of instability since the background noise of the galaxy image sometimes influences the stability of the outlier detection.

## 3.3 The Attention-CAE-KNN-based Approach

To increase the stability of the CAE KNN, we propose a novel algorithm, namely attCAE KNN, which is the first time to explore the attention strategy to CAE. Attention strategy (Xu et al. 2015; Gregor et al. 2015)

Fig. 3: The architecture of the attCAE KNN, including encoder, decoder and KNN module, where the CBAM attention strategy is added to the encoder.

makes the attCAE KNN focus on 'what' is meaningful for given astronomical images so that attCAE KNN could ignore the background noise. We build attCAE KNN by adopting a convolutional block attention module (CBAM Liu et al. 2019). Its architecture is shown in Fig. 3, including encoder, decoder, and KNN module. The decoder and KNN module have been described in Sect. 3.1 and Sect. 3.2. Next, we focus on the improved encoder.

The first part is the encoder that consists of the channel attention block and the spatial attention block (Liu et al. 2019), which differs from the classical encoder in inserting CBAM, as shown in Fig. 4. These two blocks can extract the meaningful features of astronomical images along the two dimensions of the channel axis and the spatial axis. The second part is the decoder, in which the CBAM is not inserted after the convolutional layer. This is because through the analysis of experimental results, adding the CBAM after the convolutional layer of the decoder hardly improves the experimental results. To reduce model complexity and decrease model training time, we only add the CBAM after the convolutional layer in the encoder of the CAE. The last part is the KNN module, whose input data is the latent features from the embedding layer of attCAE.

# 4 EXPERIMENT

We present the details of experiments with the data and methods described in Sect. 2 and Sect. 3 here. It includes data processing, parameters of the machine learning models, evaluation metrics and experimental environments.

Fig. 4: The encoder of the attCAE KNN, which consists of the channel attention block, the spatial attention block and other CAE blocks.

Fig. 5: The procedure of data preprocessing on the original galaxy image.

## 4.1 Data Pre-processing

As is shown in Sect. 2, we obtain 28793 RGB color images with a size of 424 × 424 × 3 pixels. Considering the valuable features of these images are concentrated at the central part, we conduct some pre-processing operations (see Fig. 5). The first step is to crop the images with a box of 170×170 pixels in all channels. The second step is to downscale images from 170 × 170 × 3 pixels to 80 × 80 × 3 pixels. The last step is to crop images from 80 × 80 × 3 pixels to 64 × 64 × 3 pixels further. The detailed operations refer to the process in (Zhu et al. 2019). Five processed examples from five categories described in Sect. 2 are displayed in Fig. 6.

The number of images in some categories is too small to be outliers for supporting machine learning algorithms for outlier detection, for instance, there are only 579 Cigar-shaped smooth galaxies. Thus, we make data augmentation by rotating these images randomly and finally obtain 17778 images in five data subsets, where each data subset consists of 16000 inliers and 1778 outliers (see Table 2).

Completely round smooth

In-between smooth

Cigar-shaped smooth

Edge-on Spiral

Fig. 6: Five representative examples from five categories.

## 4.2 Training And Clustering

We apply the three methods (KNN, CAE KNN, and attCAE KNN) to the data subsets separately. The training process consists of auto-encoder training and KNN training. The former is for extracting the representative features of the astronomical images, while the latter is for detecting outliers. The flow chart of the attCAE KNN for detecting outliers in astronomical images is shown in Fig. 7.

The training process in this paper is entirely different from the training process in the context of supervised learning. We train CAE to extract features by comparing the input images and generated images, so no labels are included in the whole process. To avoid overfitting, we divide each dataset shown in Table. 2 into training sets and testing sets with a ratio of 7:3, and the images in the training set and test set are randomly selected from the whole set with 17778 images. Considering that the number of outliers always accounts for a small part of the total dataset, we set the proportion of outlier data to account for 10% of the whole dataset for detecting outliers. For example, the number of outliers in the test set is 533, which can be calculated by 17778×0.3×0.1.

During the training procedure of CAE, parameters of the embedded layer need to be optimized in a datadriven manner. We use area under the receiver operating characteristic curve (AUC Bradley 1997; Fawcett 2006)) as the criteria. The receiver operating characteristic (ROC Fawcett 2006; Cheng et al. 2020) can be drawn with false-positive rates (F P R) and true-positive rates (T P R), which are given by Eq. (4) and Eq. (5),

$$FPR=\frac{FP}{FP+TN},\tag{4}$$

$$TPR=\frac{TP}{TP+FN},\tag{5}$$

where T P means true positive, T N means true negative, F P denotes false positive and F N means false negative, respectively. We then repeat the outlier detection process and compare the AUC of each classification to find the most optimal number of extracted features within the embedding layer in the CAE. In Fig. 8,

Fig. 7: The flow chart of the attCAE KNN for detecting outliers in astronomical images.

the blue dashed line shows the mean AUC of the outlier detection with CAE KNN, while the solid red line shows the mean AUC of the outlier detection with attCAE KNN. The lighter shadings present the standard deviation of the three results from the three training processes. One can see that the AUC of CAE KNN and attCAE KNN reach the maximum values when the feature number of the embedding layer is set to 20, which is, therefore, chosen to be the number of latent features in CAE and attCAE. In addition, it can also be seen that the stability of attCAE KNN is higher than that of CAE KNN.

The detailed implementation of KNN outlier detection refers to the modules in (Zhao et al. 2019; Ramaswamy et al. 2000), where there is a core procedure, namely *computeOutliersIndex*. The output data of procedure *computeOutliersIndex* is stored in a heap structure (Lattner & Adve 2003). We take the top 533 galaxy images with the largest values in a heap as outliers and then evaluate the model's performance based on the 533 outliers.

## 4.3 Evaluation Metrics

Besides AUC, we also employ Recall, F1 score, and Accuracy to estimate the performance of outlier detection (Cheng et al. 2020; Zhu et al. 2019; Hou 2019; Kamalov & Leung 2020) , which are given by Eq. (6), (7),(8) and (9).

$$precision=\frac{TP}{TP+FP},\tag{6}$$

$$recall=\frac{TP}{TP+FN},\tag{7}$$

$$f1=2\times\frac{precision\times recall}{precision+recall},\tag{8}$$

accuracy_ = $\frac{TP+TN}{TP+FP+TN+FN}$. (9)

Be worth mentioning, though Accuracy and F1 score are two of major performance metrics in many applications, they are considered supplements to AUC and Recall since the data distributions in this study are unbalanced (the ratio of the outliers is only 10%). In addition , T P+F N is equal to the T P+F P in all experiments, resulting in the values of recall being equal to the values of F1.

## 4.4 Implementation Details

The experimental environment of this study is as follows. We mainly use an Intel Xeon E5-2690 CPU and an Nvidia Tesla K40 GPU. Software environment includes python 3.5, Keras 2.3.1, NumPy 1.16.2, Matplotlib 3.0.3, scikit learn 0.19.1, and pyod 0.8.4. It takes less than half an hour to train 17778 images running on two NVIDIA Tesla K40 GPUs.

When training the CAE and attCAE, we set the batch size to 128 and set epoch to 100, use the binary crossentropy as described in Eq. (3) and the Adam as optimizer. One can refer to the settings of the CBAM in (Woo et al. 2018).

## 5 RESULTS

This section presents the results of experiments described in Sect. 4. The outcomes of each experiment and the comparative analysis are listed in the following two subsections.

# 5.1 The Case of *Single Type Inliers And Single Type Outliers*

Experiment 1

Fig. 8: Effect on AUC mean values of various feature numbers in embedding layer.

|  | AUC | recall | f1 | acc | Time |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.83 | 0.26 | 0.26 | 0.85 | >4hour |
| CAE KNN | 0.94 | 0.57 | 0.57 | 0.91 | 10min |
| attCAE KNN | 0.97 | 0.76 | 0.76 | 0.95 | 10min |

Table 3: The results of Experiment 1, the bold entries highlight our results.

We apply three methods illustrated in Sect. 3 to the testing set comprising images of Completely round smooth galaxies as inliers and images of Cigar-shaped smooth galaxies as outliers. Five metrics, i.e., Area under the ROC (AUC), Recall, F1 score, accuracy, and runtime, are utilized to evaluate outlier detection performance of the three methods. The results are shown in Table 3. Apparently, the attCAE KNN approach obtains the best performance in all metrics. For instance, the recall using CAE KNN is ∼ 31% higher than KNN, reaching 57%, while the recall using attCAE KNN is ∼ 19% higher than CAE KNN, which can reach 76%. Notably, the runtime of attCAE KNN is also superior to other methods, and it is ∼ 4% of that of KNN alone.

|  | AUC | recall | f1 | acc | Time |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.83 | 0.25 | 0.25 | 0.85 | >4hour |
| CAE KNN | 0.95 | 0.56 | 0.56 | 0.91 | 10min |
| attCAE KNN | 0.98 | 0.78 | 0.78 | 0.96 | 10min |

Table 4: The results of Experiment 2, the bold entries highlight our results.

Table 5: The results of Experiment 3, the bold entries highlight our results.

|  | AUC | recall | f1 | acc | Time |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.64 | 0.11 | 0.11 | 0.82 | >4hour |
| CAE KNN | 0.68 | 0.15 | 0.15 | 0.83 | 10min |
| attCAE KNN | 0.71 | 0.22 | 0.22 | 0.84 | 10min |

## Experiment 2

The second testing set contains images of Completely round smooth galaxies as inliers and images of Edge-on galaxies as outliers, as for the experimental parameters are similar to experiment 1. As is shown in Table 4, the results are similar to experiment 1 too. One of the reasons is that the differences between inliers and outliers are both well distinguished in the first and second experiments.

## Experiment 3

This experiment is similar to the previous one, except we adopt images of In-between smooth galaxies as outliers. However, as is shown in Table 5, the experimental results are different from previous ones because the similarity between inliers and outliers in this test set is less significant than the previous ones. Though including CAE and attention mechanism brings improvement, it is less considerable than the first two cases. For example, concerning recall, the CAE KNN is ∼ 4% higher than KNN and only reaches 15%, while the recall of using attCAE KNN is higher than CAE KNN by ∼ 7% and reaches 22% only. It reveals that the definition of the outliers detection problem is crucial for outlier detection. According to the results, identifying smooth elliptical galaxies with specific ellipticity is not a practical outlier detection problem.

## Experiment 4

Similarly, we adopt images of Spiral galaxies as outliers in this experiment. As expected (see Table 6), this experiment's results are better than experiment 3 but worse than experiments 1 and 2 because the distinguishability between inliers and outliers in this testing set is more noticeable than that in case 3 but less than cases 1 and 2 (mainly due to the PSF smearing). The most noteworthy difference between completely

|  | AUC | recall | f1 | acc | Time |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.68 | 0.15 | 0.15 | 0.83 | >4hour |
| CAE KNN | 0.77 | 0.24 | 0.24 | 0.84 | 10min |
| attCAE KNN | 0.81 | 0.29 | 0.29 | 0.86 | 10min |

Table 6: The results of Experiment 4, the bold entries highlight our results.

round-smooth and face-on Spiral galaxies is detailed structures and colors; thus, we hope the improvement of CAE and attention mechanism would be more significant when applying the methods to data from space-born telescopes.

## 5.2 The Case of *Single Type Inliers And Multiple Type Outliers*

The above experiments primarily explore the feasibility of unsupervised approaches for outlier detection with testing sets containing single type inliers and single type outliers. This sub-section demonstrates an experiment in a more realistic case, i.e., the testing set contains a single type of inliers plus multiple types of outliers.

## Experiment 5

We consider images of Completely round smooth galaxies as inliers and images of In-between smooth, Cigar-shaped smooth, Edge-on and Spiral as outliers. The experimental results are shown in Table 7, the attCAE KNN still achieves the best performance. The recall of CAE kNN reaches 43%, ∼ 21% higher than KNN, and the recall of attCAE KNN is ∼ 10% higher than CAE KNN, reaching to 53%. It is easy to conclude that the missing points in this experiment are dominated by In-between smooth galaxies.

Notably, recall and f1 values are the same in all the experiments since we define the most distant 10% objects to the center of the cluster of inliers in feature space as outliers during the detection of outliers, while the fraction of outliers in the testing set is 10%. Consequently, FN equals FP, then recall will equal precision, and hence recall equals f1 as well. However, when the chosen fraction does not equal the actual value, recall and f1 are not the same. The actual fraction of outliers is unknown in real cases; thus, it is impossible to choose a perfect fraction, and one needs to choose a rational fraction to define outliers according to specific scientific goals. We set the fraction to be 5% and 15%, in addition to illustrating comparative results. As is shown in Table 7, when the definition of outliers is the most distant 5% objects to the center of the inlier cluster, recall decreases to 0.37, precision increases to 0.74, and f1 is 0.5. Whereas, when the definition of outliers is the most distant 15% objects to the center of the inlier cluster, recall, precision, and f1 become 0.67, 0.44, and 0.53 separately. Accordingly, if one plans to obtain a sample of outliers with high

|  | AUC | recall | precision | f1 | acc | Time |
| --- | --- | --- | --- | --- | --- | --- |
| KNN | 0.77 | 0.22 | 0.22 | 0.22 | 0.84 | >4hour |
| CAE KNN | 0.85 | 0.43 | 0.43 | 0.43 | 0.87 | 10min |
| attCAE KNN 5% | 0.87 | 0.37 | 0.74 | 0.50 | 0.92 | 10min |
| attCAE KNN 10% | 0.87 | 0.53 | 0.53 | 0.53 | 0.92 | 10min |
| attCAE KNN 15% | 0.87 | 0.67 | 0.44 | 0.53 | 0.88 | 10min |

Table 7: The results of Experiment 5, the bold entries highlight our results.

completeness, a greater fraction (e.g., 15%) is needed, while if the goal is to find rare objects with noticeable and wired features efficiently, a lower fraction (e.g., 5%) is practical.

# 6 DISCUSSION AND CONCLUSIONS

In this study, we explore the feasibility of applying unsupervised learning to detect outliers in the data of galaxy images. Firstly, we construct three methods, which are built upon the KNN, CAE + KNN, and attCAE KNN separately. To evaluate the performance of the approaches, we organize two sorts of datasets based on the data of galaxy images given by the project of galaxy zoo challenge published on Kaggle. One group of testing sets includes inliers containing a single type of galaxy morphology plus outliers containing a single type of galaxy morphology; the other group of testing sets is similar to the first ones for inliers but with multiple types of galaxy morphology in the outliers. Comparing the results of applying three approaches to all the testing sets, we find that attCAE KNN achieves the best performance and costs the least runtime, though its superiority is limited in the case of the testing set with a substantial similarity between inliers and outliers.

Specifically, KNN is usable for outlier detection, but its performance and efficiency are deficient. For instance, the best recall is 0.25, even when the testing set (testing set 1) has significant differences between inliers and outliers. The main reason for the shortcomings is the outdated procedure for extracting features. Therefore, we involve CAE as a module for feature extractions, and then the recall reaches 0.56 in the case of testing set 1. We further employ the attention mechanism to improve the stability of the feature extraction module, and the best recall goes to 0.78 in the case of testing set 1. Repeating the above process in other testing sets that contain single type inliers and single outliers, attCAE KNN performs the best and costs the least runtime, and one can see more details in Table 3, Table 4, Table 5, Table 6.

To test the feasibility of the three methods in a more realistic context, we create testing set 5, containing single type inliers and multiple types outliers. As is expected, attCAE KNN is still superior to the other two methods. For instance, its recall is 0.53, but the recalls of CAE KNN and KNN are 0.43 and 0.22, respectively. As is shown in Table 7, the advantage of attCAE KNN is evident over all five metrics. Hence, we can conclude that outlier detection in galaxy images is feasible by combing CAE and KNN, and the performance can be enhanced by involving the attention mechanism further. Besides, we implement a comparative investigation with different definitions of outliers when detecting them with our methods. The results in Table 7 demonstrate that a tighter definition of outliers leads to higher precision but lower recall, while a looser definition of outliers leads to lower precision but higher recall; nevertheless, the overall AUC is stable.

The structures of the testing sets used in the paper are relatively simple compared to real observations since we focus on assessing the feasibility of unsupervised approaches. To make our unsupervised approach suitable for real observations, we are forming a module to reduce any complex case (multiple types inliers + multiple types outliers) to the simple one employed in this paper (single type inliers + multiple types outliers) by combining human inspection and supervised learning. Then, we will apply the pipeline to actual survey data, such as KiDs, DES, and DESI legacy imaging surveys, to test its applicability and reliability. Also, to further improve the performance of approaches, particularly attCAE kNN, we plan to optimize the architectures and hyper-parameters while applying them to observational data. Last but not least, defining the boundary of inliers and outliers is key to the outlier detection task, as is shown in the results in testing set 3. Hence, we will adopt a data-driven strategy to investigate the optimal definition of the boundaries according to specific scientific purposes.

In summary, unsupervised approaches, especially when we involve CAE and the Attention mechanism, are feasible for outlier detection in the datasets of galaxy images. It is foreseen that unsupervised approaches can mine astronomical outliers so as to expand the boundary of human knowledge of the Universe in the big data era. On the other hand, the unsupervised approaches can also detect misclassified samples in standard supervised classification, similar to outlier detection, with no additional efforts. Accordingly, an ideal pipeline for classifying astronomical objects might need to combine supervised and unsupervised manners.

Acknowledgements The dataset used in this work is collected from the Galaxy-Zoo-Challenge-Project posted on the Kaggle platform. We acknowledge the science research grants from the China Manned Space Project with NO.CMS-CSST-2021-A01 and NO.CMS-CSST-2021-B05. YH, ZQZ, and YLC are thankful for the funding and technical support from the Jiangsu Key Laboratory of Big Data Security and Intelligent Processing.

## References

- Barnett, V., & Lewis, T. 1984, Wiley Series in Probability and Mathematical Statistics. Applied Probability and Statistics 2
- Baron, D., & Poznanski, D. 2017, Monthly Notices of the Royal Astronomical Society, 465, 4530 3
- Beckman, R. J., & Cook, R. D. 1983, Technometrics, 25, 119 2
- Bengio, Y., LeCun, Y., et al. 2007, Large-scale kernel machines, 34, 1 7, 9
- Bradley, A. P. 1997, Pattern recognition, 30, 1145 12
- Chalapathy, R., & Chawla, S. 2019, arXiv preprint arXiv:1901.03407 3
- Chen, S.-X., Sun, W.-M., & He, Y. 2022, Research in Astronomy and Astrophysics, 22, 025017 2
- Cheng, T.-Y., Li, N., Conselice, C. J., et al. 2020, Monthly Notices of the Royal Astronomical Society, 494, 3750 2, 3, 8, 9, 12, 14
- Cheng, T. Y., Marc, H. C., Conselice, C. J., et al. 2021, Monthly Notices of the Royal Astronomical Society 3
- D'Addona, M., Riccio, G., Cavuoti, S., Tortora, C., & Brescia, M. 2021, in Intelligent Astrophysics, ed. I. Zelinka, M. Brescia, & D. Baron, Vol. 39, 225 3
- Dasarathy, B. V. 1991, IEEE Computer Society Tutorial 3, 7
- Dutta, H., Giannella, C., Borne, K., & Kargupta, H. 2007, in Proceedings of the 2007 SIAM International Conference on Data Mining, SIAM, 473 3
- Edgeworth, F. Y. 1888, Journal of the Royal Statistical Society, 51, 346 3
- Fawcett, T. 2006, Pattern Recognit. Lett, 27, 861 12
- Fustes, D., Manteiga, M., Dafonte, C., et al. 2013, Astronomy & Astrophysics, 559, A7 3
- Giles, D., & Walkowicz, L. 2019, Monthly Notices of the Royal Astronomical Society, 484, 834 3
- Gregor, K., Danihelka, I., Graves, A., Rezende, D., & Wierstra, D. 2015, in International Conference on Machine Learning, PMLR, 1462 9
- Gupta, R., Srijith, P. K., & Desai, S. 2022, Astronomy and Computing, 38, 100543 2
- Hart, R. E., Bamford, S. P., Willett, K. W., et al. 2016, Monthly Notices of the Royal Astronomical Society, 461, 3663 5
- Hawkins, D. M. 1980, Identification of outliers, Vol. 11 (Springer) 2
- Hendrycks, D., Mazeika, M., & Dietterich, T. 2018, arXiv preprint arXiv:1812.04606 3
- Hou, Q. 2019, Journal of Frontiers of Computer Science and Technology, 13, 586 14
- Hu, D. 2019, in Proceedings of SAI Intelligent Systems Conference, Springer, 432 7
- Ishida, E. E. O., Kornilov, M. V., Malanchev, K. L., et al. 2021, A&A, 650, A195 2
- Kamalov, F., & Leung, H. H. 2020, Journal of Information & Knowledge Management, 19, 2040013 14
- Lattner, C., & Adve, V. 2003, Data structure analysis: A fast and scalable context-sensitive heap analysis, Tech. rep., Citeseer 13
- Lintott, C. J., Schawinski, K., Slosar, A., et al. 2008, Monthly Notices of the Royal Astronomical Society, 389, 1179 4
- Liu, T., Kong, J., Jiang, M., & Huo, H. 2019, Journal of Electronic Imaging, 28, 023012 10
- Lochner, M., & Bassett, B. A. 2021, Astronomy and Computing, 36, 100481 2
- Lukic, V., de Gasperin, F., & Bruggen, M. 2019, Galaxies, 8, 3 2 ¨
- Margalef-Bentabol, B., Huertas-Company, M., Charnock, T., et al. 2020, MNRAS, 496, 2346 3
- Masci, J., Meier, U., Cires¸an, D., & Schmidhuber, J. 2011, in International conference on artificial neural networks, Springer, 52 3, 7, 8
- Nadeem, M., Marshall, O., Singh, S., Fang, X., & Yuan, X. 2016 3
- Pearsons, K. S., Barber, D. S., Tabachnick, B. G., & Fidell, S. 1995, The Journal of the Acoustical Society of America, 97, 331 2
- Pruzhinskaya, M. V., Malanchev, K. L., Kornilov, M. V., et al. 2019, Monthly Notices of the Royal Astronomical Society, 489, 3591 3
- Ramaswamy, S., Rastogi, R., & Shim, K. 2000, in Proceedings of the 2000 ACM SIGMOD international conference on Management of data, 427 13
- Ren, Y., Zhao, P., Sheng, Y., Yao, D., & Xu, Z. 2017, in Proceedings of the 26th International Joint Conference on Artificial Intelligence, 2641 9
- Reyes, E., & Estevez, P. A. 2020, arXiv e-prints, arXiv:2005.07779 2 ´
- Sharma, K., Kembhavi, A., Kembhavi, A., Sivarani, T., & Abraham, S. 2019, Bulletin de la Societe Royale des Sciences de Liege, 88, 174 3
- Solarz, A., Bilicki, M., Gromadzki, M., et al. 2017, Astronomy & Astrophysics, 606, A39 3
- Storey-Fisher, K., Huertas-Company, M., Ramachandra, N., et al. 2020, arXiv e-prints, arXiv:2012.08082 3
- Vaswani, A., Shazeer, N., Parmar, N., et al. 2017, in Advances in neural information processing systems, 5998 3
- Ventura, P., & D'Antona, F. 2011, Monthly Notices of the Royal Astronomical Society, 410, 2760 4
- Vincent, P., Larochelle, H., Lajoie, I., et al. 2010, Journal of machine learning research, 11 3
- Webb, S., Lochner, M., Muthukrishna, D., et al. 2020, MNRAS, 498, 3077 2
- Willett, K. W., Lintott, C. J., Bamford, S. P., et al. 2013, Monthly Notices of the Royal Astronomical Society, 435, 2835 4, 5
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. 2018, in Proceedings of the European conference on computer vision (ECCV), 3 14
- Xu, K., Ba, J., Kiros, R., et al. 2015, in International conference on machine learning, PMLR, 2048 9
- Zhang, Y.-X., Luo, A. L., & Zhao, Y.-H. 2004, in Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, Vol. 5493, Optimizing Scientific Return for Astronomy through Information Technologies, ed. P. J. Quinn & A. Bridger, 521 3
- Zhang, Y.-X., Luo, A.-L., & Zhao, Y.-H. 2004, in Optimizing scientific return for astronomy through information technologies, Vol. 5493, International Society for Optics and Photonics, 521 3
- Zhang, Z., Zou, Z., Li, N., & Chen, Y. 2022, Classifying Galaxy Morphologies with Few-Shot Learning, arXiv:2202.08172 2
- Zhao, Y., Nasrullah, Z., & Li, Z. 2019, arXiv preprint arXiv:1901.01588 13
- Zhou, C., & Paffenroth, R. C. 2017, in Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, 665 3
- Zhu, X.-P., Dai, J.-M., Bian, C.-J., et al. 2019, Astrophysics and Space Science, 364, 1 2, 5, 11, 14

