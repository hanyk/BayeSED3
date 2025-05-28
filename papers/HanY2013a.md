## **Bayesian analysis of galaxy spectral energy distributions with BayeSED**

**Yunkun Han**1,2 **and Zhanwen Han**1,2

1National Astronomical Observatories / Yunnan Observatory, Chinese Academy of Sciences, 100012, Beijing, China email: hanyk@ynao.ac.cn

2Key Laboratory for the Structure and Evolution of Celestial Objects, Chinese Academy of Sciences, 650011, Kunming, China email: zhanwenhan@ynao.ac.cn

**Abstract.** In Han & Han (2012), we have preliminarily built BayeSED and applied it to a sample of hyperluminous infrared galaxies. The physically reasonable results obtained from Bayesian model comparison and parameter estimation show that BayeSED could be a useful tool for understanding the nature of complex systems, such as dust obscured starburst-AGN composite galaxies, from decoding their complex SEDs. In this contribution, we present a more rigorous test of BayeSED by making a mock catalog from model SEDs with the value of all parameters to be known in advance.

**Keywords.** galaxies: active, galaxies: ISM, methods: data analysis, methods: statistical

## **1. Introduction**

As a first simple example, we have chosen a black body as the model SED, since its shape only depends on the temperature. For such a simple model, the ANN and PCA methods that are used in BayeSED are completely unnecessary. However, we can use this simple model to test the reliability of these methods.

Firstly, we have built a SED library of black body spectra with temperatures (T [K]) in the range of [100:300] and the logarithm of luminosity (L [erg/s]) in the range of [40:48]. Then, as a common procedure of the application of BayeSED, we use the PCA method to reduce this library and train a 1:20:16 ANN with it. On the other hand, we have built a mock catalog by using SEDs randomly selected from the original library. These model SEDs are convolved with the transmission function of some selected filters to generate the fluxes in different bands. Then, noise with a Gaussian distribution is added to these fluxes to simulate the real observations. Finally, we use BayeSED to analyze the SEDs of this mock catalog to estimate the parameters values. We found that when the noise was selected from a Gaussian distribution with a standard deviation of 10% of the fluxes, the temperatures and logarithm of luminosities can be recovered with a R.M.S error of only 0.85 and 0.029, respectively.

Based on the current state-of-the-art Bayesian inference tool, we have built BayeSED for performing Bayesian analysis of galaxy multi-wavelength SEDs. The preliminary application of this code gives physically reasonable results. The more rigorous test by using a mock catalog presented here shows that BayeSED could be a reliable tool for understanding the nature of objects by decoding their complex SEDs. BayeSED will soon be publicly available at http://www.ynao.ac.cn/jgsz/kybm/dybhxyhyjtz/jj.

## **Reference**

Han, Y. & Han, Z. 2012, ApJ, 749, 123

