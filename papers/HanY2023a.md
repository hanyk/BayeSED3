# BayeSED-GALAXIES. I. Performance Test for Simultaneous Photometric Redshift and Stellar Population Parameter Estimation of Galaxies in the CSST Wide-field Multiband Imaging Survey

Yunkun Han1,2,3,4 , Lulu Fan5,6 , Xian Zhong Zheng7 , Jin-Ming Bai1,2,3,4 , and Zhanwen Han1,2,3,4

1 Yunnan Observatories, Chinese Academy of Sciences, 396 Yangfangwang, Guandu District, Kunming, 650216, People's Republic of China; hanyk@ynao.ac.cn 2 Center for Astronomical Mega-Science, Chinese Academy of Sciences, 20A Datun Road, Chaoyang District, Beijing, 100012, People's Republic of China 3 Key Laboratory for the Structure and Evolution of Celestial Objects, Chinese Academy of Sciences, 396 Yangfangwang, Guandu District, Kunming, 650216,

People's Republic of China 4 International Centre of Supernovae, Yunnan Key Laboratory, Kunming 650216, People's Republic of China 5 Deep Space Exploration Laboratory/Department of Astronomy, University of Science and Technology of China, Hefei 230026, People's Republic of China 6 School of Astronomy and Space Science, University of Science and Technology of China, Hefei 230026, People's Republic of China 7 Purple Mountain Observatory, Chinese Academy of Sciences, 10 Yuanhua Road, Nanjing 210023, People's Republic of China

Received 2023 February 16; revised 2023 September 18; accepted 2023 September 20; published 2023 November 15

# Abstract

The forthcoming Chinese Space Station Telescope (CSST) wide-field multiband imaging survey will produce seven-band photometric spectral energy distributions (SEDs) for billions of galaxies. The effective extraction of astronomical information from these massive data sets of SEDs relies on the techniques of SED synthesis (or modeling) and SED analysis (or fitting). We evaluate the performance of the latest version of the BayeSED code combined with SED models with increasing complexity for simultaneously determining the photometric redshifts and stellar population parameters of galaxies in this survey. By using an empirical statistics–based mock galaxy sample without SED modeling errors, we show that the random observational errors in photometries are more important sources of errors than the parameter degeneracies and Bayesian analysis method and tool. By using a Horizon-AGN hydrodynamical simulation–based mock galaxy sample with SED modeling errors about the star formation histories (SFHs) and dust attenuation laws (DALs), the simple typical assumptions lead to significantly worse parameter estimation with CSST photometries only. SED models with more flexible (or complicated) forms of SFH/DAL do not necessarily lead to better estimation of redshift and stellar population parameters. We discuss the selection of the best SED model by means of Bayesian model comparison in different surveys. Our results reveal that Bayesian model comparison with Bayesian evidence may favor SED models with different complexities when using photometries from different surveys. Meanwhile, the SED model with the largest Bayesian evidence tends to give the best performance of parameter estimation, which is clearer for photometries with higher discriminative power.

Unified Astronomy Thesaurus concepts: Spectral energy distribution (2129); Bayesian statistics (1900); Extragalactic astronomy (506); Redshift surveys (1378); Stellar populations (1622); Galaxy stellar content (621); Galaxy properties (615); Galaxy photometry (611); Galaxy formation (595); Galaxy evolution (594)

# 1. Introduction

Understanding the complex ecosystem of stars, interstellar gas and dust, and supermassive black holes in galaxies is one of the most important challenges in modern astrophysics (National Academies of Sciences, Engineering, and Medicine 2021). The new generation of space and ground telescopes and the corresponding large surveys will provide vast amounts of multiband data for understanding the cosmic ecosystems and all the complex physical processes involved. For example, the James Webb Space Telescope (Rieke et al. 2005; Gardner et al. 2006; Beichman et al. 2012) is able to detect the earliest stages of galaxies from infrared at unprecedented depths, and is expected to provide decisive observations of the first generation of stars and galaxies (Beichman et al. 2012; Robertson 2022). Meanwhile, forthcoming deep and wide-field surveys with the Chinese Space Station Telescope (CSST; Zhan 2011, 2018, 2021), the Euclid Space Telescope (Laureijs et al. 2011;

Original content from this work may be used under the terms of the Creative Commons Attribution 4.0 licence. Any further distribution of this work must maintain attribution to the author(s) and the title of the work, journal citation and DOI.

Joachimi 2016), the Vera C. Rubin Observatory Legacy Survey of Space and Time (Ivezić et al. 2019; Breivik et al. 2022), and the Nancy Grace Roman Space Telescope (Dore et al. 2019) will provide multiband photometric and spectroscopic information for billions of galaxies. In particular, the CSST wide-field multiband imaging survey is set to image approximately 17,500 deg2 of the sky using the near-ultraviolet (NUV), u, g, r, i, z, and y bands in about 10 yr of orbital time, which aims to achieve a 5σ limiting magnitude of 26 (AB mag) or higher for point sources in the g and r bands. How to effectively and reliably measure the redshift and the properties of various physical components of galaxies from the obtained huge amount of photometric spectral energy distribution (SED) data has become an urgent task to be done. A new generation of SED synthesis and analysis methods and tools are strongly demanded to effectively extract physical information from those massive data sets of observational SEDs.

The SED synthesis and analysis of galaxies are two aspects that are both opposite and unified in nature. The reliability and efficiency of the SED synthesis and analysis methods and tools will directly determine the reliability and efficiency of physical information extraction from the massive multiwavelength data sets. In terms of SED synthesis of galaxies, the evolutionary synthesis technique of stellar populations has become the core method from the pioneering works of Tinsley & Gunn (1976) and Tinsley (1978). Nowadays, the stellar population synthesis models of BC03 (Bruzual & Charlot 2003), M05 (Maraston 2005), FSPS (Conroy et al. 2009), and BPASS (Eldridge & Stanway 2009), among others, are widely used in the study of the formation and evolution of galaxies. However, in SED synthesis models of galaxies, many important uncertainties remain in almost all the model ingredients (Conroy et al. 2009, 2010; Conroy & Gunn 2010; Conroy 2013), such as in the initial (stellar) mass function (IMF) (Padoan et al. 1997; Hoversten & Glazebrook 2008; van Dokkum 2008; Bastian et al. 2010; Cappellari et al. 2012; Ferreras et al. 2013; Gennaro et al. 2018), the physics of stellar evolution (Thomas & Maraston 2003; Zhang et al. 2005; Maraston et al. 2006; Han et al. 2007; Bertelli et al. 2008; Marigo et al. 2008; Brott et al. 2011; Hernández-Pérez & Bruzual 2013), stellar spectral libraries (Coelho 2009; Choi et al. 2019; Knowles et al. 2019, 2021; Yan et al. 2019; Coelho et al. 2020), the complex star formation and metallicity enrichment histories (SFHs and MEHs) (Côté et al. 2016; Debsarma et al. 2016; Carnall et al. 2019; Iyer et al. 2019, 2020; Leja et al. 2019; Maiolino & Mannucci 2019; Valentini et al. 2019; Aufort et al. 2020; Wang & Lilly 2020), the reprocessing by interstellar gas and dust (Draine 2003, 2010; Galliano et al. 2018; Kewley et al. 2019; Salim & Narayanan 2020; Tacconi et al. 2020), and the possible contribution from active galactic nuclei (AGNs) at the center of galaxies (Antonucci 1993, 2012; Netzer 2015; Hickox & Alexander 2018; Brown et al. 2019a, 2019b; Lyu & Rieke 2022). Different choices of these model ingredients will lead to very different estimations of the redshifts and physical parameters of galaxies, as well as to different and even conflicting conclusions about the formation and evolution of galaxies. Therefore, the proper selection of these model ingredients is an essential step in any SED analysis work of galaxies (Han & Han 2019; Han et al. 2020).

In terms of SED analysis of galaxies, the Bayesian method has been widely adopted in the last decade. For example, the most widely used and actively developed SED fitting codes, including MAGPHYS (da Cunha et al. 2008), CIGALE (Noll et al. 2009; Boquien et al. 2019), GalMC (Acquaviva et al. 2011), BayeSED (Han & Han 2012, 2014), BEAGLE (Chevallard & Charlot 2016), Prospector (Leja et al. 2017), BAGPIPES (Carnall et al. 2018), and ProSpect (Robotham & Bellstedt 2020), are all based on Bayesian methods. Besides those, a long list of new SED fitting codes, including MCSED (Bowman et al. 2020), piXedfit (Abdurro'uf et al. 2021), gsf (Morishita 2022), and Lightning (Doore et al. 2023), have been built along this way. The application of Bayesian methods implies that the SED analysis of galaxies is considered as a more general Bayesian inference problem instead of the previous chi-square minimization–based optimization problem known as SED fitting. For the parameter estimation of a given SED model, the Bayesian approach provides the complete posterior probability distribution of parameters as the solution to the SED analysis problem, which is computationally more demanding but allows a more formal and simultaneous estimation of parameter values and their uncertainties. More importantly, for the selection of model ingredients, the Bayesian approach also provides the very useful Bayesian

evidence, which can be considered as a quantified Occam's razor for effective model selection.

A noteworthy difference among the Bayesian SED analysis tools is that the earlier tools (e.g., MAGPHYS and CIGALE) are based on an irregular or regular grid search, while the newer generation of tools (e.g., GalMC and BayeSED) are based on more advanced random-sampling techniques such as Markov Chain Monte Carlo (MCMC; Sharma 2017; Hogg & Foreman-Mackey 2018) and nested sampling (NS; Skilling 2006; Buchner 2021; Ashton et al. 2022). The advantage of the grid-based Bayesian approach is that an SED library with regular or irregular model grids can be built in advance only once. Besides that, the prior probabilities can be set more freely during this procedure. Then it can be used in the analysis of a large sample of galaxies of any size without the generation of new SEDs. However, the size of the SED library needs to be very large to allow a reasonable parameter estimation for all galaxies in the sample, especially in the case of regular grids, where the number of required grids will increase dramatically with the number of free parameters. In contrast, a samplingbased Bayesian approach allows for a more detailed and efficient sampling of the parameter space for each galaxy and for a finer reconstruction of the posterior, leading to more reliable parameter estimates. However, theoretical SED synthesis needs to be done in real time and repeated many times, which could be very computationally expensive for the analysis of very large samples of galaxies. Fortunately, much more efficient SED synthesis models can be achieved with the help of machine-learning techniques. For example, in Han & Han (2014) we employed the artificial neural network and Knearest neighbor searching techniques to speed up the sampling-based Bayesian approach. The combination of sampling-based Bayesian inference and machine-learning techniques enables the detailed Bayesian SED analysis of very large samples of galaxies (Han et al. 2019). Although the training phase of a machine-learning-based SED synthesis method could be very time-consuming, especially for very complex SED models with many free parameters and the accurate synthesis of high-resolution SEDs, it is very promising with more advanced training techniques (Alsing et al. 2020; Gilda et al. 2021; Hahn & Melchior 2022; Qiu & Kang 2022) and it is worthwhile to carry out further exploration in this direction.

For the study of galaxy formation and evolution, the ideal SED synthesis and analysis tool should be able to simultaneously account for the contributions of stars, interstellar gas and dust, and AGN components, and to provide accurate and efficient estimates of the redshift and the physical properties of all components. However, in practice, it is very difficult, if not impossible, to fully satisfy all of these requirements. Therefore, a good SED synthesis and analysis tool should attempt to achieve a reasonable balance among these requirements as much as possible. This is what we are trying to achieve during the development of the BayeSED code (Han & Han 2012, 2014, 2019; Han et al. 2019, 2020). In this work, we will rigorously test the performance of the latest version of the BayeSED code combined with SED models with increasing complexity for simultaneous photometric redshift and stellar population parameter estimation of galaxies, so as to be ready for the analysis of the forthcoming massive data sets from the CSST wide-field multiband imaging survey and others.

We begin in Section 2 by introducing the methods we have employed for the generation of empirical statistics–based (Section 2.1) and hydrodynamical simulation–based (Section 2.2) mock catalogs of galaxies, observational error modeling (Section 2.3), and the selection of samples (Section 2.4) that will be used for the performance test. In Section 3, we briefly describe the Bayesian approach of photometric SED analysis methods, including parameter estimation (Section 3.1) and model selection (Section 3.2). In particular, in Section 3.3, we will introduce some runtime parameters of the MultiNest algorithm, which is the core engine of BayeSED. They need to be properly tuned to improve the performance of BayeSED. We present the results of the performance test for the case without SED modeling errors using an empirical statistics– based mock galaxy sample in Section 4. In Section 5, by employing the simplest SED model, we present the results of the performance test for the case with SED modeling errors about the SFH and dust attenuation law (DAL) of galaxies using a Horizon-AGN hydrodynamical simulation–based mock galaxy sample for a CSST-like imaging survey. In Section 6, we discuss the effectiveness of more flexible (or complex) forms of the SFH and DAL of galaxies for improving the performance of simultaneous redshift and stellar parameter estimation in CSST-like (Section 6.1), CSST+Euclid-like (Section 6.2), and COSMOS-like (Section 6.3) surveys with increasing discriminative power. In particular, we discuss the relation between the metrics of the quality of parameter estimation and Bayesian evidence, as well as how they depend on the different surveys. Finally, a summary of our results and conclusions is presented in Section 7.

Throughout this work, we assume a flat Lambda cold dark matter cosmology with H0 = 70 km s−1 Mpc−1 , Ωm = 0.3, and ΩΛ = 0.7 (Spergel et al. 2003; WMAP-5), and the Chabrier (2003) IMF. All the presented magnitudes are in the AB system (Oke 1974).

# 2. Bayesian Photometric SED Synthesis with BayeSED

The SED synthesis (or modeling) module is an essential part of any Bayesian SED fitting code. In BayeSED-V3, we have added more functions for SED synthesis, especially for the simulation of mock observations of galaxies in a Bayesian way. This is not just crucial for the current work, but also lays the foundation for future applications of machine-learning and simulation-based Bayesian inference methods in Bayesian SED fitting (Hahn & Melchior 2022; Hahn et al. 2022). For this work, we use empirical statistics–based (Section 2.1) and hydrodynamical simulation–based (Section 2.2) methods to generate mock photometric catalogs, add noise with a simple magnitude limit–based approach (Section 2.3), and select a sample (Section 2.4) similar to those in previous works for the performance test in the next two sections. In the following, we describe these in more detail.

# 2.1. Empirical Statistics–Based Photometric Mock Catalog

The first method to generate a mock photometric catalog is built by randomly drawn samples from the parameter space of a particular SED model under the constraints of some empirical statistical properties of galaxies. The sampling is performed with the same nested sampler MultiNest in the Bayesian SED analysis mode of BayeSED. A sample from this catalog will be used in Section 4 to test the performance of redshift and stellar

population parameter estimation in the case where the SED modeling is perfect, since exactly the same SED modeling method will be used in the Bayesian SED analysis of it.

#### 2.1.1. SED Modeling

As in Han & Han (2019), the SED of a galaxy is modeled as the luminosity of starlight from stellar populations of varying ages and metallicities, transmitted through the interstellar medium (ISM) and the intergalactic medium (IGM) to the observer. Specifically, the luminosity emitted at wavelength λ by a galaxy with age = t can be given as

$$L_{\lambda}(t)=\int_{0}^{t}\,dt^{\prime}\,\psi(t-t^{\prime})\,S_{\lambda}[t^{\prime},\,Z(t-t^{\prime})]\,T_{\lambda}^{\rm ISM}(t,\,t^{\prime})\tag{1}$$

$$=T_{\lambda}^{\rm ISM}\int_{0}^{t}dt^{\prime}\,\psi(t-t^{\prime})\,S_{\lambda}[t^{\prime},Z(t-t^{\prime})],\tag{2}$$

where *y*( ) *t t* - ¢ is the SFH describing the star formation rate (SFR) as a function of the time t − t′, and *Sl* [ ( )] *t Zt t* ¢ -¢ , is the luminosity emitted per unit wavelength per unit mass by a simple stellar population (SSP) of age *t*¢ and metallicity *Zt t* ( ) - ¢ .

¢ *T tt l* , ISM( ) is the transmission function of the ISM (Charlot & Longhetti 2001), which is contributed by two components:

$$T_{\lambda}^{\rm ISM}(t,\,t^{\prime})=T_{\lambda}^{+}(t,\,t^{\prime})T_{\lambda}^{0}(t,\,t^{\prime}),\tag{3}$$

where ¢ *l* + *T tt* ( ) , and ¢ *T tt l* , 0 ( ) are the transmission functions of the ionized gas and the neutral ISM, respectively. The transmission through ionized gas can be modeled with a photoionization code such as CLOUDY. However, we set *l* ¢ = + *T tt* ( ) , 1 in this work to be consistent with the hydrodynamical simulation–based catalog (Section 2.2). A detailed modeling of ¢ *l* + *T tt* ( ) , with CLOUDY to account for the combined effects of starlight absorption, nebular line emission, ionized continuum emission, and possible emission from warm dust within H II regions will be presented in a companion paper. Meanwhile, the transmission function of the neutral ISM is considered with a simple time-independent DAL and uniformly applied to the whole galaxy.

*Zt t* ( ) - ¢ is the stellar metallicity as a function of the time t − t′, which describes the chemical evolution history (CEH) of the galaxy. In previous works, we assumed a time-independent metallicity, i.e., *Zt t Z* ( ) -¢= 0, as in many SED fitting codes for galaxies. To properly consider the evolution of stellar metallicity, we additionally employ a linear SFH-to-metallicity mapping model (Driver et al. 2013; Robotham & Bellstedt 2020; Thorne et al. 2021; Alsing et al. 2023):

$$Z(t-t^{\prime})=(Z(t_{\rm age})-Z_{\rm min})\,\frac{1}{M}\int_{0}^{t}\,\psi(t-t^{\prime})dt^{\prime}+Z_{\rm min}.\tag{4}$$

Generally, the main ingredients for our SED modeling of galaxies are the SSP model, SFH, CEH, and DAL. In this work, for the construction of the Horizon-AGN hydrodynamical simulation–based catalog (Section 2.2), we use the SSP model assuming a Chabrier (2003) stellar IMF from the widely used stellar population synthesis model of Bruzual & Charlot (2003). The SFH of galaxies is typically parameterized in exponentially declining form: SFR(t) ∝ e − t/ τ (hereafter the τ model). The τ model only describes the SFH of galaxies in a closed box without inflow of pristine gas and outflow of processed gas, where the gases are converted to stars at a rate proportional to the remaining gas and with a fixed efficiency (Schmidt 1959; Tinsley 1980). It is widely discussed in the literature that this simple assumption may lead to systematically biased estimation of stellar population parameters, especially for galaxies at z 2 (Lee et al. 2009, 2010; Reddy et al. 2012; Ciesla et al. 2017; Carnall et al. 2018). Therefore, some more flexible and physically inspired forms of models have been suggested to improve the measurement of SFHs of galaxies and the estimation of their stellar population parameters and photometric redshift (Pacifici et al. 2012; Ciesla et al. 2017; Iyer & Gawiser 2017; Carnall et al. 2019; Iyer et al. 2019; Leja et al. 2019; Lower et al. 2020; Suess et al. 2022).

In the present work, we employ three extensions of the τ model with different complexities. The first one is described as

$\psi(t)\propto t^{\beta}\times\exp(-t/\tau)$, (5)

which is just an extended form of the delayed-τ model (Lee et al. 2010). Apparently, the typical τ model and delayed-τ model are just two special cases of this model (hereafter the β-τ model) with β = 0 and β = 1, respectively. The second one is the β-τ model combined with a quenching (or rejuvenation) component, which is described as (Ciesla et al. 2016)

$$\psi(t)\propto\left\{\begin{array}{ll}t^{\beta}\times\exp(-t/\tau),&\mbox{whent\leqslant t_{\rm trunc}}\\ r_{\rm SFR}\times\psi(t=t_{\rm trunc}),&\mbox{whent>t_{\rm trunc}}\end{array}\right.\tag{6}$$

where ttrunc is the time when the star formation is quenched (rSFR < 1) or rejuvenated (rSFR > 1), and rSFR is the ratio between ψ(t > ttrunc) and ψ(t = ttrunc):

$$\begin{array}{l}\mbox{\rm{FSFR}}=\frac{\psi(t>t_{\rm{trunc}})}{\psi(t_{\rm{trunc}})}.\end{array}\tag{7}$$

This model (hereafter the β-τ-r model) is a further extension of the β-τ model with the latter being a special case with rSFR = 1. The third one is the double-power-law model (Diemer et al. 2017; Carnall et al. 2018; Alsing et al. 2023) combined with a quenching (or rejuvenation) component, which is described as

$$\psi(t)\propto\left\{\begin{array}{cc}\frac{1.0}{\left(\frac{t}{t^{*}}\right)^{\alpha}+\left(\frac{t}{t^{*}}\right)^{-\beta}},&\text{when}t\leqslant t_{\text{trunc}}\\ r_{\text{SFR}}\times\psi(t=t_{\text{trunc}}),&\text{when}t>t_{\text{trunc}}\end{array}\right.\tag{8}$$

where α and β are the falling and rising slopes, respectively, and t * is related to the time at which star formation peaks, which is defined as t * ≡ τtage for the age of the galaxy tage. A major advantage of the double-power-law model is the decoupling of the rising and falling parts of the SFH. Therefore, this model (hereafter the α-β-τ-r model) is even more flexible than the β-τ-r model.

The DAL is another very important ingredient for the SED modeling of galaxies (Walcher et al. 2011; Conroy 2013). When deriving the photometric redshift and physical properties of galaxies from the analysis of their photometric or spectroscopic observations, a universal DAL as a simple uniform screen is commonly assumed. However, different choices of the universal law may lead to very different estimations of the photometric redshift and physical parameters of galaxies (Pforr et al. 2012, 2013; Salim & Narayanan 2020). In particular, many studies show that the dust attenuation

curves of different galaxies are very different (Kriek & Conroy 2013; Reddy et al. 2015; Salmon et al. 2016; Salim & Boquien 2019; Shivaei et al. 2020), and therefore there is no universal DAL as expected on theoretical grounds (Witt & Gordon 2000; Seon & Draine 2016; Narayanan et al. 2018; Lower et al. 2022). By a detailed study of the dust attenuation curves of about 230,000 individual galaxies in the local Universe using photometric data covering the UV to IR bands, Salim et al. (2018) presented new forms of attenuation laws that are suitable for normal star-forming galaxies, high-z analogs, and quiescent galaxies (see also Noll et al. 2009). In this work, we additionally employ this new form of DAL, which is parameterized as the following:

$$\frac{A_{\lambda,\rm mod}}{A_{V}}=\frac{k_{\lambda}}{R_{V}}=\frac{k_{\lambda,\rm Cal}}{R_{V,\rm Cal}}\bigg{(}\frac{\lambda}{5500\ \rm\AA}\bigg{)}^{\delta}+\frac{D_{\lambda}}{R_{V,\rm mod}},\tag{9}$$

where kλ,Cal/RV,Cal is the Calzetti et al. (2000) DAL with RV, Cal = 4.05. The power-law term with an exponent δ is introduced to deviate from the slope of the Calzetti et al. (2000) DAL. *RV*,mod is the δ-dependent ratio of total to selective extinction for the modified law. The term Dλ is introduced to add a UV bump. The relationship between *RV*,mod and δ is given by

$$R_{V,\rm mod}=\frac{R_{V,\rm Cal}}{(R_{V,\rm Cal}+1)(0.44/0.55)^{\delta}-R_{V,\rm Cal}}.\tag{10}$$

The UV bump following a Drude profile (Fitzpatrick 1986) is represented as

$$D_{\lambda}(E_{b})=\frac{E_{b}\lambda^{2}\gamma^{2}}{[\lambda^{2}-(\lambda_{0})^{2}]^{2}+\lambda^{2}\gamma^{2}},\tag{11}$$

with the amplitude Eb, fixed central wavelength λ0 = 0.2175 μm, and width γ = 0.35 μm.

In total, we consider six different combinations of SFH, CEH, and DAL with increasing complexity. A summary of these models, their parameters, and their priors is shown in Table 1. Finally, we also include the effect of IGM absorption with the description of Madau (1995). Other more recent considerations of IGM absorption are available in BayeSED. However, exploration of the effects of different choices of IGM absorption models on the redshift and stellar population parameter estimation is beyond the scope of this work and is therefore not conducted, which nevertheless will not change the conclusions given here.

#### 2.1.2. Galaxy Population Modeling

To model the galaxy population, we need to set the joint probability distribution that characterizes its statistical properties. The statistical properties of the galaxy population are the results of the complex physical processes that happened during the formation and evolution of the galaxies. In this work, we employ some widely discussed empirical statistical properties of galaxies to model the galaxy population phenomenologically. It should be mentioned that there are large uncertainties in these statistical properties, and we do not attempt to use the most up-to-date results for all of them in this work. The other choices of statistical properties of galaxies will not change the conclusions of this work.

Similar to Tanaka (2015) (see also Alsing et al. 2023), we assume that the joint probability distribution of the stellar

Table 1 Summary of SED Models, Parameters, and Priors

| Model | Parameter | Range | Prior |
| --- | --- | --- | --- |
|  | z | [0, 4] | U |
|  | * log(M M) | [8, 12] | U |
|  | c log age yr ( ) | [8, 10.14] | U |
|  | Av | [0, 4] | U |
| SFH = τ, CEH = no | log(Z Z 0 ) | [−2.30, 0.70] | U |
| DAL = Cal+00a | log yr (t ) | [6, 10] | U |
| SFH = τ, CEH = yes | log(( ) Zt Z age ) | [−2.30, 0.70] | U |
| DAL = Cal+00a | log yr (t ) | [6, 10] | U |
| SFH = τ, CEH = yes | log(( ) Zt Z age ) | [−2.30, 0.70] | U |
| DAL = Sal+18b | log yr (t ) | [6, 10] | U |
|  | Eb | [−2, 6] | U |
|  | δ | [−1.2, 0.4] | U |
| SFH = β-τ | log(( ) Zt Z age ) | [−2.30, 0.70] | U |
| CEH = yes | log yr (t ) | [6, 10] | U |
| DAL = Sal+18b | β | [0, 1] | U |
|  | Eb | [−2, 6] | U |
|  | δ | [−1.2, 0.4] | U |
| SFH = β-τ-r | log(( ) Zt Z age ) | [−2.30, 0.70] | U |
| CEH = yes | log yr (t ) | [6, 10] | U |
| DAL = Sal+18b | rSFR | [1e−6, 1e6] | LUd |
|  | ttrunc/tage | [0, 1] | U |
|  | β | [0, 1] | U |
|  | Eb | [−2, 6] | U |
|  | δ | [−1.2, 0.4] | U |
| SFH = α-β-τ-r | log(( ) Zt Z age ) | [−2.30, 0.70] | U |
| CEH = yes | * τ = t /tage | [0.007, 1] | U |
| DAL = Sal+18b | α | [0.01, 1000] | LU |
|  | β | [0.01, 1000] | LU |
|  | rSFR | [1e−6, 1e6] | LUd |
|  | ttrunc/tage | [0, 1] | U |
|  | Eb | [−2, 6] | U |
|  | δ | [−1.2, 0.4] | U |

Notes.

a Calzetti et al. (2000). b Salim et al. (2018). c We also apply the constraint that the age of the galaxy is less than the age of the Universe at z. d Uniform in log space.

population parameters and redshift of the galaxy population can be factorized as

$P(G,\,z)=P(M_{\pm},\,z)$  
  

$$\times\,P({\rm SFR}|M_{\pm},\,z)$$
 
$$\times\,P(A_{V}|{\rm SFR},\,z)$$
 
$$\times\,P({\rm age}|M_{\pm},\,z)$$
 
$$\times\,P(Z_{\rm MW}|M_{\pm},\,z).\tag{12}$$

The joint distribution of stellar mass and redshift is defined as

$P(M_{\pm},\,z)\propto\Phi(M_{\pm},\,z)dV(z)$, (13)

where Φ(M*, z) is the unnormalized stellar mass function and dV(z) is the differential comoving volume element. We employ the recent measurement of stellar mass function and its redshift evolution from Leja et al. (2020), while using a WMAP-5 (Spergel et al. 2003) cosmology for the comoving volume element.

Following Tanaka (2015), we assume that P(SFR|M*, z) can be expressed as the sum of two Gaussians to represent two distinct sequences formed by star-forming and quiescent galaxies:

$$P(\text{SFRI}M_{*},\,z)$$
 
$$\propto\frac{(1-f_{\text{Q}})}{\sigma_{\text{SF}}}\exp\left[-\frac{1}{2}\bigg{(}\frac{\log\text{SFR}-\log\text{SFR}_{\text{SF}}(M_{*},\,z)}{\sigma_{\text{SF}}}\bigg{)}^{2}\right]$$
 
$$+\frac{f_{\text{Q}}}{\sigma_{\text{Q}}}\exp\left[-\frac{1}{2}\bigg{(}\frac{\log\text{SFR}-\log\text{SFR}_{\text{SF}}(M_{*},\,z)+2}{\sigma_{\text{Q}}}\bigg{)}^{2}\right],$$

where * SFR , SF( ) *M z* is the mean SFR of star-forming galaxies given by

$${\rm SFR}_{\rm SF}(M_{\ast},\,z)={\rm SFR}^{\ast}(z)\times\frac{M_{\ast}}{10^{11}M_{\odot}}M_{\odot}\;{\rm yr}^{-1},\tag{15}$$

with

$$\text{SFR}^{*}(z)=\begin{cases}10\times(1+z)^{2.1}&(z<2)\\ 19\times(1+z)^{1.5}&(z\geq2),\end{cases}\tag{16}$$

and the fraction of quiescent galaxies is given as a function of stellar mass and redshift (Behroozi et al. 2013):

$$f_{\rm Q}(M_{\rm\#},\,z)=\left[\left(\frac{M_{\rm\#}}{10^{10.2+0.5z}M_{\odot}}\right)^{-1.3}+1\right]^{-1}.\tag{17}$$

As in Tanaka (2015), the dust attenuation is considered to positively correlate with SFR:

$$P(\tau_{V}|\text{SFR},\,z)\propto\frac{1}{\sigma_{\text{\tiny{T}V}}}\exp\left[-\frac{1}{2}\bigg{(}\frac{\tau_{V}-\langle\tau_{V}\rangle}{\sigma_{\text{\tiny{T}V}}}\bigg{)}^{2}\right],\tag{18}$$

where *st* = 0.5 *V* , and

$$\langle\tau_{V}\rangle=\begin{cases}0.2&\text{(SFR}_{0}<1)\\ 0.2+0.5\text{logSFR}_{0}&\text{(SFR}_{0}>1),\end{cases}\tag{19}$$

$${\rm SFR}_{0}=100\frac{\rm SFR}{\rm SFR}\tag{20}$$

Then, we use the relation between τV and AV to obtain P(AV| SFR, z).

The probability of the age of a galaxy is described conditionally on the stellar mass and redshift:

$$P(\text{age}|M_{\pm},\,z)\propto\exp\left[-\frac{\langle\text{SFR}\rangle}{\text{SFR}^{\ast}(z)}\right],\tag{21}$$

where

(SFR) = $M_{\star}$/age.  
  

This leads to a low probability for a massive galaxy with young age but to a high probability for a low-mass galaxy with the same age.

Finally, the probability of the mass-weighted stellar metallicity is modeled as

$$P(Z_{\rm MW}|M_{\rm s},\,z)$$
 
$$\propto\frac{1}{\sigma_{\rm log(Z_{\rm MW})}}\exp\left[-\frac{1}{2}\Bigg{(}\frac{\log(Z_{\rm MW})-\left\langle\log(Z_{\rm MW})\right\rangle}{\sigma_{\rm log(Z_{\rm MW})}}\Bigg{)}^{2}\Bigg{]},\tag{23}$$
  
  
where $\sigma_{\rm log(Z_{\rm MW})}=0.1$, and 

$$\langle\log(Z_{\rm MW})\rangle=0.40[\log M_{\rm s}-10]\tag{24}$$
 
$$+\ 0.67\exp(-0.50z)-1.04$$

is the redshift-dependent stellar mass and metallicity relation (Ma et al. 2016), which is predicted by using the highresolution cosmological zoom-in simulations from the Feedback in Realistic Environments project (Hopkins et al. 2014).

To generate an empirical statistics–based mock catalog of galaxies, we employ the MultiNest algorithm to draw samples from the joint probability distribution of the stellar population parameters and redshift of the galaxy population by setting P(G, z) in Equation (12) to be the likelihood function. Additionally, to simulate a magnitude-limited sample, we can set the likelihood function to 0 when the magnitude in a given band is larger than a given value. Since sampling points with a likelihood of 0 will be ignored by MultiNest, the obtained posterior sample can be used to build a magnitude-limited sample of mock galaxies with some physical constraints from the empirical statistical properties of the galaxies. More details about the selection of the magnitude-limited sample are presented in Section 2.4. The mock catalog can be built with the posterior sample of redshift and all physical parameters given by MultiNest. However, this is a weighted sample (Yallup et al. 2022), which cannot be directly used as a mock sample of galaxies. To build a more realistic mock sample of galaxies, we use the bootstrap resampling method to obtain an unweighted sample.

In total, we build six mock catalogs of galaxies by employing SED models with different combinations of SFH, CEH, and DAL and increasing complexity as shown in Table 1. The employed priors of the redshift and stellar population parameters are listed in the same table for each model. In Figure 1, we show the joint distributions of the redshift and physical parameters of the six empirical statistics–based mock galaxy populations. Although the same set of empirical statistics is employed, different SED models lead to slightly different distributions of parameters, especially for redshift and galaxy age. This is likely due to different mapping relations from the free parameters to the derived parameters. For example, different forms of SFH may lead to different relations between the age of the galaxy and its recent SFR.

# 2.2. Hydrodynamical Simulation–Based Photometric Mock Catalog

The second method to generate a mock photometric catalog is based on an SED library that is built by the postprocessing of galaxies from a hydrodynamical simulation. This catalog will be used in Section 5 to test the performance of redshift and stellar population parameter estimation in the case where the SED modeling is imperfect, since the SED modeling method employed in the Bayesian SED analysis will be very different from the one used to build the SED library.

We start from the rest-frame spectra of the galaxies, which are produced using the light cone from the cosmological hydrodynamical simulation Horizon-AGN (Dubois et al. 2014). The computation of these spectra, which accounts for the complex SFH and metal enrichment of Horizon-AGN galaxies and consistently includes dust attenuation, is described in detail by Laigle et al. (2019) and Davidzon et al. (2019). The dust attenuation of galaxies in the Horizon-AGN simulation is modeled for each stellar particle, assumed to be an SSP, by using the gas metal mass distribution as an approximation of the dust mass distribution, assuming a constant dust-to-metal mass ratio (Laigle et al. 2019). To obtain the amount of extinction at a given wavelength, the Weingartner & Draine (2001) model of Milky Way dust grains with RV = 3.1 and a prominent 2175 Å graphite bump is employed for postprocessing the simulated galaxies. As mentioned by Laigle et al. (2019), the overall attenuation curve becomes less steep and the bump tends to be reduced when summing up the contribution of all the SSPs to obtain the resulting galaxy spectrum. They also noticed that the average attenuation curve in the Horizon-AGN simulation cannot be well reproduced by the model of either Calzetti et al. (2000) or Arnouts et al. (2013). The more flexible form of DAL as given by Equation (9) is more likely to reproduce the attenuation curves of galaxies in the Horizon-AGN simulation. In order to isolate the possible differences in the convolution with the filter response function, the observational error modeling, and the consideration of IGM absorption, we choose to convert their rest-frame spectra of the mock galaxies to corresponding mock photometries with BayeSED, instead of using their virtual photometries directly.8 The consideration for the effects of IGM absorption is the same as in Section 2.1.1. Therefore, the differences between the empirical statistics– based (Section 2.1) and hydrodynamical simulation–based photometric mock catalogs are only driven by the different SFHs, CEHs, and DALs of the mock galaxies and the different distributions of their redshift and physical parameters.

# 2.3. Observational Error Modeling

The modeling of realistic errors on the flux is crucial for a meaningful performance test of redshift and stellar population parameter estimation. Here, we introduce the method we employ to compute the flux errors of the mock galaxies and perturb their fluxes accordingly. The flux error for a wavelength band i with an Nσ AB magnitude limit *m F* lim, lim, *i i* =- + 2.5 log 23.9 * ( ) is given by

$$\sigma_{F,i}=\sqrt{(F_{\rm lim,i}/N)^{2}+\sigma_{F,i,{\rm sys}}^{2}}\,,\tag{25}$$

where the flux limit *F*lim,*i* and the systematic flux error σF,i,sys are in units of μJy. Since the magnitude error σm ≈ 1.08574/(S/N) with the signal-to-noise ratio S/N = F/σF, we can obtain

$\sigma_{F,i,\rm sys}=\sigma_{m,\rm sys}*F_{i}/1.08574$. (26)

As in Cao et al. (2018), we assume a systematic magnitude error σm,sys = 0.02. The final mock flux is obtained by the original flux perturbed by a Gaussian noise ~ 0, *sF i*, 2 ( ). In practice, the magnitude limit may have a dispersion *sm i* ,lim, for galaxies with different sizes. So, the actually used magnitude limit is drawn from the Gaussian distribution *m*lim,*i*, *sm i* ,lim, 2 ( ). In this work, we set *sm i* ,lim, = 0.1. We generate three sets of mock catalogs for CSST-like, Euclid-like, and COSMOS-like surveys, respectively. A summary of the adopted depths in all

<sup>8</sup> We compare the two sets of photometry carefully and find that they are actually very similar, only with some differences at the very faint end.

Figure 1. The joint distributions of the redshift and physical parameters of the empirical statistics–based mock galaxy population produced with BayeSED combined with SED models of different complexities. With the same set of empirical statistics, different SED models lead to slightly different redshift and age distributions, which is likely due to different mapping relations from the free parameters to the derived parameters. Meanwhile, only the two SED models with a quenching component produce a clear region of quiescent galaxies below the star-forming main sequence.

bands of the three surveys is shown in Table 2. The response functions and modeled relation between the magnitude and magnitude error are shown in the panels of Figure 2 for the seven CSST bands, three Euclid bands, and 26 COSMOS bands. To separate the effects of observational errors on the accuracy of parameter estimation, we generate another two sets of mock catalogs without adding observational errors. In this case (the no-noise case in Figure 2), the magnitude errors are all fixed to 0.01, but the photometry is not perturbed accordingly.

# 2.4. Sample Selection

To test the performance of BayeSED, we select two sets of samples of galaxies with Ks < 24.7 (Laigle et al. 2019) and i + < 25 (Cao et al. 2018) 9 from the empirical statistics–based mock catalog (Section 2.1) and the hydrodynamical

<sup>9</sup> In Cao et al. (2018) and Zhou et al. (2022a, 2022b), only high-quality sources with S/N 10 in the g or i band were selected to test the performance of photometric redshift estimation. This selection is not used in this work to obtain a fuller picture for the performance of the CSST wide-field multiband imaging survey.

Table 2 A Summary of the Adopted Depths in All Bands for CSST-like, Euclid-like, and COSMOS-like Mock Observations

| Survey | Band | Depth |
| --- | --- | --- |
|  | NUV | 24.2 ± 0.1 |
|  | u | 24.2 ± 0.1 |
| CSST-like | g | 25.1 ± 0.1 |
| 5σ deptha | r | 24.8 ± 0.1 |
|  | i | 24.6 ± 0.1 |
|  | z | 24.1 ± 0.1 |
|  | y | 23.2 ± 0.1 |
| Euclid-like | J | 24.0 ± 0.1 |
| 5σ depthb | H | 24.0 ± 0.1 |
|  | Y | 24.0 ± 0.1 |
|  | u | 26.0 ± 0.1 |
|  | B | 26.4 ± 0.1 |
|  | V | 25.6 ± 0.1 |
|  | r | 25.9 ± 0.1 |
|  | + i | 25.6 ± 0.1 |
|  | ++ z | 25.3 ± 0.1 |
|  | Y | 24.7 ± 0.1 |
| COSMOS-like | J | 24.3 ± 0.1 |
| 5σ depthb | H | 24.0 ± 0.1 |
|  | Ks | 24.1 ± 0.1 |
|  | IB | 24−25 ± 0.1 |
|  | NB711 | 24.5 ± 0.1 |
|  | NB816 | 24.6 ± 0.1 |
|  | 3.6 μm | 24.9 ± 0.1 |
|  | 4.5 μm | 24.9 ± 0.1 |

Notes.

b The 3σ depths for the COSMOS-like survey provided by Laigle et al. (2016) have been converted to 5σ depths. The Euclid-like 5σ depth data is also provided by Laigle et al. (2016).

simulation–based mock catalog (Section 2.2), respectively. The first set of samples are obtained directly with BayeSED combined with SED models with different complexities by using the method presented in Section 2.1.2. The second sample is selected from the Horizon-AGN hydrodynamical simulation–based photometric catalogs for a COSMOS-like configuration,10 which contains 789,354 galaxies. We find that a sample with 10,000 galaxies is large enough for us to obtain stable results for the performance tests as presented in Sections 4 and 5. The redshift and magnitude distributions of the two samples are presented in Figure 3. When employing different SED models with different complexities and the same set of empirical statistics, the empirical statistics–based samples show some differences, especially in the redshift distribution. This is likely due to the different mapping relations from the physical parameters to the photometries and from the free parameters to the derived parameters for the different SED models. Generally, the hydrodynamical simulation–based sample is consistent with the empirical statistics–based samples. We attribute the difference between the two sets of samples to the different modeling of the SFH, CEH, and DAL of the galaxies

and their different distributions of redshift and physical parameters.

# 3. Bayesian Photometric SED Analysis with BayeSED

The general method for the application of Bayesian inference to the photometric SED analysis of galaxies is the same as that in Han & Han (2014, 2019). In this section, we introduce some special aspects of Bayesian parameter estimation (Section 3.1) and model selection (Section 3.2) that are relevant to the current work.

# 3.1. Bayesian Parameter Estimation

For the Bayesian analysis of the mock data generated in the last section, we employ the same SED modeling procedure and setting of priors for free parameters as in Section 2.1.1, while the commonly used Gaussian form of the likelihood function is employed. The performance of this Bayesian analysis, including its speed and quality, is crucial for the analysis of large samples of galaxies in the big data era. We need some metrics to quantify the performance of parameter estimation, which is the main subject of this work.

While the speed of parameter estimation can be easily quantified by the running time, some metrics for the quality of parameter estimation are required. Similar to Acquaviva et al. (2015), we use three metrics to quantify the quality of parameter estimation. Bias, which characterizes the median separation between the predicted and the true values, is defined as

BIA = Median($\Delta x$),

while precision, which describes the scatter between the predicted and the true values, is defined as

$\sigma_{\rm NMAD}=1.4826*$ Median($|\Delta x|$), (28)

where Δx = (xphot − xtrue)/(1 + xtrue) for redshift (Ilbert et al. 2009; Dahlen et al. 2013; Salvato et al. 2018), and D*xx x x x* =- - phot true true max true min ( )( ) for other parameters. The median-base definition makes them less sensitive to outliers (sources with unexpectedly large errors). The fraction of outliers is defined as

$$\mbox{OLF}=\frac{1}{N_{\rm obj}}\times N(|\Delta x|>0.15).\tag{29}$$

# 3.2. Bayesian Model Selection

An important advantage of NS-based algorithms, such as MultiNest, over MCMC-based methods is the ability to simultaneously carry out parameter estimation and model selection. While the main subject of this work is parameter estimation, it will also be interesting to explore the effects of the sampling parameters (namely, nlive and efr) of MultiNest on the computation of Bayesian evidence, a quantity that is crucial for Bayesian model selection.

In Han & Han (2019), we presented a mathematical framework to discriminate different assumptions about the SSP, DAL, and SFH in the SED modeling of galaxies based on the Bayesian evidence for a sample of galaxies. In this work, since the SSP model employed in the generation of mock data is the same as that employed in their Bayesian analysis, we do not need to consider different choices of SSPs. So, the problem is

a The 5σ depths for extended sources in the CSST wide-field multiband imaging survey (Gong et al. 2019). The CSST deep survey can be at least 1 mag deeper. The results of performance tests for the latter will be presented in future works.

<sup>10</sup> https://www.horizon-simulation.org/PHOTOCAT/HorizonAGN_ LAIGLE-DAVIDZON+2019_COSMOS_v1.6.fits

Figure 2. (a) Response functions for CSST and Euclid bands. (b) The modeled relation between magnitude and magnitude error for CSST bands. Sources with S/N < 1 (i.e., σm,i > 1.08574) are considered as nondetections. The nondetections in a wavelength band i with an Nσ flux limit *F*lim,*i* or magnitude limit *m*lim,*i* are represented as sources with *F F i i* = lim, and *sFi i* , lim, = -*F N* (the flux case) or with *m m i i* = lim, and σm,i = −1.08574/N (the magnitude case). These conventions make sure the conversion between flux data and magnitude data in the input file of BayeSED is consistent. (c)–(d) Same as (a) and (b), but for the COSMOS bands. For clarity, the 12 intermediate bands (IBs) and 2 narrow bands (NBs) are not shown.

significantly simplified. In this work, we focus on the computation of Bayesian evidence for the SED modeling of a sample of galaxies with the SSP, SFH, and DAL all being assumed to be universal (i.e., an M(ssp0, sfh0, dal0)-like model (see Section 5.1 of Han & Han 2019)). The sample Bayesian evidence in this case (as in Equation (33) of Han & Han 2019) is ¼ *dd dM I p* , , , ssp , sfh , dal , *N* 1 2 0 0 0 ( ∣( ))

$$=\prod_{g=1}^{N}\int p(d_{g}|\theta_{g},\,M(\mbox{ssp}_{0},\,\mbox{sfh}_{0},\,\mbox{dal}_{0}),\,I)$$
 
$$p(\theta_{g}|M(\mbox{ssp}_{0},\,\mbox{sfh}_{0},\,\mbox{dal}_{0}),\,I)d\theta_{g}$$
 
$$=\prod_{g=1}^{N}\,p(d_{g}|M(\mbox{ssp}_{0},\,\mbox{sfh}_{0},\,\mbox{dal}_{0}),\,I).\tag{30}$$

Although the detailed SFH and DAL of different galaxies can vary significantly, the sample Bayesian evidence computed in this manner remains valuable for identifying the most efficient combination of SFH and DAL for analyzing a vast sample of galaxies, such as the one provided by the CSST wide-field imaging survey.

In practice, the natural logarithm of Bayesian evidence is commonly used for Bayesian model selection. Therefore, Equation (30) can be rewritten as

$$\ln(\text{BE})\equiv\ln(p(\mathbf{d}_{1},\mathbf{d}_{2},...,\mathbf{d}_{N}|\mathbf{M}\left(\text{ssp}_{0},\text{sfh}_{0},\text{dal}_{0}\right),\mathbf{I}))$$
 
$$=\sum_{g=1}^{N}\ln(p(\mathbf{d}_{g}|\mathbf{M}\left(\text{ssp}_{0},\text{sfh}_{0},\text{dal}_{0}\right),\mathbf{I})),\tag{31}$$

where ln ssp , sfh , dal , (( ∣ ( ) ) *p dM I g* 0 0 0 ), the Bayesian evidence for an individual galaxy, can be directly obtained in BayeSED with MultiNest. However, the individual Bayesian evidence

Figure 3. The comparison of the redshift and magnitude distributions of the empirical statistics–based and Horizon-AGN hydrodynamical simulation– based mock galaxy samples.

estimated with MultiNest contains errors. A stricter Bayesian model selection should consider the effects of error propagation. In our case, the error of the sample Bayesian evidence ln BE ( ) is simply the sum of errors for the individual galaxies, which is provided by MultiNest as well.

The minimum χ2 method is also widely used for model selection. For the case with Gaussian observational errors, there is only a constant difference between the minimum χ2 and the natural logarithm of maximum likelihood. The sample maximum likelihood (as in Equation (32) of Han & Han 2019) is

$$\mathcal{L}_{\max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\ldots,\hat{\theta}_{N})$$
 
$$\equiv\max_{\theta_{1},\theta_{2},\ldots,\theta_{N}}[p(\mathbf{d}_{1},\,\mathbf{d}_{2},\ldots,\mathbf{d}_{N}|\theta_{1},\,\theta_{2},\ldots,\theta_{N},$$
 
$$\mathbf{M}(\mathrm{ssp}_{0},\,\mathrm{sfh}_{0},\,\mathrm{dal}_{0}),\,\mathbf{I})]$$
 
$$=\prod_{g=1}^{N}\max_{\theta_{g}}[p(\mathbf{d}_{g}|\theta_{g},\,\mathbf{M}(\mathrm{ssp}_{0},\,\mathrm{sfh}_{0},\,\mathrm{dal}_{0}),\,\mathbf{I})].\tag{32}$$

Then, the natural logarithm of the sample maximum likelihood is

$$\ln(\text{ML})\equiv\ln(\mathcal{L}_{\max}(\hat{\theta}_{1},\,\hat{\theta}_{2},...,\hat{\theta}_{N}))$$
 
$$=\sum_{g=1}^{N}\ln\Biggl{(}\max_{\theta_{g}}\left[p(\mathbf{d}_{g}|\theta_{g},\,\mathbf{M}\left(\text{ssp}_{0},\,\text{sfh}_{0},\,\text{dal}_{0}\right),\,\mathbf{I})\right]\Biggr{)},\tag{33}$$

where *q q* ln max , ssp , sfh , dal , *p dM I g g* 0 0 0 *g* ( [ ( ∣ ( ) )]), the natural logarithm of maximum likelihood for an individual galaxy, can be directly obtained in BayeSED with MultiNest. As in the model selection with Bayesian evidence, only the difference in ln ML ( ) between different models is useful for the model selection. Therefore, the model selection with ln ML ( ) is equivalent to that with the minimum χ2 . In Section 6, we will discuss the difference between the two model selection methods.

# 3.3. Runtime Parameters of MultiNest Algorithm

As the Bayesian inference engine of BayeSED, MultiNest has some runtime parameters. The values of these runtime parameters have very important effects on the performance of BayeSED for the redshift and stellar population parameter estimation of galaxies. Here, we briefly introduce the meaning of these runtime parameters of the MultiNest algorithm.

NS (Skilling 2004, 2006), as a Monte Carlo method primarily designed for the efficient computation of Bayesian evidence, allows posterior inference as a by-product at the same time. So, it provides a way to simultaneously carry out Bayesian parameter estimation and model selection. As an algorithm built on the NS framework, MultiNest (Feroz & Hobson 2008; Feroz et al. 2009) is special for its efficiency in sampling from posteriors that may contain several modes and/ or degeneracies. It has been improved further by the implementation of importance NS (INS) (Cameron & Pettitt 2014; Feroz et al. 2019) to increase the efficiency of evidence computation. In the latest version of BayeSED, version 3.12 of MultiNest, which includes the implementation of INS, is employed.

Similar to most other NS algorithms, MultiNest explores the posterior distribution by maintaining a fixed number (see also Higson et al. 2019 and Speagle 2020 for new methods using a variable number) of samples drawn from the prior distribution, called live points, and iteratively replaces the point with the lowest likelihood value (the dead point) with another point drawn from the prior having a higher value of likelihood. While there are many runtime parameters of MultiNest that can be set in BayeSED, only two of them are of particular importance. They largely determine the accuracy and computational cost for the running of the MultiNest algorithm and therefore of BayeSED. The first one is the total number of live points (nlive), which determines the effective sampling resolution. The second one is the target sampling efficiency (efr), which determines the ratio of points accepted to those sampled. Generally, a larger nlive and lower efr lead to more accurate posteriors and evidence values but higher computational cost. The optimal values of nlive and efr should be problemdependent, although an efr equal to 0.8 and 0.3 is recommended by the authors of MultiNest for parameter estimation and evidence evalutaion, respectively.

In this work, we will explore the effects of nlive and efr on the estimation of photometric redshift and stellar population parameters. The results are presented in Sections 4.1 and 4.2, respectively.

# 4. Results of Performance Tests Using Empirical Statistics– Based Mock Galaxy Sample

In this section, we present the results of performance tests of photometric redshift and stellar population parameter estimation by using an empirical statistics–based mock galaxy sample for the CSST wide-field multiband imaging survey. Since the SED model employed in the Bayesian parameter estimation is exactly the same as that used in the generation of mock observations, the error of parameter estimation is mainly contributed by random error in the data, parameter degeneracies, the stochastic nature of the employed MultiNest sampling algorithm, and other potential errors in the BayeSED code. To separate out the effects of random photometric error in the data, we will consider two cases, one with and one without random noise added to the photometric data. To find out the optimal run parameters, we consider six different choices of target sampling efficiency (efr) and eight choices of number of live points (nlive) for the MultiNest sampling algorithm. Furthermore, we compare the performance of different SED models with increasing complexity in terms of running time and quality of parameter estimation.

# 4.1. Photometric Redshift Estimation

The results of performance tests for photometric redshift estimation are shown in Figure 4. In Figure 4(a), we show the results for only the simplest SED model (SFH = τ, –CEH, DAL = Cal+00) employed in this work. As shown in the top right panel of this figure, in the case without noise, there is a clear anticorrelation between the computation time (or the sampling resolution nlive) and the error σNMAD (defined in Equation (28)) of photometric redshift estimation. Meanwhile, a lower efr makes the anticorrelation converge faster with the increase of nlive. There is a clear lower limit for the value of σNMAD, which is about 0.006. As shown in the top left panel of Figure 4(a), in the case with noise, the error of photometric redshift estimation does not always decrease with the sampling resolution nlive. When we set efr = 0.1, the lowest error (0.056) of redshift estimation is obtained when nlive is about 25. When nlive > 25, the error of redshift estimation starts to increase with nlive and finally converges to ∼0.058. This is most likely due to overfitting to the noise added to the mock data.

The middle two panels of Figure 4(a) show the relation between the computation time (or nlive) and the bias (defined in Equation (27)) of photometric redshift estimation. In the case with noise, the relation between the computation time (or nlive) and bias has almost the opposite profile of that of the error σNMAD. However, the bias of photometric redshift estimation is generally very small, which is almost zero in the noisefree case.

The bottom two panels of Figure 4(a) show the relation between the computation time (or nlive) and the fraction of outliers OLF (defined in Equation (29)) of photometric redshift estimation. Similar to that for σNMAD, in the noise-free case, there is a clear anticorrelation between the computation time (or nlive) and OLF. In this case, the lower limit for the value of OLF is about 0.002. In the case with noise, the relation between the computation time (or nlive) and OLF has the same profile as that of the error σNMAD. When we set efr = 0.1, the lowest OLF (0.215) of redshift estimation is also obtained when nlive is about 25. When nlive > 25, the OLF of redshift estimation starts to increase with nlive and finally converges to ∼0.225.

In Figure 4(b), we show the results for all of the six SED models with increasing complexity. Here, only the results with efr = 0.1 are shown. In the case without noise, as shown in the top right panel of this figure, the error σNMAD of photometric redshift estimation tends to converge to a larger value when a more complicated SED model is employed. This is not strange, since more complicated SED models have more free parameters and thus suffer from more severe parameter degeneracies. Besides, more complicated SED models apparently require longer running time. The bias of redshift estimation is always very small no matter which SED is employed. In general, when a more complicated SED model is employed, the OLF of redshift estimation also increases significantly, and decreases much more slowly with the increase of nlive.

In the case with noise, as shown in the left panels of Figure 4(b), the results are a little more complicated. For the first three simplest SED models, the error σNMAD of photometric redshift estimation apparently increases with the increase of model complexity. However, when more complicated forms of SFH are considered, σNMAD starts to decrease with the increase of model complexity, although not very significantly. Meanwhile, the most complicated SED model (SFH = α-β-τ-r, +CEH, DAL = Sal+18) leads to the smallest absolute value of bias, although the bias is actually very small in all cases. The situation for OLF is somewhat similar to that of σNMAD. No matter which SED model is employed, when nlive 25, both the σNMAD and the OLF of redshift estimation start to increase, and then slowly decrease to a stable value.

# 4.2. Stellar Population Parameter Estimation

In this subsection, we show the results of performance tests for the photometric stellar population parameter estimation. While the estimates of many stellar population parameters are available, we only show the results for stellar mass and SFR, which are two of the most important physical parameters for the study of the formation and evolution of galaxies.

### 4.2.1. Stellar Mass

The results of performance tests for stellar mass estimation are shown in Figure 5. In Figure 5(a), we show the results for only the simplest SED model (SFH = τ, –CEH, DAL = Cal +00) employed in this work. As shown in the top right panel of

Figure 4. Performance test with empirical statistics–based mock galaxy sample for the photometric redshift estimation of galaxies in the CSST wide-field multiband imaging survey. (a) The results for only the simplest SED model (SFH = τ, –CEH, DAL = Cal+00) employed in this work. We consider six different choices of efr (target sampling efficiency) as shown by the different symbols. For each efr, we consider eight cases with the number of live points (nlive), which determines the effective sampling resolution, equal to 10, 15, 20, 25, 50, 100, 200, and 400, respectively. The relations between the computation time (in s/object/CPU, by employing one core of a 2.2 GHz CPU) and the performance metrics σNMAD (top panels), BIA (middle panels), and OLF (bottom panels) are shown. The results for the two cases with (left panels) and without (right panels) observational noise in the mock data are shown. In general, a larger value of nlive and a smaller value of efr lead to better quality of redshift estimation, but with the cost of longer running time. (b) The results for six different SED models with increasing complexity. Only the results with efr = 0.1 are shown. In general, more complicated SED models require longer running time. They also lead to worse quality of photometric redshift estimation, which is likely due to more severe parameter degeneracies. Actually, for the last four more complicated SED models with the DAL of Salim et al. (2018), the number of free parameters is greater than the number of photometric data points (seven for the CSST imaging survey), as shown in Table 1.

Figure 5. Same as Figure 4, but for the stellar mass estimation. The quality of stellar mass estimation, in terms of σNMAD, BIA, and OLF, is worse than that of redshift estimation and more sensitive to the selection of SED models. In the case with noise, for the most complicated SED model (SFH = α-β-τ-r, +CEH, DAL = Sal+18) used in this work, the σNMAD, bias, and OLF of stellar mass estimation increase significantly when nlive > 100. This should be a clear indication of overfitting to the noise in the data. In general, more complicated SED models lead to worse quality of stellar mass estimation.

this figure, in the case without noise, there is also a clear anticorrelation between the computation time (or the sampling resolution nlive) and the error σNMAD of photometric stellar mass estimation. The behavior of the error σNMAD of stellar mass with respect to the change of efr is similar to that of the photometric redshift. There is also a clear lower limit for the value of σNMAD, which is about 0.04. As shown in the top left panel of Figure 5(a), in the case with noise, the error of stellar mass estimation does not always decrease with the sampling resolution nlive as well. When we set efr = 0.1, the lowest error (∼0.1130) of stellar mass estimation is also obtained when nlive is about 25. When nlive > 25, the error of stellar mass estimation only slightly increases with nlive. The error of stellar mass is about two times larger than that of photometric redshift estimation. The bias of stellar mass estimation is also larger, but still very small when compared with σNMAD. In the noise-free case, there is also a clear anticorrelation between the computation time (or nlive) and the OLF of stellar mass estimation, where the lower limit for the value of OLF is about 0.03. In the case with noise, when we set efr = 0.1, the lowest OLF (0.285) of stellar mass estimation is also obtained when nlive is about 25. When nlive > 25, the OLF of stellar mass estimation only slightly increases with nlive as well. The OLF of stellar mass estimation is slightly larger than that of photometric redshift estimation.

In Figure 5(b), we show the results for all of the six SED models with increasing complexity, where only the results with efr = 0.1 are shown. In the two cases with or without noise, as shown in the top right panel of this figure, the error σNMAD of photometric stellar mass estimation tends to converge to a larger value when a more complicated SED model is employed. The same is true for the OLF of photometric stellar mass estimation. The behavior of bias is somewhat different, but it is generally very small when compared with σNMAD. In the case with noise, for the most complicated SED model (SFH = α-βτ-r, +CEH, DAL = Sal+18) used in this work, the σNMAD, bias, and OLF of stellar mass estimation increase significantly when nlive > 100. This should be a very clear indication of overfitting to the noise in the data. In general, more complicated SED models lead to worse quality of stellar mass estimation.

#### 4.2.2. SFR

The results of performance tests for SFR estimation are shown in Figure 6. In Figure 6(a), we show the results for only the simplest SED model (SFH = τ, –CEH, DAL = Cal+00) employed in this work. Similar to the results for photometric redshift and stellar mass estimation, in the case without noise, there is a clear anticorrelation between the computation time (or the sampling resolution nlive) and the error σNMAD of photometric SFR estimation. The behavior of the error σNMAD of SFR with respect to the change of efr is similar to that of the photometric redshift and stellar mass. There is also a clear lower limit for the value of σNMAD, which is about 0.02. As shown in the top left panel of Figure 5(a), in the case with noise, the error of SFR estimation increases apparently when the sampling resolution nlive 25. Generally, the error of SFR estimation is slightly smaller than that of stellar mass estimation. The bias of SFR estimation is also slightly smaller, and ignorable with respect to σNMAD. In the noise-free case, the relation between the computation time (or nlive) and the OLF of SFR estimation is somewhat different from that of

photometric redshift and stellar mass. Even with nlive = 500, the OLF of SFR estimation does not seem to converge. The lower limit for the value of OLF seems near 0.1. In the case with noise, the OLF of SFR estimation converges much faster to about 0.255 when we set efr = 0.1. This is slightly smaller than that of stellar mass estimation.

In Figure 6(b), we show the results for all of the six SED models with increasing complexity, where only the results with efr = 0.1 are shown. In the two cases with or without noise, as shown in the top right panel of this figure, the error σNMAD of SFR estimation tends to converge to a larger value when a more complicated SED model is employed, and is more sensitive to the selection of SED models than that of stellar mass. The same is true for the OLF of SFR estimation. The behavior of bias is somewhat different, but it is generally very small when compared with σNMAD. In the case with noise, for the four most complicated SED models used in this work, the σNMAD, bias, and OLF of SFR estimation significantly increase with nlive. This is an even clearer indication of overfitting to the noise in the data. In general, more complicated SED models lead to worse quality of SFR estimation.

# 4.3. Computation of Bayesian Evidence

In this subsection, we present the results of performance tests for the computation of Bayesian evidence, a quantity that is crucial for Bayesian model selection.

In Figure 7(a), we show the results for only the simplest SED model (SFH = τ, –CEH, DAL = Cal+00) employed in this work. As shown in the top and middle panels of this figure, the Bayesian evidence computed with importance sampling is more stable than that computed without importance sampling, especially in the case with noise. So, hereafter and especially in Sections 5 and 6, all mentioned Bayesian evidence has been computed with importance sampling. The value of Bayesian evidence increases with the number of live points (nlive), which determines the effective sampling resolution. In all cases, it eventually converges to a stable value when nlive is very large, while a lower sampling efficiency (efr) leads to a faster convergence rate. A good balance between the speed and quality of Bayesian evidence estimation can be achieved when the MultiNest runtime parameters efr and nlive equal 0.1 and 50, respectively.

On the other hand, as shown in the bottom panels of Figure 7(a), the error of Bayesian evidence decreases with nlive, while a higher sampling efficiency (efr) also leads to a faster convergence rate. However, unlike the value of Bayesian evidence, the error of Bayesian evidence converges more slowly with the increase of nlive in all cases. As a result, if we set efr = 0.1 and nlive = 50, the error of Bayesian evidence would be overestimated. A much larger value of nlive seems required to obtain a more reliable estimation for the error of Bayesian evidence with MultiNest, which would be very computationally expensive and not suitable for the analysis of massive photometric data. However, in practice, this may not be a serious issue, since an overestimated error of Bayesian evidence only leads to a more conservative conclusion about model comparison. We just need to keep this in mind.

In Figure 7(b), we show the results for all SED models with increasing complexity, where only the results with efr = 0.1 are shown. Although the data used for the computation of Bayesian evidence is different for the different SED models, the value of Bayesian evidence clearly decreases with the increase of

Figure 6. Same as Figure 4, but for the SFR estimation. The quality of SFR estimation, in terms of σNMAD, BIA, and OLF, is slightly better than that of stellar mass estimation, but even more sensitive to the selection of SED models. In the case with noise, for the four most complicated SED models used in this work, the σNMAD, bias, and OLF of SFR estimation significantly increase with nlive. This is an even clearer indication of overfitting to the noise in the data. In general, more complicated SED models lead to worse quality of SFR estimation.

Figure 7. Performance test of Bayesian evidence estimation with empirical statistics–based mock galaxy sample for CSST imaging survey. The relations between the computation time (in seconds per object by employing a single 2.2 GHz CPU core) and the value of the natural logarithm of Bayesian evidence (as computed with Equation (31)) and its error (bottom panels) for the whole galaxy sample are shown. The results for two versions of Bayesian evidence with (top panels) or without (middle panels) importance sampling are shown. (a) The results for only the simplest SED model (SFH = τ, –CEH, DAL = Cal+00) employed in this work. We consider six different choices of efr (target sampling efficiency) as shown by the different symbols. The results for seven cases with nlive equal to 15, 20, 25, 50, 100, 200, or 400 are shown. The left panels show the results with noisy data while the right panels show the results with noise-free data. (b) The results for six different SED models with increasing complexity. Only the results with efr = 0.1 are shown. The value of Bayesian evidence clearly decreases with the increase of complexity of the SED model.

complexity of the SED model. This is reasonable. Since the same SED model is employed for the generation of mock data and their Bayesian analysis, the mock data can always be

interpreted well. However, a more complicated SED model is penalized for being distributed over a larger space, of which only a smaller fraction is useful for the given mock data.

# 5. Results of Performance Tests Using Hydrodynamical Simulation–Based Mock Galaxy Sample

In this section, we present the results of performance tests of photometric redshift and stellar population parameter estimation by using a hydrodynamical simulation–based mock galaxy sample for a CSST-like imaging survey. Only the results obtained with the simplest SED model are shown. In Section 6, we will discuss the effect of a more flexible SFH and DAL for CSST-like, CSST+Euclid-like, and COSMOS-like surveys.

As mentioned in Section 2.2, the generation of this mock galaxy sample accounts for the complex SFH and metal enrichment of Horizon-AGN galaxies, and consistently includes dust attenuation. However, for the Bayesian analysis of this more theoretical mock galaxy sample, we first employ the widely used assumptions about SFH (exponentially declining), MEH (constant but free), and dust attenuation (uniform foreground dust screen with Calzetti et al. 2000 DAL). The results in this section will help us to quantify the systematic errors resulting from these simplified assumptions. Further, as mentioned in Section 2.4, the galaxies in the mock sample used here are selected with Ks < 24.7 and i + < 25. In the literature, it is quite common to exclude some pathological cases with a kind of χ2 selection (Caputi et al. 2015; Davidzon et al. 2017; Laigle et al. 2019) before presenting the results of the performance test. However, no such cut is made here because the pathological cases are precisely what we want to investigate.

# 5.1. Photometric Redshift Estimation

In Figure 8, we investigate the performance of BayeSED combined with the simplest SED model to estimate the photometric redshifts of the hydrodynamical simulation–based mock galaxy sample for a CSST-like imaging survey. As a reference, panel (a) of this figure shows the ideal case without observational noise and SED modeling errors. So, in this case, the effects of parameter degeneracies, the stochastic nature of the MultiNest sampling algorithm, and other potential errors in the BayeSED code are the sources of error. As shown clearly, the total error from all of these sources is very small. By comparing panels (a) and (b) of this figure, with only the observational noise added, the σNMAD of photometric redshift estimation increases by eight times and the OLF increases by more than 40 times. The bias also increases, but is ignorable with respect to σNMAD. By comparing panels (a) and (c) of this figure, with only the error from the imperfect SED modeling added, the σNMAD of photometric redshift estimation increases by more than 3 times and the OLF decreases slightly. Further, there are some additional systematic patterns in the relation between the true and estimated values of photometric redshift. The bias also increases and is comparable to σNMAD. In general, the observational noise is a more important source of error for the photometric redshift estimation of galaxies, although the other one is also very important.

Finally, as shown in Figure 8(d), when all sources of error are included, the σNMAD of photometric redshift estimation increases to 0.097, the OLF increases to 0.264, and the bias becomes 0.003. The systematic patterns shown in panel (c) seem to be hidden due to the added noise. The algorithm seems to be struggling to estimate the photometric redshifts correctly by only using the seven-band photometries from the CSST-like imaging survey, especially for galaxies with redshift larger than 1.

# 5.2. Stellar Population Parameter Estimation

In Figure 9, we investigate the performance of BayeSED combined with the simplest SED model to estimate the stellar population parameters of hydrodynamical simulation–based mock galaxies for a CSST-like imaging survey. As a reference, panel (a) of this figure shows the ideal case without observational noise and SED modeling errors. In this ideal case, the σNMAD, bias, and OLF of stellar mass estimation are 0.047, 0.005, and 0.052, respectively. As shown in panel (b), with only the observational noise added, the results become 0.115, 0.02, and 0.283, respectively. As shown in panel (c), with only the error from imperfect SED modeling added, the results become 0.103, 0.003, and 0.141, respectively. By comparing panels (a) and (c), the performance of stellar mass estimation is severely affected by the simplified assumptions in the SED modeling. However, by comparing panels (b) and (c), the observational noise is the more important source of error for the photometric stellar mass estimation of galaxies, although the other one is also very important. Finally, as shown in panel (d) of this figure, when all sources of error are included, the σNMAD of photometric stellar mass estimation increases to 0.135, the bias increases to 0.034, and the OLF increases to 0.341. The algorithm seems to be struggling even more to estimate the photometric stellar mass correctly by only using the seven-band photometries from the CSST-like imaging survey.

Similarly, Figure 10 shows the performance of BayeSED combined with the simplest SED model to estimate the SFR of the hydrodynamical simulation–based mock galaxy sample for a CSST-like imaging survey. Compared with the results for stellar mass estimation in Figure 9, the photometric SFR estimation is even more severely affected by the simplified assumptions in the SED modeling. Actually, by comparing panels (b) and (c) of this figure, it is clear that the error from the imperfect SED modeling is a more important source of error for the photometric SFR estimation of galaxies, although the other one is also very important. Finally, it becomes even more difficult to estimate the photometric SFR correctly by only using the seven-band photometries from the CSST-like imaging survey.

# 6. Discussion

By comparing the results of performance tests for simultaneous photometric redshift and stellar parameter estimation using an empirical statistics–based mock galaxy sample (Section 4) and a hydrodynamical simulation–based mock galaxy sample (Section 5), especially those presented in Figures 8–10, it is clear that the simple typical assumptions about the SFH and DAL of galaxies have severe impact on the performance of photometric parameter estimation of galaxies for a CSST-like imaging survey. It is not very surprising, since the SFHs and MEHs of galaxies in cosmic hydrodynamical simulations, such as Horizon-AGN (Volonteri et al. 2016; Beckmann et al. 2017; Kaviraj et al. 2017), are much more complex and diverse (see also Iyer et al. 2020) than the simple assumptions that have been employed in the previous Bayesian analysis of photometric mock data.

In this section, we will discuss the effects of more flexible forms of SFH and DAL on the performance of simultaneous photometric redshift and stellar population parameter estimation of galaxies. As in Han & Han (2012, 2014, 2019) (see also Dries et al. 2016, 2018; Salmon et al. 2016; Lawler & Acquaviva 2021), we mainly employ the Bayesian model comparison method to compare six different combinations of

Figure 8. The results of photometric redshift estimation with (right panels) and without (left panels) noise, for the analysis of the empirical statistics–based (top panels) and hydrodynamical simulation–based (bottom panels) mock galaxy samples. The error from imperfect SED modeling will only present for the analysis of the hydrodynamical simulation–based mock galaxy sample. The photometric redshifts (zphot) are estimated by employing the τ model of SFH without consideration of metallicity evolution and the Calzetti et al. (2000) model of DAL. The red solid line indicates the identity while the red dashed lines indicate the outlier limits, i.e., ∣*zz z* phot true true - +> ∣( ) 1 0.15. Here, we show the results obtained with the MultiNest runtime parameters efr and nlive equal to 0.1 and 50, respectively. In general, the observational noise is the more important source of error for the photometric redshift estimation of galaxies, though the contribution from imperfect SED modeling is also very important.

these model ingredients with increasing complexity (see Table 1 for details). In addition to a CSST-like survey (Section 6.1) where only the photometries from seven broad bands are available, we discuss the results obtained by using mock data for CSST+Euclid-like (Section 6.2) and COSMOS-like surveys (Section 6.3) with increasing discriminative power.

Figure 9. Same as Figure 8, but for the stellar mass. In general, the observational noise is the more important source of error for the photometric stellar mass estimation of galaxies, and the contribution from imperfect SED modeling is almost comparable.

# 6.1. Effects of More Flexible SFH and DAL for a CSST-like Survey

In Table 3, we present a summary of the Bayesian evidence, maximum likelihoods, and metrics of the quality of parameter estimation from the Bayesian analysis of the hydrodynamical simulation–based mock galaxy sample for a CSST-like survey employing six different combinations of SFH and DAL with increasing complexity, as well as for cases with and without noise. The same results are also shown more clearly in Figure 11.

Figure 10. Same as Figure 8, but for the SFR. In general, the imperfect SED modeling is the more important source of error for the photometric SFR estimation of galaxies, though the contribution from observational noise is also very important. In particular, in the case without noise, it is clear that the simple τ model of SFH and the Calzetti et al. (2000) form of DAL lead to a severely biased estimation of SFR for the Horizon-AGN simulation–based mock sample of galaxies.

#### 6.1.1. Model Comparison

In the case without noise, as shown in the top left panel of Figure 11, the simplest model "SFH = τ, –CEH, DAL = Cal +18" has the lowest Bayesian evidence of ln BE ( ) = -88,042 5936 . With the additional consideration of

metallicity evolution, the Bayesian evidence of the model "SFH = τ, +CEH, DAL = Cal+18" increases to ln BE 74,128 597 ( ) =- 3. Then, with the adoption of the DAL of Salim et al. (2018), the Bayesian evidence of the model "SFH = τ, +CEH, DAL = Sal+18" increases significantly to ln BE 58,878 5823 ( ) = . Apparently, the

Table 3 A Summary of the Quality of Parameter Estimation for the CSST-like Survey

| Survey | Noise | SFH | DAL | ln(BE) | ln(ML) | Parameter |  | NMAD | BIA | OLF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CSST | – | τ, –CEH | Cal+00 | −88,042 ± 5936 | 127,197 | zphot |  | 0.024 0.103 | 0.011 0.003 | 0.003 0.141 |
|  |  |  |  |  |  | * log ( ) M phot |  |  |  |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.140 | −0.117 | 0.436 |
| CSST | – | τ, +CEH | Cal+00 | −74,128 ± 5973 | 144,380 | zphot * log ( ) M phot |  | 0.023 0.092 | 0.010 −0.010 | 0.002 0.141 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.148 | −0.141 | 0.491 |
| CSST | – | τ, +CEH | Sal+18 | 58,878 ± 5823 | 282,267 | zphot |  | 0.016 | 0.014 | 0.004 0.170 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.086 | 0.041 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.153 | −0.019 | 0.390 |
| CSST | – | β-τ, +CEH | Sal+18 | 56,343 ± 5780 |  | zphot |  | 0.015 | 0.014 | 0.004 0.150 |
|  |  |  |  |  | 283,410 | * log ( ) M phot |  | 0.079 | 0.041 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.133 | 0 | 0.370 |
| CSST | – | β-τ-r, +CEH | Sal+18 | 34,213 ± 5867 | 275,861 | zphot |  | 0.017 | 0.011 | 0.006 0.227 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.097 | 0.052 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.257 | 0.045 | 0.579 |
| CSST | – | α-β-τ-r, +CEH | Sal+18 | 33,044 ± 5800 | 275,886 | zphot * log ( ) M phot |  | 0.016 0.098 | 0.010 0.051 | 0.005 0.239 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.214 | 0.065 | 0.544 |
| CSST | + | τ, –CEH | Cal+00 | −18,871 ± 2649 | 49,807 | zphot * log ( ) M phot |  | 0.097 0.135 | 0.003 −0.034 | 0.264 0.341 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.157 | 0.011 | 0.371 |
| CSST | + | τ, +CEH | Cal+00 | −18,661 ± 2649 | 49,469 | zphot |  | 0.096 | 0.005 | 0.264 0.344 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.135 | −0.034 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.155 | 0 | 0.366 |
| CSST | + | τ, +CEH | Sal+18 | −26,098 ± 2956 | 54,493 | zphot * log ( ) M phot |  | 0.098 0.158 | 0.012 −0.023 | 0.263 0.387 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.201 | 0 | 0.468 |
| CSST | + | β-τ, +CEH | Sal+18 | −28,141 ± 2904 | 54,484 | zphot * log ( ) M phot |  | 0.099 0.161 | 0.011 −0.029 | 0.263 0.396 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.194 | 0.028 | 0.472 |
|  |  |  |  |  |  | zphot |  | 0.100 | 0.013 | 0.268 |
| CSST | + | β-τ-r, +CEH | Sal+18 | −35,048 ± 3020 | 54,444 | * log ( ) M phot |  | 0.171 | −0.037 | 0.415 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.232 | 0.041 | 0.544 |
| CSST | + | α-β-τ-r, +CEH | Sal+18 | −33,983 ± 2869 | 54,440 | zphot * log ( ) M phot |  | 0.104 0.177 | 0.007 −0.052 | 0.290 0.430 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.224 | 0.080 | 0.570 |

Note. A summary of the Bayesian evidence (BE), maximum likelihoods (ML), and metrics of the quality of parameter estimation from the Bayesian analysis of the hydrodynamical simulation–based mock galaxy sample for the CSST-like survey employing six SED models with increasing complexity in the form of SFH and DAL, as well as for cases with and without noise.

DAL of Salim et al. (2018) is a much better choice than that of Calzetti et al. (2000) for the hydrodynamical simulation–based mock galaxy sample in the CSST-like survey. Furthermore, by employing a more complicated β-τ form of SFH, the Bayesian evidence of the model "SFH = β-τ, +CEH, DAL = Sal+18" decreases a little to ln BE 56,343 5780 ( ) = . Actually, the latter two SED models ("SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal+18") have the largest Bayesian evidence, which are comparable within error bars. They are neither the simplest nor the most complex models.

With a quenching (or rejuvenation) component added to the SFH, the Bayesian evidence of the model "SFH = β-τ-r, +CEH, DAL = Sal+18" obviously decreases to ln BE 34,213 5867 ( ) = . It seems that, while rejuvenation or rapid quenching events may happen in some galaxies, this additional component of SFH is not very effective for most of the galaxies in the sample. Finally, by employing an even more flexible double-power-law form of SFH, the Bayesian evidence of the model "SFH = α-β-τ-r, +CEH, DAL = Sal+18" decreases a little to ln BE 33,044 5800 ( ) = . However, the

Figure 11. The Bayesian evidence (BE), maximum likelihood (ML), and metrics of photometric redshift (red cross), stellar mass (blue square), and SFR (green circle) estimation from the Bayesian analysis of the hydrodynamical simulation–based mock galaxy sample for a CSST-like survey employing six SED models (0: SFH = τ, –CEH, DAL = Cal+18; 1: SFH = τ, +CEH, DAL = Cal+18; 2: SFH = τ, +CEH, DAL = Sal+18; 3: SFH = β-τ, +CEH, DAL = Sal+18; 4: SFH = β-τ-r, +CEH, DAL = Sal+18; 5: SFH = α-β-τ-r, +CEH, DAL = Sal+18) with increasing complexity in the forms of SFH and DAL, as well as for cases with (bottom panels) and without noise (top panels). In the case without noise, the simplest model "SFH = τ, –CEH, DAL = Cal+18" has the lowest Bayesian evidence of ln BE 88,042 5936 ( ) =- . Meanwhile, the SED models "SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal+18," which are neither the simplest nor the most complex models, have the largest Bayesian evidences of ln BE 58,878 5823 ( ) = and ln BE 56,343 5780, ( ) = which are comparable within error bars. Interestingly, the same two models give the highest-quality parameter estimates. It is worth mentioning that the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to similar results. In contrast to the case without noise, in the case with noise, the two simplest SED models "SFH = τ, –CEH, DAL = Cal+00" and "SFH = τ, +CEH, DAL = Cal+00" have the largest Bayesian evidences of ln BE 18,871 2649 ( ) =- and ln BE 18,661 2649 ( ) =- , which are comparable within error bars. It is very interesting that the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to exactly the opposite results, where the more complex (or flexible) models are always more favored. Similar to the case without noise, the two simplest SED models also give the highest-quality parameter estimates. For photometric redshift, stellar mass, and SFR, accurate estimation becomes increasingly difficult, with the latter two being more sensitive to the selection of SED models. Generally, the quality of parameter estimates is closely related to the level of Bayesian evidence, which is especially clear in the more realistic case with noise. Actually, the quality of parameter estimation, especially that of stellar mass and SFR estimation, significantly decreases with the increase of SED model complexity, which is similar to the case with perfect SED modeling as shown in Figures 4–6, and should be caused by more severe parameter degeneracies suffered by the more flexible SED model. It is clear that, in the more realistic case with noise, the model selection with the maximum likelihood (or equivalently the minimum χ2 ) is not consistent with the measurements of the quality of parameter estimation. Since direct measurements of metrics such as NMAD, BIA, and OLF are usually unavailable, Bayesian model comparison with Bayesian evidence can be used to find the best SED model that is not only the most efficient but also gives the best parameter estimation.

latter two SED models are actually comparable within error bars. Further, it is worth mentioning that the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to similar results.

In the case with noise, as shown in the bottom left panel of Figure 11, the two simplest SED models "SFH = τ, –CEH, DAL = Cal+00" and "SFH = τ, +CEH, DAL = Cal+00" have the largest Bayesian evidences of ln BE 18,871 2649 ( ) = and ln BE 18,661 2649 ( ) =- , respectively. Although the latter, which additionally considers the metallicity evolution, seems better, the Bayesian evidence is actually comparable within error bars. Then, with the adoption of the DAL of Salim et al. (2018), the Bayesian evidence of the model "SFH = τ, +CEH, DAL = Sal+18" decreases significantly to ln BE 26,098 2956 ( ) =- , exactly the opposite of the situation without noise. It is likely that the more complicated form of DAL does not give a better fit to the noisier data. By employing a more complicated β-τ form of SFH, the Bayesian evidence of the model "SFH = β-τ, +CEH, DAL = Sal+18" decreases further to ln BE 28,141 2904 ( ) =- , although it is comparable with the former within error bars. With a quenching (or rejuvenation) component added to the SFH, the Bayesian evidence of the model "SFH = β-τ-r, +CEH, DAL = Sal+18" obviously decreases to ln BE ( )= -35,048 3020 , which is similar to the case without noise. Finally, by employing an even more flexible double-power-law form of SFH, the Bayesian evidence of the model "SFH = α-βτ-r, +CEH, DAL = Sal+18" increases a little to ln BE 33,983 2869 ( ) =- , although it is comparable with the former within error bars. It is very interesting that the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to exactly the opposite results, where the more complex (or flexible) models are always more favored.

#### 6.1.2. Parameter Estimation

In the three rightmost panels of Figure 11, we show the three metrics of the quality of photometric redshift, stellar mass, and SFR estimation for different SED models. In general, in the case without noise, the SED models "SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal+18," which are just the two with the largest Bayesian evidence, give the highest-quality parameter estimates. In the case with noise, the two simplest SED models "SFH = τ, –CEH, DAL = Cal +00" and "SFH = τ, +CEH, DAL = Cal+00," which are also the two with the largest Bayesian evidence, give the highestquality parameter estimates. In the following, we discuss these results in more detail.

The detailed results of photometric redshift estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 12. In the case without noise, by comparing the results in Figure 8(c) with those in Figure 12(a), it becomes clear that, with the additional consideration of metallicity evolution and the adoption of the DAL of Salim et al. (2018), the σNMAD of photometric redshift estimation is obviously reduced while the bias and OLF are only slightly increased. Meanwhile, the systematic patterns in the former results are also largely reduced. However, as shown in Figure 12(c), by additionally employing a more complicated β-τ form of SFH, the σNMAD of photometric redshift estimation is only slightly reduced while the bias and OLF are exactly the same. Besides, as shown in Table 3 and Figure 11, the other two even more complicated forms of SFH lead to a similar quality of photometric redshift estimation. In the case with noise, the best two models are quite similar in their quality of photometric redshift estimation. With the increase of complexity of the SED models, the quality of photometric redshift estimation tends to decrease, although not very significantly.

The detailed results of photometric stellar mass estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 13. In the case without noise, by comparing the results in Figure 9(c) with those in Figure 13(a), it becomes clear that, with the additional consideration of metallicity evolution and the adoption of the DAL of Salim et al. (2018), the σNMAD of photometric stellar mass estimation is reduced, although the bias and OLF are somewhat increased. By additionally employing a more complicated β-τ form of SFH, as shown in Figure 13(c), the quality of photometric stellar mass estimation increases further. However, as shown in Table 3 and Figure 11, with the increase of complexity of the SED models, the quality of photometric stellar mass estimation decreases obviously. In the case with noise, the best two models are exactly the same in terms of the quality of photometric stellar mass estimation. Meanwhile, with the increase of complexity of the SED models, the quality of photometric stellar mass estimation decreases even more obviously.

The detailed results of photometric SFR estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 14. In the case without noise, by comparing the results in Figure 10(c) with those in Figure 14(a), it becomes clear that, with the additional consideration of metallicity evolution and the adoption of the DAL of Salim et al. (2018), the systematic bias and OLF of photometric SFR estimation are largely reduced while the σNMAD is slightly increased. By additionally employing a more complicated β-τ form of SFH, as shown in Figure 14(c), the quality of

photometric SFR estimation increases to the best level. However, as shown in Table 3 and Figure 11, with an additional quenching (or rejuvenation) component, the β-τ-r form of SFH leads to a much worse quality of photometric SFR estimation. Finally, the most complicated α-β-τ-r form of SFH leads to a slightly better SFR estimation. In the case with noise, the best two models are also very similar in their quality of photometric SFR estimation. Meanwhile, with the increase of complexity of the SED models, the quality of photometric SFR estimation increases significantly.

For photometric redshift, stellar mass, and SFR, accurate estimation becomes increasingly difficult. Besides, the latter two are more sensitive to the selection of SED models. Generally, the quality of parameter estimates is closely related to the level of Bayesian evidence, which is especially clear in the more realistic case with noise. Meanwhile, the model selection with the maximum likelihood (or equivalently the minimum χ2 ), where the more complex (or flexible) models are always more favored, is not consistent with the measurements of the quality of parameter estimation. In practice, direct measurements of the quality of parameter estimation as indicated by NMAD, BIA, and OLF are usually unavailable. So, Bayesian model comparison with Bayesian evidence can be used to find the best SED model that is not only the most efficient but also gives the best parameter estimation.

# 6.2. Effects of More Flexible SFH and DAL for a CSST +Euclid-like Survey

The results of both model comparison and parameter estimation are strongly dependent on the used data sets, which may have very different discriminative powers. In Figure 15, we show an example of 1D and 2D posterior probability distribution functions (PDFs) of free parameters obtained from the Bayesian analysis of the photometric data of a mock galaxy in the CSST-like, CSST+Euclid-like, and COSMOS-like surveys. It is clear that different data sets lead to very different PDFs, due to their very different discriminative powers. In this section and in Section 6.3, we discuss the effects of more flexible SFHs and DALs for the CSST+Euclid-like and COSMOS-like surveys, respectively. The addition of Euclid data extends the wavelength coverage of the data to the longer near-IR band as compared with the CSST-only data, which should be useful for enhancing the discriminative power of model comparison and the quality of the parameter estimation.

In Table 4, we present a summary of the Bayesian evidence, maximum likelihoods, and metrics of the quality of parameter estimation from the Bayesian analysis of the hydrodynamical simulation–based mock galaxy sample for the CSST+Euclidlike survey employing six different combinations of SFH and DAL with increasing complexity, as well as for cases with and without noise. The same results are also shown more clearly in Figure 16.

#### 6.2.1. Model Comparison

In the case without noise, as shown in the top left panel of Figure 16, the simplest model "SFH = τ, –CEH, DAL = Cal +18" has the lowest Bayesian evidence of ln BE ( ) = -243,527 6307 . With the additional consideration of metallicity evolution, the Bayesian evidence of the model "SFH = τ, +CEH, DAL = Cal+18" increases to ln BE ( ) = -216,613 6390 . Then, with the adoption of the DAL of

Figure 12. The results of photometric redshift estimation from the Bayesian analysis of the hydrodynamical simulation–based mock data for the CSST-like imaging survey employing the two SED models with the largest Bayesian evidence. (a) By comparing with the results in Figure 8(c), it becomes clear that, with the additional consideration of metallicity evolution and the adoption of the DAL of Salim et al. (2018), the σNMAD of photometric redshift estimation is obviously reduced, although the bias and OLF are slightly increased. Meanwhile, the systematic patterns in the former results are also largely reduced. (c) By additionally employing a more complicated β-τ form of SFH, the σNMAD of photometric redshift estimation is only slightly reduced while the bias and OLF are exactly the same. (b), (d) In the case with noise, the best two models are quite similar in their quality of photometric redshift estimation. Besides, there are two clear branches of outliers caused by the misidentification of Lyman and Balmer break features.

Figure 13. Same as Figure 12, but for the stellar mass. (a) By comparing with the results in Figure 9(c), it becomes clear that, with the additional consideration of metallicity evolution and the adoption of the DAL of Salim et al. (2018), the σNMAD of photometric stellar mass estimation is reduced, although the bias and OLF are somewhat increased. (c) By additionally employing a more complicated β-τ form of SFH, the quality of photometric stellar mass estimation increases further. (b), (d) In the case with noise, the best two models are exactly the same in terms of the quality of photometric stellar mass estimation.

Salim et al. (2018), the Bayesian evidence of the model "SFH = τ, +CEH, DAL = Sal+18" increases significantly to ln BE 72,092 6475 ( ) = . Apparently, the DAL of Salim et al. (2018) is also a much better choice than that of Calzetti et al. (2000) for the hydrodynamical simulation–based mock galaxy sample in the CSST+Euclid-like survey. Furthermore, by employing a more complicated β-τ form of SFH, the Bayesian evidence of the model "SFH = β-τ, +CEH, DAL = Sal+18"

Figure 14. Same as Figure 12, but for the SFR. (a) By comparing with the results in Figure 10(c), it becomes clear that, with the additional consideration of metallicity evolution and the adoption of the DAL of Salim et al. (2018), the systematic bias and OLF of photometric SFR estimation are largely reduced, although the σNMAD is slightly increased. (c) By additionally employing a more complicated β-τ form of SFH the quality of photometric SFR estimation increases to the best level. (b), (d) In the case with noise, the best two models are also very similar in their quality of photometric SFR estimation.

decreases a little to ln BE 70,340 6409 ( ) = . As in the case of the CSST-like survey, the latter two SED models ("SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal +18") have the largest Bayesian evidence, which is comparable within error bars. With a quenching (or rejuvenation) component

added to the SFH, the Bayesian evidence of the model "SFH = β-τ-r, +CEH, DAL = Sal+18" decreases significantly to ln BE 25,728 6566 ( ) = . Finally, by employing an even more flexible double-power-law form of SFH, the Bayesian evidence of the model "SFH = α-β-τ-r, +CEH, DAL = Sal+18" decreases a

Figure 15. An example of 1D and 2D posterior PDFs of free parameters obtained for the Bayesian analysis of a mock galaxy with CSST-like (gray), CSST+Euclidlike (red), and COSMOS-like (blue) photometric data. The contours show the 1σ, 2σ, and 3σ confidence regions, while the red dashed lines show the ground truth values of each parameter. It is clear that the parameters are more tightly constrained and some degeneracies between them are broken when using data sets with increasing discriminative powers.

little to ln BE 19,560 6454 ( ) = , which is comparable with the former within error bars. The model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to similar results.

In the case with noise, the two simplest SED models "SFH = τ, –CEH, DAL = Cal+00" and "SFH = τ, +CEH, DAL = Cal+00" have the largest Bayesian evidences of ln BE 31,783 3314 ( ) =- and ln BE 31,347 330 ( ) =- 2, respectively. Then, with the adoption of the DAL of Salim et al. (2018), the Bayesian evidence of the model "SFH = τ, +CEH,

DAL = Sal+18" decreases significantly to ln BE ( ) = -38,081 3529 . By employing a more complicated β-τ form of SFH, the Bayesian evidence of the model "SFH = β-τ, +CEH, DAL = Sal+18" decreases further to ln BE ( ) = -41,164 3437 , although it is comparable with the former within error bars. With a quenching (or rejuvenation) component added to the SFH, the Bayesian evidence of the model "SFH = β-τ-r, +CEH, DAL = Sal+18" obviously decreases to ln BE 49,796 356 ( ) =- 1, which is similar to the case without noise. Finally, by employing an even more

Table 4 Same as Table 3, but for the CSST+Euclid-like Survey

| Survey | Noise | SFH | DAL | ln(BE) | ln(ML) | Parameter |  | NMAD | BIA | OLF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CSST+Euclid | – | τ, –CEH | Cal+00 | −243,527 ± 6307 | −3717 | zphot |  | 0.020 | 0.002 | 0.001 0.045 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.055 | −0.023 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.097 | −0.088 | 0.299 |
| CSST+Euclid | – | τ, +CEH | Cal+00 | −216,613 ± 6390 | 30,195 | zphot |  | 0.021 | 0.001 −0.014 | 0.001 0.037 |
|  |  |  |  |  |  | * log ( ) M phot log SFR ( [ yr M 1 | - ])phot | 0.057 0.094 | −0.108 | 0.342 |
| CSST+Euclid | – | τ, +CEH | Sal+18 | 72,092 ± 6475 | 340,038 | zphot |  | 0.013 0.029 | 0.010 −0.015 | 0.005 0.009 |
|  |  |  |  |  |  | * log ( ) M phot log SFR ( [ yr M 1 | - ])phot | 0.081 | −0.046 | 0.218 |
| CSST+Euclid | – | β-τ, +CEH | Sal+18 | 70,340 ± 6409 | 339,916 | zphot |  | 0.012 | 0.010 | 0.004 0.007 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.027 | −0.010 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.065 | −0.021 | 0.195 |
| CSST+Euclid | – | β-τ-r, +CEH | Sal+18 | 25,728 ± 6566 | 315,401 | zphot |  | 0.017 | 0.012 | 0.010 0.017 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.032 | −0.005 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.087 | 0.006 | 0.302 |
| CSST+Euclid | – | α-β-τ-r, +CEH | Sal+18 | 19,560 ± 6454 | 305,020 | zphot * log ( ) M phot |  | 0.016 0.035 | 0.012 −0.007 | 0.016 0.030 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.087 | 0.011 | 0.286 |
| CSST+Euclid | + | τ, –CEH | Cal+00 | −31,783 ± 3314 | 58,861 | zphot * log ( ) M phot |  | 0.065 0.058 | −0.003 −0.030 | 0.119 0.083 |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.164 | −0.020 | 0.379 |
| CSST+Euclid | + | τ, +CEH | Cal+00 | −31,347 ± 3302 | 58,766 | zphot |  | 0.064 | −0.001 | 0.118 0.080 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.058 | −0.027 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.171 | −0.027 | 0.392 |
| CSST+Euclid | + | τ, +CEH | Sal+18 | −38,081 ± 3529 | 64,421 | zphot |  | 0.067 | 0.016 | 0.129 0.103 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.061 | −0.016 |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.186 | 0.011 | 0.449 |
| CSST+Euclid | + | β-τ, +CEH | Sal+18 | −41,164 ± 3437 | 64,349 | zphot |  | 0.068 | 0.015 | 0.130 0.112 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.063 | −0.017 |  |
|  |  |  |  |  |  | yr M 1 log SFR ( [ | - ])phot | 0.177 | 0.042 | 0.476 |
| CSST+Euclid | + | β-τ-r, +CEH | Sal+18 | −49,796 ± 3561 | 63,873 | zphot |  | 0.069 | 0.017 | 0.141 0.127 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.065 | −0.023 |  |
|  |  |  |  |  |  | yr M 1 log SFR ( [ | - ])phot | 0.213 | 0.059 | 0.571 |
| CSST+Euclid | + | α-β-τ-r, +CEH | Sal+18 | −49,608 ± 3426 | 64,065 | zphot |  | 0.068 0.073 | 0.016 −0.025 | 0.136 0.169 |
|  |  |  |  |  |  | * log ( ) M phot |  |  |  |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | - ])phot | 0.181 | 0.084 | 0.560 |

flexible double-power-law form of SFH, the Bayesian evidence of the model "SFH = α-β-τ-r, +CEH, DAL = Sal+18" increases a little to ln BE 49,608 3426 ( ) =- , although it is comparable with the former within error bars. However, the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to exactly the opposite results, where the more complex (or flexible) models are always more favored.

In general, all of these results are very similar to those for the CSST-like survey. The model selection with Bayesian evidence is still more consistent with measurements of the quality of parameter estimation. However, the relative errors of the Bayesian evidence is reduced, especially in the case without noise.

#### 6.2.2. Parameter Estimation

In the three rightmost panels of Figure 16, we show the three metrics of the quality of photometric redshift, stellar mass, and SFR estimation for different SED models. In general, in the case without noise, the SED models "SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal+18," which are just the two with the largest Bayesian evidence, give the highest-quality parameter estimates. In the case with noise, the two simplest SED models "SFH = τ, –CEH, DAL = Cal +00" and "SFH = τ, +CEH, DAL = Cal+00," which are also the two with the largest Bayesian evidence, give the highestquality parameter estimates. In the following, we discuss these results in more detail.

Figure 16. The Bayesian evidence (BE), maximum likelihood (ML), and metrics of photometric redshift (red cross), stellar mass (blue square), and SFR (green circle) estimation from the Bayesian analysis of the hydrodynamical simulation–based mock galaxy sample for the CSST+Euclid-like survey employing six SED models (0: SFH = τ, –CEH, DAL = Cal+18; 1: SFH = τ, +CEH, DAL = Cal+18; 2: SFH = τ, +CEH, DAL = Sal+18; 3: SFH = β-τ, +CEH, DAL = Sal+18; 4: SFH = β-τr, +CEH, DAL = Sal+18; 5: SFH = α-β-τ-r, +CEH, DAL = Sal+18) with increasing complexity in the forms of SFH and DAL, as well as for cases with (bottom panels) and without noise (top panels). In the case without noise, the simplest model "SFH = τ, –CEH, DAL = Cal+18" has the lowest Bayesian evidence of ln BE 243,527 6307 ( ) =- . Meanwhile, the SED models "SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal+18," which are neither the simplest nor the most complex models, have the largest Bayesian evidences of ln BE 72,092 6475 ( ) = and ln BE 70,340 6409 ( ) = , which are comparable within error bars. As in the case of the CSST-like survey, the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to similar results. In contrast to the case without noise, in the case with noise, the two simplest SED models "SFH = τ, –CEH, DAL = Cal+00" and "SFH = τ, +CEH, DAL = Cal+00" have the largest Bayesian evidences of ln BE 31,783 3314 ( ) =- and ln BE 31,347 330 ( ) =- 2, which are comparable within error bars. However, the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to exactly the opposite results, where the more complex (or flexible) models are always more favored. Generally, the quality of parameter estimates is closely related to the level of Bayesian evidence, which is especially clear in the more realistic case with noise. The model selection with Bayesian evidence is still more consistent with the measurements of the quality of parameter estimation than that with the maximum likelihood (or equivalently the minimum χ2 ). For photometric redshift, stellar mass, and SFR, accurate estimation becomes increasingly difficult, with the latter two being more sensitive to the selection of SED models. In the case without noise, the SED models "SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal+18," which are just the two with the largest Bayesian evidence, give the highest-quality parameter estimates. In the case with noise, the two simplest SED models "SFH = τ, –CEH, DAL = Cal+00" and "SFH = τ, +CEH, DAL = Cal+00," which are also the two with the largest Bayesian evidence, give the highestquality parameter estimates. All of these results are very similar to those for the CSST-like survey. However, the relative errors of the Bayesian evidence is reduced, especially in the case without noise. Besides, the quality of parameter estimation, especially that of stellar mass estimation, is significantly improved. Furthermore, the quality of parameter estimation, especially that of stellar mass estimation, increases more slowly with the increase of SED model complexity.

The detailed results of photometric redshift estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 17. By comparing with the results for the CSST-like survey in Figure 12, it becomes clear that the quality of photometric redshift estimation is obviously increased in both cases with and without noise. In particular, in the more realistic case with noise, the outliers caused by the misidentification of Lyman and Balmer break features are largely reduced. This suggests that the inclusion of the J, H, and Y bands from Euclid is helpful for improving the photometric redshift estimation. However, the more complicated β-τ form of SFH is not very helpful for improving the quality of photometric redshift estimation.

The detailed results of photometric stellar mass estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 18. By comparing with the results for the CSST-like survey in Figure 13, it becomes clear that the quality of photometric stellar mass estimation is significantly improved in both cases with and without noise. Apparently, the inclusion of the J, H, and Y bands from Euclid is crucial for a more accurate estimation of stellar mass. With the more complicated β-τ form of SFH, the quality of photometric stellar mass estimation is slightly improved. However, as shown in Table 4 and Figure 16, the even more complicated forms of SFH are still not helpful for improving the quality of photometric stellar mass estimation.

The detailed results of photometric SFR estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 19. In the case without noise, by comparing with the results for the CSST-like survey in Figure 14, it becomes clear that the σNMAD and OLF of photometric SFR estimation are largely reduced, although the bias is slightly increased. However, in the case with noise, all of the σNMAD, bias, and OLF of photometric SFR estimation slightly

Figure 17. The results of photometric redshift estimation from the Bayesian analysis of the hydrodynamical simulation–based mock data for the CSST+Euclid-like survey employing the two SED models with the largest Bayesian evidence. By comparing with the results for the CSST-like survey in Figure 12, it becomes clear that the quality of photometric redshift estimation is obviously increased in both cases with and without noise. In particular, in the more realistic case with noise, the outliers caused by the misidentification of Lyman and Balmer break features are largely reduced. This suggests that the inclusion of the J, H, and Y bands from Euclid is helpful for improving the photometric redshift estimation. However, the more complicated β-τ form of SFH is not very helpful for improving the quality of photometric redshift estimation.

Figure 18. Same as Figure 17, but for the stellar mass. By comparing with the results for the CSST-like survey in Figure 13, it becomes clear that the quality of photometric stellar mass estimation is significantly improved in both cases with and without noise. Apparently, the inclusion of the J, H, and Y bands from Euclid is crucial for a more accurate estimation of stellar mass. Besides, with the more complicated β-τ form of SFH, the quality of photometric stellar mass estimation is only slightly improved.

Figure 19. Same as Figure 17, but for the SFR. (a), (c) The cases without noise. By comparing with the results for the CSST-like survey in the left panels of Figure 14, it becomes clear that the σNMAD and OLF of photometric SFR estimation are largely reduced, although the bias is slightly increased. Besides, with the more complicated β-τ form of SFH, the quality of photometric SFR estimation is improved. (b), (d) The cases with noise. By comparing with the results for the CSST-like survey in the right panels of Figure 14, all of the σNMAD, bias, and OLF of photometric SFR estimation slightly increase. This suggests that the inclusion of the J, H, and Y bands from Euclid is not very helpful for improving the photometric SFR estimation. Besides, with the additional consideration of metallicity evolution, the quality of photometric SFR estimation becomes slightly worse.

| Table 5 |
| --- |
| Same as Table 3, but for the COSMOS-like Survey |

| Survey | Noise | SFH | DAL | ln(BE) | ln(ML) | Parameter |  | NMAD | BIA | OLF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COSMOS | – | τ, –CEH |  |  |  | zphot |  | 0.007 | −0.002 | 0 |
|  |  |  | Cal+00 | −1,710,604 ± 6811 | −1,431,790 | * log ( ) M phot |  | 0.030 | −0.026 | 0.013 |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.110 | −0.062 | 0.290 |
|  |  |  |  |  |  | zphot |  | 0.006 | −0.002 | 0 |
| COSMOS | – | τ, +CEH | Cal+00 | −1,499,618 ± 6997 | −1,207,887 | * log ( ) M phot |  | 0.032 | −0.017 | 0.010 |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.101 | −0.082 | 0.299 |
| COSMOS | – | τ, +CEH | Sal+18 | 354,581 ± 7310 | 681,554 | zphot |  | 0.002 | 0 | 0 0.012 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.023 | −0.022 |  |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.069 | −0.068 | 0.124 |
| COSMOS | – | β-τ, +CEH | Sal+18 | 387,009 ± 7273 | 717,135 | zphot |  | 0.002 | 0 | 0 0.006 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.020 | −0.011 |  |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.047 | −0.041 | 0.057 |
| COSMOS | – | β-τ-r, +CEH | Sal+18 | 255,069 ± 7552 | 613,256 | zphot |  | 0.003 0.028 | 0.001 −0.003 | 0 0.017 |
|  |  |  |  |  |  | * log ( ) M phot - |  |  |  |  |
|  |  |  |  |  |  | log SFR ( [ yr M 1 | ])phot | 0.070 | −0.001 | 0.124 |
| COSMOS | – | α-β-τ-r, +CEH | Sal+18 | 275,129 ± 7218 | 612,401 | zphot * log ( ) M phot |  | 0.003 0.031 | 0.001 −0.005 | 0 0.012 |
|  |  |  |  |  |  | - yr M 1 log SFR ( [ | ])phot | 0.067 | −0.006 | 0.121 |
|  |  | τ, –CEH |  |  |  | zphot |  | 0.017 | −0.003 | 0.004 |
| COSMOS | + |  | Cal+00 | 99,881 ± 4881 | 259,510 | * log ( ) M phot |  | 0.044 | −0.020 | 0.006 |
|  |  |  |  |  |  | - yr M 1 log SFR ( [ | ])phot | 0.127 | −0.032 | 0.282 |
| COSMOS | + | τ, +CEH | Cal+00 | 107,009 ± 4896 | 268,492 | zphot |  | 0.017 | −0.004 | 0.003 0.005 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.044 | −0.012 |  |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.126 | −0.044 | 0.288 |
| COSMOS | + | τ, +CEH | Sal+18 | 131,578 ± 5115 | 313,110 | zphot |  | 0.015 | 0.002 | 0.003 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.042 | −0.017 | 0.036 |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.107 | −0.021 | 0.211 |
| COSMOS | + | β-τ, +CEH |  |  |  | zphot |  | 0.015 | 0.002 | 0.003 |
|  |  |  | Sal+18 | 130,766 ± 5064 | 315,205 | * log ( ) M phot |  | 0.037 | −0.012 | 0.029 |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.090 | 0.002 | 0.176 |
|  |  |  |  |  |  | zphot |  | 0.016 | 0.006 | 0.004 |
| COSMOS | + | β-τ-r, +CEH | Sal+18 | 109,572 ± 5116 | 305,267 | * log ( ) M phot |  | 0.044 | −0.021 | 0.052 |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.102 | 0.063 | 0.321 |
| COSMOS | + | α-β-τ-r, +CEH | Sal+18 | 110,909 ± 5006 | 306,387 | zphot |  | 0.015 | 0.005 | 0.003 0.066 |
|  |  |  |  |  |  | * log ( ) M phot |  | 0.045 | −0.020 |  |
|  |  |  |  |  |  | - log SFR ( [ yr M 1 | ])phot | 0.097 | 0.055 | 0.304 |

increase. So, the inclusion of the J, H, and Y bands from Euclid is not very helpful for improving the photometric SFR estimation.

# 6.3. Effects of More Flexible SFH and DAL for a COSMOS-like Survey

The COSMOS-like survey covers many more bands than the CSST-like data. Although there is no NUV data, the coverage extends to the longer wavelengths and includes some IBs (Laigle et al. 2016). Since the COSMOS-like mock data has much stronger discriminative power than the CSST-like and CSST+Euclid-like mock data, the Bayesian evidence of different SED models should show much larger differences, and the photometric redshift and stellar population parameter estimation should be better.

In Table 5, we present a summary of the Bayesian evidence, maximum likelihoods, and metrics of the quality of parameter estimation from the Bayesian analysis of the hydrodynamical simulation–based mock galaxy sample for the COSMOS-like survey employing six different combinations of SFH and DAL with increasing complexity, as well as for cases with and without noise. The same results are also shown more clearly in Figure 20.

#### 6.3.1. Model Comparison

In the case without noise, as shown in the top left panel of Figure 20, the simplest model "SFH = τ, –CEH, DAL = Cal

Figure 20. The Bayesian evidence (BE), maximum likelihood (ML), and metrics of photometric redshift (red cross), stellar mass (blue square), and SFR (green circle) estimation from the Bayesian analysis of the hydrodynamical simulation–based mock galaxy sample for the COSMOS-like survey employing six SED models (0: SFH = τ, –CEH, DAL = Cal+18; 1: SFH = τ, +CEH, DAL = Cal+18; 2: SFH = τ, +CEH, DAL = Sal+18; 3: SFH = β-τ, +CEH, DAL = Sal+18; 4: SFH = β-τ-r, +CEH, DAL = Sal+18; 5: SFH = α-β-τ-r, +CEH, DAL = Sal+18) with increasing complexity in the forms of SFH and DAL, as well as for cases with (bottom panels) and without noise (top panels). In the case without noise, the simplest model "SFH = τ, –CEH, DAL = Cal+18" has the lowest Bayesian evidence of ln BE 1,710,604 681 ( ) =- 1. Meanwhile, the SED model "SFH = β-τ, +CEH, DAL = Sal+18," which is neither the simplest nor the most complex model, has the largest Bayesian evidence of ln BE 387,009 7273 ( ) = , and gives the highest-quality parameter estimates. As in the cases of the CSST-like and CSST +Euclid-like surveys, the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to similar results. In the case with noise, the simplest model "SFH = τ, –CEH, DAL = Cal+18" still has the lowest Bayesian evidence of ln BE 99,881 4881 ( ) = . Meanwhile, the SED models "SFH = τ, +CEH, DAL = Sal+18" and "SFH = β-τ, +CEH, DAL = Sal+18" have the largest Bayesian evidences of ln BE 131,578 5115 ( ) = and ln BE 130,766 5064, ( ) = which are comparable within error bars. Unlike the cases of the CSST-like and CSST+Euclid-like surveys, the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to similar results. The two models with the largest Bayesian evidence (or maximum likelihoods) also give the highest-quality parameter estimates. In general, in the cases with and without noise, much clearer results of model comparison are obtained. Meanwhile, in the more realistic case with noise, the more complicated SED models are more favored than in the cases of the CSST-like and CSST+Euclid-like survey. All of these are natural results of the much stronger discriminative power of the COSMOS-like survey compared to that of the CSST-like and CSST+Euclid-like surveys.

+18" has the lowest Bayesian evidence of ln BE ( ) = -1,710,604 6811 . With the additional consideration of metallicity evolution, the Bayesian evidence of the model "SFH = τ, +CEH, DAL = Cal+18" increases to ln BE ( ) = -1,499,618 6997 . Then, with the adoption of the DAL of Salim et al. (2018), the Bayesian evidence of the model "SFH = τ, +CEH, DAL = Sal+18" increases significantly to ln BE 354,581 7310 ( ) = . Apparently, the DAL of Salim et al. (2018) is also a much better choice than that of Calzetti et al. (2000) for the hydrodynamical simulation–based mock galaxy sample in the COSMOS-like survey. Furthermore, by employing a more complicated β-τ form of SFH, the Bayesian evidence of the model "SFH = β-τ, +CEH, DAL = Sal+18" increases further to ln BE 387,009 7273 ( ) = . However, with a quenching (or rejuvenation) component added to the SFH, the Bayesian evidence of the model "SFH = β-τ-r, +CEH, DAL = Sal+18" significantly decreases to ln BE ( ) = 255,069 755 2, which is similar to the case without noise. Finally, by employing an even more flexible double-power-law form of SFH, the Bayesian evidence of the model "SFH = α-βτ-r, +CEH, DAL = Sal+18" increases to ln BE ( ) = 275,129 7218 . As in the cases of the CSST-like and CSST +Euclid-like surveys, the model selection with the maximum likelihood (or equivalently the minimum χ2 ) leads to similar results.

In the case with noise, as shown in the bottom left panel of Figure 20, almost the same conclusions about the SED model comparison are obtained, although the detailed values of the Bayesian evidence are apparently different. Unlike the cases of the CSST-like and CSST+Euclid-like surveys, the model selection with Bayesian evidence and maximum likelihood (or equivalently the minimum χ2 ) leads to similar results. In general, for the COSMOS-like survey, we can obtain clearer results of SED model comparison. Meanwhile, in the more realistic case with noise, the more complicated SED models are more favored than in the cases of the CSST-like and CSST +Euclid-like surveys. This is reasonable, since the COSMOSlike survey has much stronger discriminative power than the CSST-like and CSST+Euclid-like surveys.

### 6.3.2. Parameter Estimation

In the three rightmost panels of Figure 20, we show the three metrics of the quality of photometric redshift,

Figure 21. The results of photometric redshift estimation from the Bayesian analysis of the hydrodynamical simulation–based mock data for the COSMOS-like survey employing the two SED models with the largest Bayesian evidence. By comparing with the results for the CSST-like survey in Figure 12 and those for the CSST +Euclid-like survey in Figure 17, it becomes clear that the quality of photometric redshift estimation is significantly increased in both cases with and without noise. In both cases, the best two SED models give an identical quality of photometric redshift estimation. In the case without noise, the bias and OLF of photometric redshift estimation are almost zero while the σNMAD is only 0.002. Since the errors from parameter degeneracy and SED model errors should be largely reduced, this suggests that the contribution of errors from the stochastic nature of the MultiNest sampling algorithm and other potential errors in the BayeSED code should be less than 0.002. In the more realistic case with noise, the outliers caused by the misidentification of Lyman and Balmer break features are largely resolved.

Figure 22. Same as Figure 21, but for the stellar mass. By comparing with the results for the CSST+Euclid-like survey in Figure 18, it becomes clear that the quality of photometric stellar mass estimation is improved further in both cases with and without noise. Besides, the more complicated β-τ form of SFH makes the quality of photometric stellar mass estimation a little better.

stellar mass, and SFR estimation for different SED models. In the cases with and without noise, the same SED model "SFH = β-τ, +CEH, DAL = Sal+18," which is just the one

with the largest Bayesian evidence, gives the highest-quality parameter estimates. In the following, we discuss these results in more detail.

Figure 23. Same as Figure 21, but for the SFR. By comparing with the results for the CSST+Euclid-like survey in Figure 18, it becomes clear that the quality of photometric SFR estimation is significantly improved in both cases with and without noise. Besides, the more complicated β-τ form of SFH makes the quality of photometric SFR estimation much better.

The detailed results of photometric redshift estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 21. By comparing with the results for the CSST-like survey in Figure 12 and those for the CSST+Euclid-like survey in Figure 17, it becomes clear that the quality of photometric redshift estimation is significantly increased in both cases with and without noise. In both cases, the best two SED models give an identical quality of photometric redshift estimation. In the case without noise, the bias and OLF of photometric redshift estimation are almost zero while the σNMAD is only 0.002. In this case, the errors from parameter degeneracy and SED model errors should be largely reduced. This suggests that the contribution of errors from the stochastic nature of the MultiNest sampling algorithm and other potential errors in the BayeSED code should be less than 0.002. In the more realistic case with noise, the outliers caused by the misidentification of Lyman and Balmer break features are largely resolved. Finally, as in the cases of the CSST-like and CSST+Euclid-like surveys, the more complicated SED models are not very helpful for improving the quality of photometric redshift estimation.

The detailed results of photometric stellar mass estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 22. By comparing with the results for the CSST+Euclid-like survey in Figure 18, it becomes clear that the quality of photometric stellar mass estimation is improved further in both cases with and without noise. Besides, the more complicated β-τ form of SFH makes the quality of photometric stellar mass estimation a little better. However, as shown in Table 5 and Figure 16, the even more complicated forms of SFH make the quality of photometric stellar mass estimation worse.

The detailed results of photometric SFR estimation obtained by employing the two models with the largest Bayesian evidence are shown in Figure 23. By comparing with the results for the CSST+Euclid-like survey in Figure 18, it becomes clear that the quality of photometric SFR estimation is significantly improved in both cases with and without noise. Besides, the more complicated β-τ form of SFH makes the quality of photometric SFR estimation much better. However, as shown in Figure 20, with a quenching (or rejuvenation) component added to the SFH, the quality of photometric SFR estimation becomes obviously worse. Finally, by employing an even more flexible double-power-law form of SFH, the quality of photometric SFR estimation becomes a little better.

# 7. Summary and Conclusion

In this work, based on the Bayesian SED synthesis and analysis techniques employed in the BayeSED-V3 code, we present a comprehensive and systematic test of its performance for simultaneous photometric redshift and stellar population parameter estimation of galaxies when combined with six SED models with increasing complexity in the form of SFH and DAL. The main purpose is to make a systematic analysis of various factors affecting the simultaneous photometric redshift and stellar population parameter estimation of galaxies in the context of Bayesian SED fitting, so as to provide clues for further improvement.

To separate the different factors that could contribute to the errors of photometric redshift and stellar population parameter estimation of galaxies, empirical statistics–based and hydrodynamical simulation–based approaches have been employed to generate mock photometric samples of galaxies with or without noise for CSST-like, COSMOS-like, and CSST +Euclid-like surveys. We compare the performance of photometric parameter estimation with different run parameters of the Bayesian analysis algorithm, different assumptions about the SFH and DAL of galaxies, and different observational data sets. Our main findings are as follows.

For the performance tests using an empirical statistics–based mock galaxy sample with idealized SED modeling:

- 1. The performance of photometric redshift and stellar population parameter estimation, in terms of speed and quality, is sensitive to the runtime parameters (the target sampling efficiency efr and the number of live points nlive) of the MultiNest algorithm.
- 2. A good balance among the speed, quality of parameter estimation, and accuracy of model comparison can be achieved when adopting MultiNest runtime parameters efr = 0.1 and nlive = 50.
- 3. By employing the optimized runtime parameters of MultiNest and the simplest SED modeling, a speed of ∼2 s/object/CPU (∼10 s/object/CPU) can be achieved for a detailed Bayesian analysis of photometries from a CSST-like (COSMOS-like) survey, which is sufficient for the analysis of massive photometric data. Meanwhile, a quality of photometric redshift estimation with σNMAD = 0.056, BIA = −0.0025, and OLF = 0.215; a quality of photometric stellar mass estimation with σNMAD = 0.113, BIA = −0.025, and OLF = 0.285; and a quality of photometric SFR estimation with σNMAD = 0.08, BIA = −0.01, and OLF = 0.255 can be achieved.
- 4. With the optimized runtime parameters of MultiNest, the value of Bayesian evidence that is crucial for Bayesian model comparison can also be well estimated, although the error of Bayesian evidence tends to be overestimated, which may lead to a more conservative conclusion about model comparison.
- 5. The random observational errors in photometries are more important sources of errors than the parameter degeneracies and Bayesian analysis method and tool.
- 6. More complicated SED models apparently require longer running time. They also tend to overfit noisy photometries and lead to worse quality of photometric redshift, stellar mass, and SFR estimation, which is likely due to more free parameters and more severe parameter degeneracies.
- 7. The value of Bayesian evidence clearly decreases with the increase of complexity of the SED model in both cases with and without noise.

For the performance tests using a hydrodynamical simulation–based mock galaxy sample without idealized SED modeling:

- 1. The commonly used simple assumptions about the SFH and DAL of galaxies have severe impact on the quality of photometric parameter estimation of galaxies, especially for a CSST-like survey with only photometries from seven broad bands.
- 2. The performance of both Bayesian parameter estimation and model comparison highly depends on the discriminative power of the observational photometries. With more informative photometries, clearer results about SED model comparison and higher quality of photometric parameter estimation can be obtained.
- 3. While SED model comparison with Bayesian evidence may favor SED models with very different complexities when using photometries from different surveys, the maximum likelihood (or equivalently the minimum χ2 )

tends to favor more complex models. For photometries with strong enough discriminative power, the two methods lead to more consistent results. However, for photometries without strong enough discriminative power, the two methods may lead to contradictory results. In both cases, the results of model selection with Bayesian evidence are more consistent with the measurements of the quality of parameter estimation.

- 4. In both cases with and without noise, the additional consideration of metallicity evolution helps to improve the quality of photometric redshift and stellar parameter estimation of galaxies, and increases the Bayesian evidence of the corresponding SED model.
- 5. In the case without noise, the DAL of Salim et al. (2018) is a much better choice than that of Calzetti et al. (2000) for the hydrodynamical simulation–based mock galaxy sample in CSST-like, CSST+Euclid-like, and COSMOSlike surveys. However, in the more realistic case with noise, it is only more favored in the COSMOS-like survey with Bayesian evidence–based model selection. With maximum likelihood (or equivalently minimum χ2 )–based model selection, it could be more favored, but leads to worse parameter estimation.
- 6. In the case without noise, more flexible forms of SFH lead to better quality of parameter estimation and increase the Bayesian evidence of the corresponding SED model. However, in the more realistic case with noise, they are only more favored in the COSMOS-like survey.
- 7. With a quenching (or rejuvenation) component added to the SFH, the quality of parameter estimation and the Bayesian evidence of the corresponding SED model decrease in all cases. Although rejuvenation or rapid quenching events may happen in some galaxies, this additional component of SFH is not very effective for most of the galaxies in the hydrodynamical simulation– based mock galaxy sample.
- 8. The quality of parameter estimation is closely related to the level of Bayesian evidence such that the SED model with the largest Bayesian evidence tends to give the best quality of parameter estimation, which is clearer for photometries with higher discriminative power. By using photometries without strong enough discriminative power, the quality of parameter estimation, especially that of stellar mass and SFR estimation, tends to decrease with the increase of SED model complexity.
- 9. Since direct measurements of the quality of parameter estimation as indicated by NMAD, BIA, and OLF are usually unavailable, Bayesian model comparison with Bayesian evidence can be used to find the best SED model that is not only the most efficient but also gives the best parameter estimation.
- 10. For photometric redshift, stellar mass, and SFR, accurate estimation becomes increasingly difficult, with the latter two being more sensitive to the selection of SED models.
- 11. For the photometric redshift estimation of galaxies in the CSST-like survey, observational noise is a more important source of error than the imperfect SED modeling. However, for the photometric stellar mass and SFR estimation of galaxies, the opposite is true.
- 12. The combination of photometries from CSST-like and Euclid-like surveys is helpful for improving the quality of photometric redshift estimation and crucial for more

accurate stellar mass estimation, but not very useful for SFR estimation.

- 13. With photometries in 26 bands from COSMOS-like surveys, by employing the same SED model, BayeSED-V3 can achieve similar quality of photometric redshift, stellar mass, and SFR estimation to previous works. Besides, with photometries in 26 bands from COSMOSlike surveys, more complicated SED models tend to be more favored, which is very different from the two cases with only photometries from CSST-like (seven bands) or CSST+Euclid-like (10 bands) surveys.
We conclude that the latest version of BayeSED is capable of achieving a good balance among speed, quality of simultaneous photometric redshift and stellar population parameter estimation of galaxies, and reliability of SED model comparison. This makes it suitable for the analysis of existing and forthcoming massive photometric data of galaxies in the CSST wide-field multiband imaging survey and others.

Generally, the current main bottleneck that limits the performance of the Bayesian approach for the simultaneous photometric redshift and stellar population parameter estimation of galaxies is the reliability of the SED synthesis (or modeling) procedure. Assuming a more flexible model is not a complete solution. We need an SED model that is not only more flexible but also more precisely accurate. It can be achieved by gradually adding more informative priors and physical constraints to the SED synthesis (modeling) procedure of galaxies, which is the subject of future works. The Bayesian model selection method with Bayesian evidence, a quantified Occam's razor, is very helpful to identifying the best SED model that is not only the most efficient but also gives the best parameter estimation.

The results about simultaneous photometric redshift and stellar population estimation presented in this work are not yet optimal, especially those about CSST. The contributions of nebular lines and continuum emission to the SED, which may help break some parameter degeneracies, are still mising in this work. It is also worth mentioning that the results of Bayesian SED model comparison and the metrics (OLF, BIA, and NMAD) of parameter estimation highly depend on the selected samples. In this paper, we have chosen a relatively broader sample to test overall performance in the CSST wide-field imaging survey. For a differently selected sample that is designed for answering a more specific scientific question, the results could be different.

Finally, in addition to multiband photometries, we may need more information from other forms of data, such as slitless spectroscopy and morphology parameters from the imaging to break severe parameter degeneracies. More advanced methods may be able to take advantage of all this information to give better redshift and/or stellar population parameter estimations of galaxies. These will be the subjects of future works as well.

# Acknowledgments

The authors gratefully acknowledge the PHOENIX Supercomputing Platform jointly operated by the Binary Population Synthesis Group and the Stellar Astrophysics Group at Yunnan Observatories, Chinese Academy of Sciences (CAS). We warmly thank the Horizon-AGN team, especially C. Laigle, for making their photometric catalogs and spectral data publicly available. We thank Fengshan Liu, Ye Cao, and Xinwen Shu for helpful discussions about S/N and magnitude limits.

We acknowledge support from the National Key R&D Program of China (Nos. 2021YFA1600401 and 2021YFA1600400), the National Science Foundation of China (grant Nos. 11773063, 12173037, 12233008, 12233005, 12073078, and 12288102), the China Manned Space Project (grant Nos. CMS-CSST-2021-A02, CMS-CSST-2021-A04, CMS-CSST-2021-A06, and CMS-CSST-2021-A07), the Light of West China program of the CAS, the Yunnan Ten Thousand Talents Plan Young & Elite Talents Project, the International Centre of Supernovae, and the Yunnan Key Laboratory (No. 202302AN360001). L.F. also gratefully acknowledges the support of the CAS Project for Young Scientists in Basic Research (No. YSBR-092), the Fundamental Research Funds for the Central Universities (WK3440000006), and the Cyrus Chun Ying Tang Foundation.

This work made use of the following software: ASTROPY (Astropy Collaboration et al. 2013, 2018), MATPLOTLIB (Hunter 2007), NUMPY (van der Walt et al. 2011), and H5PY (Collette et al. 2023).

# ORCID iDs

Yunkun Han https://orcid.org/0000-0002-2547-0434 Lulu Fan https://orcid.org/0000-0003-4200-4432 Xian Zhong Zheng https://orcid.org/0000-0003-3728-9912 Zhanwen Han https://orcid.org/0000-0001-9204-7778

# References

- Abdurro'uf, Lin, Y.-T., Wu, P.-F., & Akiyama, M. 2021, ApJS, 254, 15
- Acquaviva, V., Gawiser, E., & Guaita, L. 2011, ApJ, 737, 47
- Acquaviva, V., Raichoor, A., & Gawiser, E. 2015, ApJ, 804, 8
- Alsing, J., Peiris, H., Leja, J., et al. 2020, ApJS, 249, 5
- Alsing, J., Peiris, H., Mortlock, D., Leja, J., & Leistedt, B. 2023, ApJS, 264, 29 Antonucci, R. 1993, ARA&A, 31, 473
- Antonucci, R. 2012, A&AT, 27, 557
- Arnouts, S., Le Floc'h, E., Chevallard, J., et al. 2013, A&A, 558, A67
- Ashton, G., Bernstein, N., Buchner, J., et al. 2022, Nat. Rev. Methods Primers, 2, 39
- Astropy Collaboration, Price-Whelan, A. M., Sipőcz, B. M., et al. 2018, AJ, 156, 123
- Astropy Collaboration, Robitaille, T. P., Tollerud, E. J., et al. 2013, A&A, 558, A33
- Aufort, G., Ciesla, L., Pudlo, P., & Buat, V. 2020, A&A, 635, A136
- Bastian, N., Covey, K. R., & Meyer, M. R. 2010, ARA&A, 48, 339
- Beckmann, R. S., Devriendt, J., Slyz, A., et al. 2017, MNRAS, 472, 949
- Behroozi, P. S., Wechsler, R. H., & Conroy, C. 2013, ApJ, 770, 57
- Beichman, C. A., Rieke, M., Eisenstein, D., et al. 2012, Proc. SPIE, 8442, 84422N
- Bertelli, G., Girardi, L., Marigo, P., & Nasi, E. 2008, A&A, 484, 815
- Boquien, M., Burgarella, D., Roehlly, Y., et al. 2019, A&A, 622, A103
- Bowman, W. P., Zeimann, G. R., Nagaraj, G., et al. 2020, ApJ, 899, 7
- Breivik, K., Connolly, A. J., Ford, K. E. S., et al. 2022, arXiv:2208.02781
- Brott, I., de Mink, S. E., Cantiello, M., et al. 2011, A&A, 530, A115
- Brown, A., Nayyeri, H., Cooray, A., et al. 2019a, ApJ, 871, 87
- Brown, M. J. I., Duncan, K. J., Landt, H., et al. 2019b, MNRAS, 489, 3351
- Bruzual, G., & Charlot, S. 2003, MNRAS, 344, 1000
- Buchner, J. 2021, StSur, 17, 169
- Calzetti, D., Armus, L., Bohlin, R. C., et al. 2000, ApJ, 533, 682
- Cameron, E., & Pettitt, A. 2014, StaSc, 29, 397
- Cao, Y., Gong, Y., Meng, X.-M., et al. 2018, MNRAS, 480, 2178
- Cappellari, M., McDermid, R. M., Alatalo, K., et al. 2012, Natur, 484, 485 Caputi, K. I., Ilbert, O., Laigle, C., et al. 2015, ApJ, 810, 73
- Carnall, A. C., Leja, J., Johnson, B. D., et al. 2019, ApJ, 873, 44
- Carnall, A. C., McLure, R. J., Dunlop, J. S., & Davé, R. 2018, MNRAS,
- 480, 4379
- Chabrier, G. 2003, PASP, 115, 763
- Charlot, S., & Longhetti, M. 2001, MNRAS, 323, 887
- Chevallard, J., & Charlot, S. 2016, MNRAS, 462, 1415
- Choi, J., Conroy, C., & Johnson, B. D. 2019, ApJ, 872, 136
- Ciesla, L., Boselli, A., Elbaz, D., et al. 2016, A&A, 585, A43
- Ciesla, L., Elbaz, D., & Fensch, J. 2017, A&A, 608, A41
- Coelho, P. 2009, in AIP Conf. Proc. 1111, Probing Stellar Populations out to the Distant Universe: CEFALU 2008, ed. G. Giobbi et al. (Melville, NY: AIP ), 67
- Coelho, P. R. T., Bruzual, G., & Charlot, S. 2020, MNRAS, 491, 2025 Collette, A., Kluyver, T., Caswell, T. A., et al., 2023 h5py/h5py: 3.8.0 aarch64-wheels, Zenodo, 10.5281/zenodo.7568214
- Conroy, C. 2013, ARA&A, 51, 393
- Conroy, C., & Gunn, J. E. 2010, ApJ, 712, 833
- Conroy, C., Gunn, J. E., & White, M. 2009, ApJ, 699, 486
- Conroy, C., White, M., & Gunn, J. E. 2010, ApJ, 708, 58
- Côté, B., Ritter, C., O'Shea, B. W., et al. 2016, ApJ, 824, 82
- da Cunha, E., Charlot, S., & Elbaz, D. 2008, MNRAS, 388, 1595
- Dahlen, T., Mobasher, B., Faber, S. M., et al. 2013, ApJ, 775, 93
- Davidzon, I., Ilbert, O., Laigle, C., et al. 2017, A&A, 605, A70
- Davidzon, I., Laigle, C., Capak, P. L., et al. 2019, MNRAS, 489, 4817
- Debsarma, S., Chattopadhyay, T., Das, S., & Pfenniger, D. 2016, MNRAS, 462, 3739
- Diemer, B., Sparre, M., Abramson, L. E., & Torrey, P. 2017, ApJ, 839, 26
- Doore, K., Monson, E. B., Eufrasio, R. T., et al. 2023, ApJS, 266, 39
- Dore, O., Hirata, C., Wang, Y., et al. 2019, BAAS, 51, 341
- Draine, B. 2010, Physics of the Interstellar and Intergalactic Medium (Princeton, NJ: Princeton Univ. Press)
- Draine, B. T. 2003, ARA&A, 41, 241
- Dries, M., Trager, S. C., & Koopmans, L. V. E. 2016, MNRAS, 463, 886
- Dries, M., Trager, S. C., Koopmans, L. V. E., Popping, G., & Somerville, R. S. 2018, MNRAS, 474, 3500
- Driver, S. P., Robotham, A. S. G., Bland-Hawthorn, J., et al. 2013, MNRAS, 430, 2622
- Dubois, Y., Pichon, C., Welker, C., et al. 2014, MNRAS, 444, 1453
- Eldridge, J. J., & Stanway, E. R. 2009, MNRAS, 400, 1019
- Feroz, F., & Hobson, M. P. 2008, MNRAS, 384, 449
- Feroz, F., Hobson, M. P., & Bridges, M. 2009, MNRAS, 398, 1601
- Feroz, F., Hobson, M. P., Cameron, E., & Pettitt, A. N. 2019, OJAp, 2, 10
- Ferreras, I., La Barbera, F., de la Rosa, I. G., et al. 2013, MNRAS, 429, L15
- Fitzpatrick, E. L. 1986, AJ, 92, 1068
- Galliano, F., Galametz, M., & Jones, A. P. 2018, ARA&A, 56, 673
- Gardner, J. P., Mather, J. C., Clampin, M., et al. 2006, SSRv, 123, 485
- Gennaro, M., Tchernyshyov, K., Brown, T. M., et al. 2018, ApJ, 855, 20
- Gilda, S., Lower, S., & Narayanan, D. 2021, ApJ, 916, 43
- Gong, Y., Liu, X., Cao, Y., et al. 2019, ApJ, 883, 203
- Hahn, C., Kwon, K. J., Tojeiro, R., et al. 2022, ApJ, 945, 16
- Hahn, C., & Melchior, P. 2022, ApJ, 938, 11
- Han, Y., & Han, Z. 2012, ApJ, 749, 123
- Han, Y., & Han, Z. 2014, ApJS, 215, 2
- Han, Y., & Han, Z. 2019, ApJS, 240, 3
- Han, Y., Han, Z., & Fan, L. 2019, The Art of Measuring Galaxy Physical Properties, 5
- Han, Y., Han, Z., & Fan, L. 2020, in IAU Symp. 341, Challenges in Panchromatic Modelling with Next Generation Facilities, ed. M. Boquien et al. (Cambridge: Cambridge Univ. Press), 143
- Han, Z., Podsiadlowski, P., & Lynas-Gray, A. E. 2007, MNRAS, 380, 1098
- Hernández-Pérez, F., & Bruzual, G. 2013, MNRAS, 431, 2612
- Hickox, R. C., & Alexander, D. M. 2018, ARA&A, 56, 625
- Higson, E., Handley, W., Hobson, M., & Lasenby, A. 2019, Stat. Comput., 29, 891
- Hogg, D. W., & Foreman-Mackey, D. 2018, ApJS, 236, 11
- Hopkins, P. F., Kereš, D., Oñorbe, J., et al. 2014, MNRAS, 445, 581
- Hoversten, E. A., & Glazebrook, K. 2008, ApJ, 675, 163
- Hunter, J. D. 2007, CSE, 9, 90

MNRAS, 504, 2286

486, 1814

38

- Ilbert, O., Capak, P., Salvato, M., et al. 2009, ApJ, 690, 1236
- Ivezić, Ž., Kahn, S. M., Tyson, J. A., et al. 2019, ApJ, 873, 111
- Iyer, K., & Gawiser, E. 2017, ApJ, 838, 127
- Iyer, K. G., Gawiser, E., Faber, S. M., et al. 2019, ApJ, 879, 116
- Iyer, K. G., Tacchella, S., Genel, S., et al. 2020, MNRAS, 498, 430
- Joachimi, B. 2016, in ASP Conf. Ser. 507, Multi-Object Spectroscopy in the Next Decade: Big Questions, Large Surveys, and Wide Fields, ed. I. Skillen, M. Balcells, & S. Trager (San Francisco, CA: ASP), 401
- Kaviraj, S., Laigle, C., Kimm, T., et al. 2017, MNRAS, 467, 4739
- Kewley, L. J., Nicholls, D. C., & Sutherland, R. S. 2019, ARA&A, 57, 511 Knowles, A. T., Sansom, A. E., Allende Prieto, C., & Vazdekis, A. 2021,

Knowles, A. T., Sansom, A. E., Coelho, P. R. T., et al. 2019, MNRAS,

- Kriek, M., & Conroy, C. 2013, ApJL, 775, L16
- Laigle, C., Davidzon, I., Ilbert, O., et al. 2019, MNRAS, 486, 5104
- Laigle, C., McCracken, H. J., Ilbert, O., et al. 2016, ApJS, 224, 24
- Laureijs, R., Amiaux, J., Arduini, S., et al. 2011, arXiv:1110.3193
- Lawler, A. J., & Acquaviva, V. 2021, MNRAS, 502, 3993
- Lee, S.-K., Ferguson, H. C., Somerville, R. S., Wiklind, T., & Giavalisco, M. 2010, ApJ, 725, 1644
- Lee, S.-K., Idzi, R., Ferguson, H. C., et al. 2009, ApJS, 184, 100
- Leja, J., Carnall, A. C., Johnson, B. D., Conroy, C., & Speagle, J. S. 2019, ApJ, 876, 3
- Leja, J., Johnson, B. D., Conroy, C., van Dokkum, P. G., & Byler, N. 2017, ApJ, 837, 170
- Leja, J., Speagle, J. S., Johnson, B. D., et al. 2020, ApJ, 893, 111
- Lower, S., Narayanan, D., Leja, J., et al. 2020, ApJ, 904, 33
- Lower, S., Narayanan, D., Leja, J., et al. 2022, ApJ, 931, 14
- Lyu, J., & Rieke, G. 2022, Univ, 8, 304
- Ma, X., Hopkins, P. F., Faucher-Giguère, C.-A., et al. 2016, MNRAS, 456, 2140
- Madau, P. 1995, ApJ, 441, 18
- Maiolino, R., & Mannucci, F. 2019, A&ARv, 27, 3
- Maraston, C. 2005, MNRAS, 362, 799
- Maraston, C., Daddi, E., Renzini, A., et al. 2006, ApJ, 652, 85
- Marigo, P., Girardi, L., Bressan, A., et al. 2008, A&A, 482, 883
- Morishita, T., 2022 gsf: Grism SED Fitting Package, Astrophysics Source Code Library, ascl:2211.012
- Narayanan, D., Conroy, C., Davé, R., Johnson, B. D., & Popping, G. 2018, ApJ, 869, 70
- National Academies of Sciences, Engineering, and Medicine 2021, Pathways to Discovery in Astronomy and Astrophysics for the 2020s (Washington, DC: The National Academies Press)
- Netzer, H. 2015, ARA&A, 53, 365
- Noll, S., Burgarella, D., Giovannoli, E., et al. 2009, A&A, 507, 1793
- Oke, J. B. 1974, ApJS, 27, 21
- Pacifici, C., Charlot, S., Blaizot, J., & Brinchmann, J. 2012, MNRAS, 421, 2002
- Padoan, P., Nordlund, A., & Jones, B. J. T. 1997, MNRAS, 288, 145
- Pforr, J., Maraston, C., & Tonini, C. 2012, MNRAS, 422, 3285
- Pforr, J., Maraston, C., & Tonini, C. 2013, MNRAS, 435, 1389
- Qiu, Y., & Kang, X. 2022, ApJ, 930, 66
- Reddy, N. A., Kriek, M., Shapley, A. E., et al. 2015, ApJ, 806, 259
- Reddy, N. A., Pettini, M., Steidel, C. C., et al. 2012, ApJ, 754, 25
- Rieke, M. J., Kelly, D., & Horner, S. 2005, Proc. SPIE, 5904, 1
- Robertson, B. E. 2022, ARA&A, 60, 121
- Robotham, A. S. G., & Bellstedt, S. 2020, MNRAS, 495, 905
- Salim, S., & Boquien, M. 2019, ApJ, 872, 23
- Salim, S., Boquien, M., & Lee, J. C. 2018, ApJ, 859, 11
- Salim, S., & Narayanan, D. 2020, ARA&A, 58, 529
- Salmon, B., Papovich, C., Long, J., et al. 2016, ApJ, 827, 20
- Salvato, M., Ilbert, O., & Hoyle, B. 2018, NatAs, 3, 212
- Schmidt, M. 1959, ApJ, 129, 243
- Seon, K.-I., & Draine, B. T. 2016, ApJ, 833, 201 Sharma, S. 2017, ARA&A, 55, 213
- 
- Shivaei, I., Reddy, N., Rieke, G., et al. 2020, ApJ, 899, 117 Skilling, J. 2004, in AIP Conf. Proc. 735, 24th International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and
- Engineering, ed. U. V. T. R. Fischer & R. Preuss (Melville, NY: AIP), 395 Skilling, J. 2006, BayAn, 1, 833
- Speagle, J. S. 2020, MNRAS, 493, 3132
- Spergel, D. N., Verde, L., Peiris, H. V., et al. 2003, ApJS, 148, 175
- Suess, K. A., Leja, J., Johnson, B. D., et al. 2022, ApJ, 935, 146
- Tacconi, L. J., Genzel, R., & Sternberg, A. 2020, ARA&A, 58, 157
- Tanaka, M. 2015, ApJ, 801, 20
- Thomas, D., & Maraston, C. 2003, A&A, 401, 429
- Thorne, J. E., Robotham, A. S. G., Davies, L. J. M., et al. 2021, MNRAS, 505, 540
- Tinsley, B. M. 1978, ApJ, 222, 14
- Tinsley, B. M. 1980, FCPh, 5, 287
- Tinsley, B. M., & Gunn, J. E. 1976, ApJ, 203, 52
- Valentini, M., Borgani, S., Bressan, A., et al. 2019, MNRAS, 485, 1384
- van der Walt, S., Colbert, S. C., & Varoquaux, G. 2011, CSE, 13, 22
- van Dokkum, P. G. 2008, ApJ, 674, 29
- Volonteri, M., Dubois, Y., Pichon, C., & Devriendt, J. 2016, MNRAS, 460, 2979
- Walcher, J., Groves, B., Budavári, T., & Dale, D. 2011, Ap&SS, 331, 1
- Wang, E., & Lilly, S. J. 2020, ApJ, 892, 87
- Weingartner, J. C., & Draine, B. T. 2001, ApJ, 548, 296
- Witt, A. N., & Gordon, K. D. 2000, ApJ, 528, 799
- Yallup, D., Janßen, T., Schumann, S., & Handley, W. 2022, EPJC, 82, 678
- Yan, R., Chen, Y., Lazarz, D., et al. 2019, ApJ, 883, 175
- Zhan, H. 2011, SSPMA, 41, 1441
- Zhan, H. 2018, 42nd COSPAR Scientific Assembly, E1.16
- Zhan, H. 2021, ChSBu, 66, 1290
- Zhang, F., Han, Z., Li, L., & Hurley, J. R. 2005, MNRAS, 357, 1088
- Zhou, X., Gong, Y., Meng, X.-M., et al. 2022a, RAA, 22, 115017
- Zhou, X., Gong, Y., Meng, X.-M., et al. 2022b, MNRAS, 512, 4593

