# DECODING SPECTRAL ENERGY DISTRIBUTIONS OF DUST-OBSCURED STARBURST–ACTIVE GALACTIC NUCLEUS

Yunkun Han1*,*2*,*3 and Zhanwen Han1*,*3

1 National Astronomical Observatories*/*Yunnan Observatory, Chinese Academy of Sciences, Beijing 100012, China; hanyk@ynao.ac.cn 2 Graduate University of Chinese Academy of Sciences, Beijing 100049, China 3 Key Laboratory for the Structure and Evolution of Celestial Objects, Chinese Academy of Sciences, Kunming 650011, China

*Received 2011 November 11; accepted 2012 February 11; published 2012 March 29*

## ABSTRACT

We present BayeSED, a general purpose tool for Bayesian analysis of spectral energy distributions (SEDs) using pre-existing model SED libraries or their linear combinations. The artificial neural networks, principal component analysis, and multimodal-nested sampling (MultiNest) techniques are employed to allow the highly efficient sampling of posterior distribution and the calculation of Bayesian evidence. As a demonstration, we apply this tool to a sample of hyperluminous infrared galaxies (HLIRGs). The Bayesian evidence obtained for a pure starburst, a pure active galactic nucleus (AGN), and a linear combination of starburst+AGN models show that the starburst+AGN model has the highest evidence for all galaxies in this sample. The Bayesian evidence for the three models and the estimated contributions of starbursts and AGNs to infrared luminosity show that HLIRGs can be classified into two groups: one dominated by starbursts and the other dominated by AGNs. Other parameters and corresponding uncertainties about starbursts and AGNs are also estimated using the model with the highest Bayesian evidence. We find that the starburst region of the HLIRGs dominated by starbursts tends to be more compact and has a higher fraction of OB stars than that of HLIRGs dominated by AGNs. Meanwhile, the AGN torus of the HLIRGs dominated by AGNs tends to be more dusty than that of HLIRGs dominated by starbursts. These results are consistent with previous researches, but need to be tested further with larger samples. Overall, we believe that BayeSED could be a reliable and efficient tool for exploring the nature of complex systems such as dust-obscured starburst–AGN composite systems by decoding their SEDs.

*Key words:* galaxies: active – galaxies: evolution – galaxies: ISM – galaxies: starburst – methods: data analysis – methods: statistical

*Online-only material:* color figures

## 1. INTRODUCTION

The formation and evolution of galaxies and super-massive black holes (SMBHs) are now believed to be tightly related. Meanwhile, the violent formation of stars (starbursts) and the growth of SMBHs (AGNs) are found to be coupled and ongoing together in the most infrared-luminous galaxies, such as ultraluminous infrared galaxies (ULIRGs) and hyperluminous infrared galaxies (HLIRGs). These galaxies represent important phases in the formation and evolution of galaxies and are ideal laboratories for studying starburst–active galactic nucleus (AGN) connections. Since both the star formation and SMBH accretion are taking place with a large amount of dust distributed throughout, these kinds of galaxies are very complex dust-obscured starburst–AGN composite systems. The spectral energy distributions (SEDs) are the primary source of information for our understanding of them. However, it is currently still very challenging to efficiently extract the basic physical properties of these galaxies from the analysis of their SEDs.

The analysis of SED, or SED fitting, tries to extract one or several physical properties of a galaxy by fitting models to the observed SED. Nowadays, new observing facilities and large surveys allow us to obtain the full SEDs at wavelengths from the X-rays to the radio for galaxies extending from local to redshifts higher than six. On the other hand, as the starting point of SED fitting, a library of model SEDs must be built in advance. For most galaxies, stars are the main sources of lights. The evolutionary population synthesis models (Tinsley 1972, 1978; Searle et al. 1973; Larson & Tinsley 1978; Bruzual 1983; Fioc & Rocca-Volmerange 1997; Leitherer et al. 1999; Bruzual & Charlot 2003; Maraston 2005; Zhang et al. 2005a, 2005b; Han et al. 2007; Conroy et al. 2009), which are based on the knowledge of stellar evolution such as the assumed stellar initial mass function, star formation history, stellar evolutionary tracks, and stellar libraries, are standard tools for modeling the SEDs of galaxies.

Meanwhile, the dusty interstellar medium (ISM), if presented, has important effects on the resulting SEDs of galaxies. A fraction of the initial radiations from stars, or most of them in extreme cases such as ULIRGs and HLIRGs, are absorbed and reprocessed by the gas and dust in the ISM. Gases heated by young stars produce luminous nebular emission lines, while dusts heated by stars of all ages produce the mid-infrared (MIR) and far-infrared (FIR) emission. A simple method is to handle the absorption of star lights and their re-emission independently. Then, they can be connected by assuming that the total energy absorbed in the UV–optical equals the total energy re-emitted in the MIR and FIR (Devriendt et al. 1999; da Cunha et al. 2008; Noll et al. 2009). A more self-consistent treatment requires detailed radiative transfer (RT) calculations to be performed using ray tracing (Silva et al. 1998; Efstathiou et al. 2000; Granato et al. 2000; Tuffs et al. 2004; Siebenmorgen & Krugel ¨ 2007; Groves et al. 2008) or Monte Carlo method (Baes et al. 2003; Dullemond & Dominik 2004; Jonsson 2006; Chakrabarti & Whitney 2009). However, these RT calculations are commonly computationally expensive.

If a powerful AGN is presented in the center of a galaxy, the resulting SED would be largely modified. AGNs can contribute to all wavelength regimes of the electromagnetic spectrum, with accretion disk+corona to the X-ray–UV–optical, torus to the IR, and jet to the radio and gamma ray in some cases. In quasars, the AGN light dominates the integrated galaxy light at almost any wavelength, while for AGNs with lower luminosities, the contribution of the host galaxy may dominate in many wavelengths. The modeling of the SEDs of various components of AGNs has been developed independently, and all need a special suite of parameters. Meanwhile, the highenergy radiations from the center AGN can also be absorbed by the ISM in the host galaxy and re-emitted in the IR. So, if an AGN is presented, the relative geometry of the starburst, ISM, and AGN is important for modeling the SEDs of such dustobscured starburst–AGN composite systems. Furthermore, if violent starburst and AGN activities are coupled and ongoing together, it may not be reasonable to model the SEDs of such systems by a simple linear combination of models for starbursts and AGNs developed independently.

Overall, the SEDs of dust-obscured starburst–AGN composite systems are very complicated. A completely self-consistent SED model for such complicated systems must be very hard to construct. To make progress, parameterizations of all possible components, their relative geometry, and their possible physical relations are inevitable. Given the complexities mentioned above, it is natural to expect a large number of parameters and many possible degeneracies between them. Meanwhile, since the effects of dust attenuation, line emission, and dust emission have to be taken into account, the problem of determining the physical parameters from fitting the model to observations is highly nonlinear.

The SED-fitting methods have been improved significantly in the last decade, which allows us to extract much more complex information imprinted in the SEDs (see Walcher et al. 2011, for a recent review of this field). Among the numerous SED-fitting methods, we believe the method based on Bayesian inference (Ben´ıtez 2000; Kauffmann et al. 2003; Feldmann et al. 2006; Salim et al. 2007; Bailer-Jones 2011) in which multi-wavelength SEDs are fitted by first precomputing a library of model SEDs with varying degrees of complexity and afterward determining the model and*/*or model parameters that best fit the data, is the best choice for the problem we are facing. This method gives us detailed probability distributions of model parameters and the Bayesian evidence as a quantitative evaluation of the entire model given the data.

However, the Bayesian approach requires an intensive sampling of the posterior distribution, which is a function of all parameters, resulting from combining all prior knowledge about the parameters of the model and the new information introduced by the observations. So, if the model used to explain the observations is itself computationally expensive, the sampling must be very time consuming. Furthermore, as mentioned above, the model SEDs are commonly precomputed as a library, rather than being computed during the sampling. For this reason, an interpolation method must be employed. Since the dependences of parameters with the resulting SED are highly nonlinear and the number of parameters is very large, common interpolation methods are not suitable. This problem can be solved more easily with artificial neural networks (ANNs; Lahav et al. 1996; Bertin & Arnouts 1996; Andreon et al. 2000; Firth et al. 2003; Collister & Lahav 2004; Vanzella et al. 2004; Carballo et al. 2008; Auld et al. 2008). An ANN can be trained to approximate a library of model SEDs with highly nonlinear complexities, allowing the parameter space of the model to be explored more continuously. On the other hand, a principal component analysis (PCA; Francis et al. 1992; Glazebrook et al. 1998; Wild & Hewett 2005; Wild et al. 2007; Budavari et al. ´ 2009) can be applied to the library of model SEDs in advance to simplify the required structure of the ANN.

These methods have been demonstrated nicely by Asensio Ramos & Ramos Almeida (2009), who applied them to a recently developed public database of a clumpy dusty torus model (Nenkova et al. 2008a, 2008b). Almeida et al. (2010) and Silva et al. (2011) implemented ANN into the RT code GRASIL to speed up the computation of the SED of galaxies when applied to semianalytic models (White & Rees 1978; Lacey & Silk 1991; White & Frenk 1991). The core of these methods is actually very general and can be applied to any problem regarding fitting precomputed libraries of model SEDs to observations. However, these methods are commonly implemented for a specific special problem and are not convenient to be used in other similar problems. Inspired by these works, we have built a suite of general purpose programs to generalize these methods such that they can be integrated together to do the Bayesian analysis of SEDs by comparing pre-existing model SED libraries or their linear combinations with observations.

This paper is structured as follows: In Section 2, we describe the PCA and ANN methods used to boost the generation of model SEDs, and our implements for them. In Section 3, we present our Bayesian inference tool, BayeSED. In Section 3.1, we give a general introduction to Bayesian inference methods. In Section 3.2, we discuss the posterior sampling methods. The construction of BayeSED is presented in Section 3.3. In Section 4, we apply this tool to the HLIRG sample of Ruiz et al. (2007). Finally, a summary of this paper is presented in Section 5.

## 2. GENERATION OF MODEL SEDs

## *2.1. Principal Component Analysis of Model SED Libraries*

An SED can be described by a vector of *N* flux densities (*f*1*,f*2*,...,fN* ) at wavelengths (*λ*1*,λ*2*,...,λN* ). However, when the flux at a given wavelength is changed, the fluxes at surrounding wavelength points are also changed in a similar way due to the continuity of the SED. Therefore, the fluxes at different wavelengths are not completely independent, and the actual dimension of the SED is much smaller than*N*. This simple fact makes it possible to apply some dimensionality reduction techniques to efficiently compress the representation of an SED.

One such technique is called PCA. It can be used to derive an optimal set of linear components, called principal components (PCs), by diagonalizing the covariance matrix of a set of SEDs to find the directions of greatest variation. Then, the original data can be approximated by a linear combination of first *N N* independent PCs. It is worth noting that PCA performs a linear analysis. If the dependences of parameters with corresponding SED are nonlinear, the number of necessary eigenvectors is commonly larger than the number of physical parameters of the model.

We adopt an IDL package for PCA,4 which gathers several algorithms for PCA into a single package, all with the same usage. The Singular Value Decomposition (SVD) is used rather than the Robust and Iterative algorithm provided in this package

<sup>4</sup> http://www.roe.ac.uk/vw/vwpca.tar.gz

**Figure 1.** First 16 eigenvectors obtained from PCA of the SBgrid (left) and CLUMPY (right) model SED libraries.

(Budavari et al. ´ 2009). The later two algorithms, when applied to the observed SEDs, can be used to obtain eigenvectors with clearer physical meanings. However, we apply PCA to model SEDs to reduce them in a purely mathematical sense. Meanwhile, the SVD algorithm is much faster and is good enough for our purpose.

We have applied PCA to two widely used model SED libraries: SBgrid for starburst galaxies and ULIRGs (Siebenmorgen & Krugel ¨ 2007) and CLUMPY for AGN clumpy torus (Nenkova et al. 2008a, 2008b). The model SEDs in the SBgrid library have five parameters: the nuclear radius *R*, the total luminosity *L*tot, the ratio of the luminosity of OB stars with hot spots to the total luminosity *f*OB, the visual extinction from the edge to the center of the nucleus *A*V, and the dust density in the hot spots *n*hs (see Siebenmorgen & Krugel ¨ 2007, for detailed explanations about these parameters). On the other hand, the model SEDs in the CLUMPY library have six parameters for the dusty and clumpy torus: the ratio of the outer to the inner radii of the toroidal distribution *Y* = *R*o*/R*d, the optical depth of clumps *τ*V, the number of clouds along a radial equatorial path *N*, the power of the power law (*r*−*q* ) describing radial density profile *q*, the width parameter characterizing the angular distribution *σ*, and the viewing angle measured from the torus polar axis *i*.

It is necessary to do some normalizations to the libraries before applying the PCA. First, we find that the PCA will perform better if we use the logarithm of flux. Second, the mean spectra of an SED library is found and removed from every SED in the library in advance. These normalizations impact the resulting eigenvectors. The different physical mechanisms associated with each eigenvector are not important for us, and we have not tried to find them out. We only treat the PCA eigenvectors as a set of a purely mathematical basis that allows us to efficiently reconstruct all SEDs in a library.

The first 16 eigenvectors that we have obtained for the two libraries are shown in Figure 1. This figure shows that low-order eigenvectors, which have larger variations in amplitude for different SEDs, are much smoother than the high-order eigenvectors, which have smaller variations in amplitude for different SEDs. The low-order eigenvectors determine the general shape of an SED, while the high-order eigenvectors give some more details of the SED. Then, any SED*i* in the original library can be approximately reconstructed from a linear combination of these eigenvectors as follows:

$$\mathrm{SED}_{i}\approx\sum_{j=1}^{16}C_{i,j}PC_{j}.\qquad\qquad(1)$$

Since the SEDs in the library have been normalized in advance, the corresponding inversions are needed after this. In Figure 4 of the next subsection, an example of SED reconstruction is shown for comparison with that obtained from ANN. Like Asensio Ramos & Ramos Almeida (2009), we found that 16 PCs are enough to obtain an acceptable reconstruction for model SED libraries as complex as CLUMPY and SBgrid. Now, each SED of the two libraries can be represented by a vector of 16 coefficients corresponding to 16 PCs, rather than a vector of 124 fluxes for the CLUMPY library or a vector of 318 fluxes for the SBgrid library. It is clear that with the help of PCA the size of a model SED library can be greatly reduced.

## *2.2. Implements of Artificial Neural Networks*

ANNs are mathematical constructs designed to simulate some intellectual behaviors of the human brain. For example, it can "learn" relations between certain inputs and outputs by training with living examples. After that, it can be used to predict the outputs from a new set of inputs. Nowadays, ANNs have been used successfully in a wide range of problems in cosmology and astrophysics (Lahav et al. 1996; Bertin & Arnouts 1996; Andreon et al. 2000; Firth et al. 2003; Collister & Lahav 2004; Vanzella et al. 2004; Carballo et al. 2008; Auld et al. 2008). Here, we use ANNs to learn the relations between parameters and the resulting SEDs for libraries of model SEDs. After training, an ANN can be used as a substitute for the model SED library which is used to train it and can even interpolate the library to obtain the SED for values of the parameters not present in the original grid.

There are different implements of ANNs which differ in neuron (node) organization and information exchanging methods. We have modified ANNz, a widely used tool for estimating photometric redshifts using ANN, to be a more convenient and general purpose ANN code without changing the technical implement of ANN. The type of ANN implemented in ANNz is so-called multi-layer perceptron (MLP) feed-forward network. An MLP network consists of a number of layers of nodes with

**Figure 2.** Network architecture of ANNs for the SBgrid (five input parameters) and CLUMPY (six input parameters) libraries. In both cases, a hidden layer with 20 nodes is used. The outputs of each ANN are the coefficients corresponding to the first 16 PCs.

the first layer containing the inputs, the final layer containing the outputs, and one or more intervening (or "hidden") layers. In a feed-forward network, which is the most widely used due to its simplicity, information propagates sequentially from the input layer through the hidden layers to the output neurons without any feedback. The network architecture of such an ANN can be denoted by *N*in:*N*1:*N*2: *...* :*N*out, where *N*in is the number of input nodes, *Ni* is the number of nodes in *i*th hidden layer, and *N*out is the number of output nodes.

In Figure 2, we show the network architectures of ANNs used for SBgrid and CLUMPY libraries. The inputs of an ANN are the parameters of the library used to train it. So, the ANN for SBgrid library has five inputs, while the ANN for CLUMPY library has six inputs. The capability of an ANN is determined by the structure of its hidden layers. In mathematics, the universal approximation theorem (Cybenko 1989; Hornik 1991; Haykin 1999) states that a standard multi-layer feed-forward network with only a single hidden layer and an arbitrary continuous, bounded, and nonconstant activation function can approximate any continuous function to arbitrary precision, provided only that a sufficiently high number of hidden units are available. More nodes in a single hidden layer or even more hidden layers can increase the degree of approximation, but with the expense of much more training time. In practice, we found that a single hidden layer with 20 nodes is enough for the two libraries. The outputs of ANNs are set to be the projections of an SED on the first 16 PCs (eigenvector). So, the structure of ANNs for the SBgrid and CLUMPY libraries can be denoted as 5:20:16 and 6:20:16, respectively.

An ANN "learns" the relationship between inputs and outputs from examples (pairs of inputs and corresponding outputs). In our case, the examples are model SEDs of a library whose parameters and corresponding projections are already known. When a set of inputs is given, the ANN "learns" the relationship by adapting weights associated with connections between nodes so as to minimize the cost function, which represents the difference between the prediction made by the ANN and the expected outputs. An iterative quasi-Newtonian method is used in ANNz to perform this minimization. Meanwhile, an activation function, which is taken to be a sigmoid function in ANNz, is defined at each node to simulate the behavior of biological neurons. This defines the signal propagation rule of an ANN in the sense that a neuron is activated, which means it transmits the received signal further on, when the total of received signals is greater than a certain threshold.

To avoid overfitting to the training set and to optimize the generalization performance of the network, the SEDs in a library are separated into two sets: a training set and a validation set. Both of these are randomly selected from the library. For the SBgrid library,5 the training set contains 6495 (90%) SEDs while the validation set contains 721 (10%) SEDs. The CLUMPY library currently contains about 1,307,980 SEDs. Although all of these SEDs can be used to train the ANN, we found that this is not necessary. Therefore, we have randomly selected about 10% of the SEDs in the CLUMPY library, amounting to 130,800 SEDs. Then, 117,720 (90%) of these are used as the training set while the other 13,080 (10%) are used as the validation set. The ANN usually converges to different local minima of the cost function, depending on the particular initialization. For each library, a group of four networks (called a "committee") with the same structure but different initializations are trained independently, and the mean of the individual outputs of the four networks are used as a more accurate estimate for the outputs.

In Figure 3, the projections of the first four PCs for the SEDs in the validation set from the ANN are compared with the one directly from PCA of the libraries. As clearly shown, the projections can be reliably predicted by the ANN. For both libraries, the rms error *σ*rms of the predicted projections are very small. It is therefore reasonable to expect that the SEDs in the libraries can be reliably reconstructed using these projections as coefficients for the linear combination of PCs. In Figure 4, examples of SED reconstruction using projections as coefficients for the linear combination of PCs are shown. It is clear that the SEDs in the two libraries can be reliably reconstructed using the projections directly from PCA of the libraries or the one predicted from ANN.

## 3. BayeSED: A TOOL FOR BAYESIAN ANALYSIS OF SED

## *3.1. Bayesian Inference*

Bayesian methods have already been widely used in astrophysics and cosmology (see, e.g., Trotta 2008, for a review). They have the advantages over traditional statistical tools of higher efficiency and of a more consistent conceptual basis for dealing with problems of induction in the presence of uncertainties. Bayesian methods are basically divided into two categories: parameter estimation and model comparison. The basis of these methods is the so-called Bayes' theorem, which states that

$$P(\theta|d,M)=\frac{P(d|\theta,M)P(\theta|M)}{P(d|M)},\tag{2}$$

where *θ* represents a vector of parameters, *d* represents a vector of data sets, and *M* represents a model under consideration.

The left side of Equation (2), *P*(*θ*|*d,M*) is called the *posterior probability* of parameter *θ* given data *d* and model *M*. It is proportional to the *sampling distribution* of the data *P*(*d*|*θ,M*) assuming the model is true, and the *prior probability* of the model, *P*(*θ*|*M*) ("the prior"), which describes

<sup>5</sup> Four SEDs in this library have been excluded, since they become discontinuous below about 1 *μ*m.

**Figure 3.** Left: the projections on the first four PCs for the 721 SEDs (validation set) randomly selected from the SBgrid library are predicted from ANN and then compared with the projection obtained directly from PCA. Right: as left, but for 13,080 SEDs (validation set) randomly selected from the CLUMPY library. In both cases, the projections can be predicted from the ANN with very small *σ*rms. The SEDs in the libraries can therefore be reliably reconstructed from PCs by using these projections as coefficients for the linear combination of PCs.

(A color version of this figure is available in the online journal.)

**Figure 4.** Left: examples of model SEDs (red points) of the SBgrid (top) and CLUMPY (bottom) libraries compared with the one directly reconstructed using the projections on the first 16 PCs (blue dash line) obtained by PCA of the library, and the one reconstructed using the projections predicted by the ANN for the same set of parameters (black solid line). Right: the projections of the first 16 PCs of the model SED directly from PCA of the libraries (blue points) compared with those from the ANN for the same set of parameters (black points with error bar and connected with black line).

(A color version of this figure is available in the online journal.)

knowledge about the parameters acquired before seeing (or irrespective of) the data. The *sampling distribution* describes how the degree of plausibility of the parameter *θ* changes when new data *d* is acquired. It is called *the likelihood function* when being considered as a function of the parameter *θ* and is often written as *L*(*θ*) ≡ *P*(*d*|*θ,M*).

The posterior probability density function (PDF) for one parameter is obtained by marginalizing out (integrating out) other parameters from the full posterior distribution:

$$P(\theta_{i}|\mathbf{d},\,M)=\int d\theta_{1}\ldots d\theta_{i-1}d\theta_{i+1}\ldots d\theta_{N_{\rm{pu}}}\,P(\mathbf{\theta}|\mathbf{d},\,M).\tag{3}$$

The normalization constant *P*(*d*|*M*) is called the *marginal likelihood* (or "Bayesian evidence"), which is not important for parameter estimation but is critical for model comparison and given by

$$P(d|M)\equiv\sum_{\theta}P(d|\theta,\,M)P(\theta|M),\tag{4}$$

where the sum runs over all the possible choices of the parameter *θ*. For a continuous parameter space Ω*M* , this can be rewritten as

$$P(d|M)\equiv\int_{\Omega_{M}}P(d|\theta,M)P(\theta|M)d\theta.\tag{5}$$

In the case of SED fitting, *d* represents the observed SED of a galaxy while *θ* represents the parameters of a model SED library. Commonly, *M* represents an SED library as a whole. However, multiple independent SED components (e.g., a starburst component and an AGN component) are needed in many cases. Then, different combinations of independent SED components should be considered as different models. All parameters of sub-models are combined together to be a new vector of parameters *θ*. For libraries giving relative flux, a free scaling factor needs to be considered as an additional parameter in the new *θ*.

## *3.2. Posterior Sampling Methods*

A key step in the Bayesian inference problem outlined above is the evaluation of the posterior of Equation (2), where accurate analytical solutions are commonly either difficult to obtain or simply do not exist. As a consequence, some efficient and robust sampling techniques have been developed. A widely used technique is called Markov Chain Monte Carlo (MCMC). An MCMC sampler, which is often based on the standard Metropolis–Hastings algorithm, provides a way to explore the posterior distribution such that the number density of samples is asymptotically converged to be proportional to the joint posterior PDF of all parameters. It therefore allows one to map out numerically the posterior distribution even in the case where the parameter space has hundreds of dimensions and the posterior is multimodal and has a complicated structure.

However, such methods can be computationally intensive when the posterior distribution is multimodal or with large degeneracies between parameters, particularly in high dimensions. On the other hand, the calculation of Bayesian evidence, which is the key ingredient for Bayesian model comparison, is extremely computationally intensive using MCMC techniques. Another Monte Carlo method called Nested sampling (Skilling 2004; Mukherjee et al. 2006; Shaw et al. 2007) provides a more efficient method for the calculation of Bayesian evidence but also produces posterior inferences as a by-product. Here, we adopt a newly developed, highly efficient, and freely available Bayesian inference tool, called MultiNest (Feroz & Hobson 2008; Feroz et al. 2009). It is as efficient as standard MCMC methods for Bayesian parameter estimation, more efficient for very accurate evaluation of Bayesian evidences for model comparison, and fully parallelized.

## *3.3. Building-Up of BayeSED*

The general structure of our Bayesian inference tool for the analysis of SED, BayeSED, is shown in Figure 5. The core of BayeSED is the sampling of posterior probability with an MCMC or MultiNest sampler. This is shown as a loop in the figure. During the sampling, the sampler provides proposal parameter vectors for the ANN, and the ANN predicts the coefficients for the reconstruction of model SEDs using the proposed parameter set. After training with some model SEDs in a library, an ANN can help generate the model SED of any parameter vector within the parameter space covered by the library used to train it. Here, it is allowed to simultaneously use multiple ANNs, which are trained with different model SED libraries. The comparison of the model with observations gives a *χ*(*θ*) 2 . Then, the likelihood function is given by *L*(SEDobs|*θ,M*) ≡ *e*−*χ*( −→*θ* ) 2 */*2.

The priors represent our knowledge about the parameters of the model that are independent of current observations. If we have no prior knowledge about the model parameters, the prior distributions are commonly assumed to be uniform between two physically chosen bounds. When the sampling is converged, the posterior PDF for all parameters and the Bayesian evidence for the model are obtained. Then, a posterior distribution differing from a uniform distribution would imply that new information about the corresponding parameter is obtained from observations. On the other hand, the ratio of evidence for two models, the so-called Bayes factor, tells us how their relative plausibility should be changed as suggested by the new observations.

## 4. APPLICATION TO A SAMPLE OF HYPERLUMINOUS INFRARED GALAXIES

## *4.1. The HLIRG Sample and Data*

The sample studied here is the one selected by Ruiz et al. (2007) from the Rowan-Robinson (2000) sample of 45 HLIRGs. The sample is limited to sources with available X-ray data and with redshift less than ∼2 to avoid strong biasing toward highredshift quasars. Consequently, the final sample contains 13 objects. Ruiz et al. (2010) have built multi-wavelength (from radio to X-rays) SEDs for these HLIRGs. They fitted standard empirical AGN and starburst templates to these SEDs and classified the HLIRGs into two groups, named class A and class B, according to their different SED shapes. These authors also suggested that their simple template-fitting approach should be complemented with other theoretical models of starburst and AGN emission.

Here, we present a re-analysis of the SEDs of these HLIRGs using different RT models of starburst and AGN emission, putting it on a solid statistical basis. The redshifts and observed SEDs of these galaxies are taken from the Table 1 and B of Ruiz et al. (2010), respectively. The SEDs have been converted to monochromatic flux density, corrected for the Galactic reddening, and blueshifted to the rest frame. Before comparing with model SEDs, we convert the monochromatic flux density to monochromatic luminosity by using the luminosity distance *dL*(*z*).

## *4.2. Bayesian Analysis of SEDs*

### *4.2.1. Models and Priors*

Three different models are employed to do Bayesian analysis of the SEDs of these HLIRGs. The first is the pure starburst

**Figure 5.** Simple flowchart for the Bayesian analysis of SED boosted by PCA and ANN.

**Table 1** The Bayesian Evidences of the "SB," "AGN," and "SB+AGN" Models for Galaxies in the Classes A and B of the Ruiz et al. (2007) HLIRG Sample

| Source | ln(evSB) | ln(evAGN) | ln(evSB+AGN) |
| --- | --- | --- | --- |
| Class A HLIRG |  |  |  |
| PG1206+459 | −12.65+0.09 −0.09 | −15.08+0.08 −0.08 | −9.39+0.08 −0.08 |
| PG1247+267 | −11.45+0.09 −0.09 | −8.02+0.08 −0.08 | −7.71+0.08 −0.08 |
| IRASF12509+3122 | −9.78+0.09 −0.09 | −7.81+0.07 −0.07 | −7.38+0.07 −0.07 |
| IRAS14026+4341 | −54.16+0.13 −0.13 | −21.64+0.12 −0.12 | −19.27+0.12 −0.12 |
| IRASF14218+3845 | −5.91+0.06 −0.06 | −6.36+0.06 −0.06 | −5.59+0.06 −0.06 |
| IRAS16347+7037 | −35.98+0.12 −0.12 | −23.48+0.12 −0.12 | −18.10+0.12 −0.12 |
| IRAS18216+6418 | −172.25+0.14 −0.14 | −72.53+0.14 −0.14 | −26.36+0.14 −0.14 |
| Class B HLIRG |  |  |  |
| IRASF00235+1024 | −11.15+0.09 −0.09 | −38.92+0.09 −0.09 | −11.09+0.09 −0.09 |
| IRAS07380−2342 | −159.50+0.12 −0.12 | −179.24+0.15 −0.15 | −76.36+0.17 −0.17 |
| IRAS00182−7112 | −22.09+0.10 −0.10 | −57.88+0.14 −0.14 | −19.10+0.12 −0.12 |
| IRAS09104+4109 | −31.31+0.11 −0.11 | −71.56+0.15 −0.15 | −29.62+0.13 −0.13 |
| IRAS12514+1027 | −63.33+0.10 −0.10 | −62.12+0.13 −0.13 | −30.01+0.14 −0.14 |
| IRASF15307+3252 | −12.64+0.10 −0.10 | −51.24+0.14 −0.14 | −11.97+0.10 −0.10 |

model of Siebenmorgen & Krugel ( ¨ 2007), as presented in the SBgrid library (noted as hereafter "SB"). The priors for the five parameters of this model are assumed to be uniform distributions truncated to the following intervals: *R* = [0*.*35*,* 15] kpc, *f*OB = [0*.*4*,* 1], log(*L*tot*/L* ) = [10*,* 14*.*7], *A*V = [2*.*2*,* 144], and log(*n*hs*/*cm−3) = [2*,* 4]. The SEDs in the SBgrid library are in unit of absolute flux at a distance of 50 Mpc. The absolute flux values have been multiplied by 4*π* × 50 Mpc × 50 Mpc in advance to convert to absolute luminosity values.

The second is the pure AGN model of Nenkova et al. (2008a), as presented in the CLUMPY library (noted as hereafter "AGN"). The priors for the six parameters of this model are also assumed to be uniform distributions truncated to the following intervals: *σ* = [15*,* 75], *Y* = [5*,* 200], *N* = [1*,* 24], *q* = [0*,* 4*.*5], *τ*V = [5*,* 200], and *i* = [0*,* 90]. Since the SEDs in the CLUMPY library have been normalized, an additional scaling factor needs to be considered as a new parameter. The prior for this parameter is assumed to be uniform in the log space: log(scaleAGN*/*erg s−1) = [44, 50].

Finally, the linear combination of the pure starburst and pure AGN models is considered as an additional new model (noted as "SB+AGN" hereafter). The assumed priors are the same ones as above. As discussed in Section 2.2, the model SEDs in the two libraries have been used to train two groups of ANNs, respectively. The trained ANNs are used as substitutions of the original models, and the models can be evaluated continuously in the whole parameter space. Since both the starburst and the AGN model used here are not extended to the X-ray range, in this paper we mainly focus on the IR range (i.e., 1–1000*μ*m) of the SEDs. The X-ray data for galaxies in the HLIRG sample have also been provided by Ruiz et al. (2010). However, it is very hard to construct a self-consistent SED model that is able to reproduce the whole SED covering such a wide range of wavelengths.

### *4.2.2. Model Comparison*

The Bayesian evidence represents a practical implementation of the Occam's razor principle. A complex model with more parameters has a lower Bayesian evidence unless it provides a significantly better fitting to the observations. As mentioned above, in this paper we consider three different models: "SB," "AGN," and "SB+AGN." They have 5, 7, and 12 parameters, respectively. In Table 1, we present the Bayesian evidences of the three models for HLIRGs in class A and class B as defined by Ruiz et al. (2010). Since the Bayesian evidence for different models cover a very wide range, we use ln(evmodel) instead of the evidence itself.

As shown in Table 1, the "SB+AGN" model has the highest Bayesian evidence for all galaxies in the HLIRG sample, although it has the largest number of parameters. So, the "SB+AGN" model provides a much better fitting to all of the HLIRGs, which means starburst and AGN activities are probably ongoing together in these galaxies. On the other hand, for most class A HLIRGs, the pure "AGN" model has higher

**Figure 6.** Best-fit (or MAP) model SEDs for class A HLIRGs obtained from sampling the "SB+AGN" model, which has the highest Bayesian evidence among the three models considered. The dotted, dashed, and solid lines represent the starburst component, AGN component, and total, respectively.

**Figure 7.** Similar to Figure 6, but for class B HLIRGs.

Bayesian evidence than a pure "SB," while for most class B HLIRGs the pure "SB" model has higher Bayesian evidence than a pure "AGN" model. These results imply that class A HLIRGs are dominated by AGNs while class B HLIRGs are dominated by starbursts, although both starbursts and AGNs are present in all cases.

### *4.2.3. Parameter Estimation*

In Figures 6 and 7, we show the best fit, i.e., the maximum a posteriori (MAP) model SEDs found during sampling parameter space of the "SB+AGN" model, which has the highest Bayesian evidence among the three models considered. Commonly, the values of parameters corresponding to these best fit models are taken as the best estimation of parameters. However, the Bayesian analysis method has the advantage of providing detailed posterior distributions for all parameters, which represent our full knowledge about these parameters given the priors and new observations. The best expectations and uncertainties about all parameters can be deduced from these possibility distributions.

In Figure 8, we show the posterior PDFs of all parameters of the "SB+AGN" model for IRAS18216+6418. Given the very limited observations and the large number of parameters, it is clear that not all of the parameters can be well constrained. From the detailed posterior PDFs of parameters, it is much easier to find out if a parameter is well constrained or not. For

**Figure 8.** One-dimensional marginal posterior probability density functions of the basic and derived parameters of the "SB+AGN" model for IRAS18216+6418. It is clear that some parameters are poorly constrained. However, the derived IR luminosities are well constrained.

example, the basic parameters *R*, *L*tot of the starburst component and *σ*, *q* of the AGN component are well constrained. The derived parameters log(*L*IR SB), log(*L*IR AGN)*,* and log(*L*IR TOT) are nicely constrained.

Apparently, for a large number of galaxies in a sample, it is not possible to plot such PDFs for all of them. It would be more convenient to use a summary statistics to give a good estimate for a parameter and its spread. Here, we use the median and percentile statistics. The median is found by first sorting all values in ascending order, then taking the element in the middle so that half of all points are below the median and the other half above it. The lower and upper quartiles are the values below which 25% and 75% of points fall, respectively. This statistic is much better than the more frequently used mean and standard deviation statistics when the distribution of PDF is asymmetrically skewed or multimodal.

### *4.2.4. Relations between Starburst and AGN Parameters*

In Tables 2 and 3, we present the estimated starburst and AGN parameters for class A and B HLIRGs by employing the "SB+AGN" model. With the estimated starburst and AGN parameters of these HLIRGs, it would be interesting to explore some possible relations between these parameters, especially those between starbursts and AGNs. However, with the very limited observations not all parameters can be well constrained. In Figure 9, we present some relations between the starburst and AGN parameters that are relatively better constrained.

Figure 9(a) shows the IR luminosity of starbursts and AGNs for all HLIRGs in the sample. As shown clearly, the IR luminosity of most class A HLIRGs is dominated by AGNs, while the IR luminosity of class B HLIRGs is dominated by starbursts. This is consistent with the conclusions drawn according to the Bayesian evidences as shown in Section 4.2.2. Ruiz et al. (2010) classified the HLIRGs in their sample into class A or B according to the shape of their SEDs. So, our results show that the class A and B HLIRGs essentially differ in their dominating emission source.

Figure 9(b) shows the relation between the IR luminosity of AGNs and the fraction of OB stars in the starburst region. The figure shows an anti-correlation between the fraction of OB

**Table 2** The Estimated Starburst Parameters and Corresponding Uncertainties for Classes A and B HLIRGs by Employing the "SB+AGN" Model

| Source | R | fOB | Ltot | Av | log(nhs) | log(LIR SB) |
| --- | --- | --- | --- | --- | --- | --- |
|  | (kpc) |  | (L ) | (mag) | (cm−3) | (erg s−1) |
| Class A HLIRG |  |  |  |  |  |  |
| PG1206+459 | 6.99+3.07 −2.14 | 0.72+0.13 −0.14 | 12.19+0.73 −0.99 | 67.83+34.71 −31.51 | 3.00+0.48 −0.48 | 46.25+0.46 −0.59 |
| PG1247+267 | 7.66+3.38 −3.25 | 0.69+0.15 −0.14 | 11.98+0.95 −0.89 | 69.44+34.67 −34.21 | 3.01+0.47 −0.49 | 46.13+0.69 −0.60 |
| IRASF12509+3122 | 7.50+2.97 −2.75 | 0.67+0.15 −0.13 | 12.59+0.49 −0.98 | 47.93+40.44 −27.90 | 3.01+0.47 −0.47 | 46.47+0.31 −0.60 |
| IRAS14026+4341 | 5.28+2.39 −1.27 | 0.62+0.18 −0.13 | 12.16+0.55 −0.94 | 33.03+46.58 −14.07 | 3.01+0.48 −0.46 | 46.18+0.17 −0.52 |
| IRASF14218+3845 | 6.47+2.89 −2.25 | 0.63+0.17 −0.13 | 12.94+0.30 −1.13 | 63.27+36.66 −27.43 | 2.99+0.49 −0.47 | 46.69+0.17 −0.59 |
| IRAS16347+7037 | 3.58+2.89 −1.19 | 0.71+0.13 −0.14 | 13.42+0.19 −1.02 | 69.83+22.19 −23.45 | 2.91+0.49 −0.43 | 47.08+0.11 −0.57 |
| IRAS18216+6418 | 5.08+1.09 −0.89 | 0.54+0.22 −0.10 | 12.78+0.10 −0.12 | 60.16+48.19 −26.65 | 3.07+0.45 −0.48 | 46.41+0.11 −0.08 |
| Class B HLIRG |  |  |  |  |  |  |
| IRASF00235+1024 | 5.02+1.88 −1.33 | 0.86+0.07 −0.12 | 12.98+0.06 −0.25 | 59.78+35.62 −20.74 | 2.96+0.50 −0.47 | 46.64+0.06 −0.14 |
| IRAS07380−2342 | 8.32+1.13 −1.36 | 0.88+0.09 −0.13 | 12.82+0.06 −0.06 | 47.66+31.68 −15.95 | 3.25+0.46 −0.55 | 46.53+0.05 −0.05 |
| IRAS00182−7112 | 2.84+1.54 −0.84 | 0.74+0.12 −0.13 | 12.86+0.06 −0.07 | 84.62+14.86 −14.57 | 2.95+0.48 −0.46 | 46.46+0.03 −0.03 |
| IRAS09104+4109 | 1.21+2.88 −0.25 | 0.91+0.03 −0.06 | 13.11+0.05 −0.07 | 21.13+12.85 −7.49 | 3.42+0.37 −0.55 | 46.70+0.06 −0.04 |
| IRAS12514+1027 | 3.89+1.20 −1.07 | 0.83+0.09 −0.15 | 12.64+0.05 −0.07 | 49.66+17.00 −11.59 | 3.07+0.47 −0.50 | 46.30+0.04 −0.07 |
| IRASF15307+3252 | 7.02+1.33 −1.44 | 0.72+0.10 −0.09 | 13.47+0.05 −0.09 | 27.28+21.99 −8.87 | 3.11+0.45 −0.50 | 47.12+0.04 −0.09 |

**Note.** The median and percentile statistics are used to the obtain the best estimation of a parameter and its upper and lower limits.

|  |  |  | Table 3 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | Similar to Table 2, but for AGN Parameters |  |  |  |  |
| Source | σ | Y | N | q | τV | i | log(LIR AGN) |
|  |  |  |  |  |  |  | (erg s−1) |
| Class A HLIRG |  |  |  |  |  |  |  |
| PG1206+459 | 40.05+12.56 −11.22 | 97.48+44.47 −42.53 | 11.11+5.36 −4.63 | 1.85+0.76 −0.66 | 111.98+37.44 −36.20 | 31.94+22.61 −16.85 | 47.31+0.06 −0.21 |
| PG1247+267 | 34.70+14.38 −10.24 | 93.72+45.77 −43.87 | 9.72+5.54 −4.18 | 3.13+0.68 −0.90 | 81.23+48.02 −27.97 | 36.34+20.79 −17.24 | 47.85+0.10 −0.41 |
| IRASF12509+3122 | 41.73+13.82 −11.81 | 99.25+47.14 −45.48 | 12.57+5.26 −5.05 | 2.40+0.87 −0.86 | 97.81+46.83 −44.13 | 41.77+20.61 −19.82 | 46.74+0.11 −0.24 |
| IRAS14026+4341 | 26.93+6.37 −5.43 | 149.83+31.80 −57.93 | 8.17+4.95 −4.57 | 2.23+0.68 −0.41 | 151.33+22.62 −28.80 | 31.04+7.95 −8.56 | 46.45+0.08 −0.22 |
| IRASF14218+3845 | 39.59+14.20 −12.35 | 109.19+43.97 −45.32 | 11.79+5.87 −5.67 | 1.70+1.13 −0.81 | 113.35+41.27 −45.08 | 42.45+21.09 −19.50 | 46.21+0.39 −0.38 |
| IRAS16347+7037 | 23.22+7.87 −4.85 | 133.65+36.68 −49.48 | 3.28+4.31 −1.41 | 2.90+0.71 −0.80 | 80.42+57.14 −33.09 | 45.90+13.40 −15.82 | 47.56+0.04 −0.03 |
| IRAS18216+6418 | 22.75+6.87 −4.27 | 128.43+41.11 −55.38 | 2.27+2.07 −0.75 | 2.26+0.76 −0.40 | 127.57+38.67 −48.49 | 41.81+14.00 −16.55 | 46.63+0.05 −0.09 |
| Class B HLIRG |  |  |  |  |  |  |  |
| IRASF00235+1024 | 47.49+13.68 −14.45 | 99.18+46.82 −45.71 | 13.55+5.06 −5.49 | 2.72+0.94 −1.20 | 100.74+47.47 −45.73 | 42.04+22.69 −20.55 | 45.36+0.17 −0.17 |
| IRAS07380−2342 | 34.96+7.86 −3.97 | 178.88+16.22 −93.19 | 18.85+2.71 −9.16 | 0.65+2.91 −0.59 | 58.29+126.95 −52.79 | 15.56+8.71 −5.52 | 46.39+0.20 −0.24 |
| IRAS00182−7112 | 45.54+11.20 −11.03 | 109.72+44.76 −48.92 | 12.53+4.94 −4.97 | 2.77+0.82 −1.04 | 27.35+81.14 −15.69 | 30.07+19.92 −14.24 | 45.67+0.13 −0.22 |
| IRAS09104+4109 | 43.17+11.33 −8.95 | 96.12+54.40 −57.72 | 14.38+4.72 −5.76 | 2.75+0.82 −1.25 | 39.52+70.69 −31.15 | 28.05+19.02 −12.07 | 45.95+0.19 −0.37 |
| IRAS12514+1027 | 42.50+7.13 −7.84 | 124.10+50.15 −75.66 | 6.08+4.86 −1.84 | 2.90+0.72 −1.19 | 29.47+123.27 −23.40 | 21.00+9.78 −8.41 | 45.96+0.09 −0.07 |
| IRASF15307+3252 | 48.28+12.51 −13.43 | 91.14+47.50 −42.15 | 13.22+5.01 −5.36 | 2.69+0.88 −1.12 | 94.57+48.90 −43.55 | 35.07+23.06 −17.97 | 46.14+0.27 −0.43 |

stars in the starburst region and the IR luminosity of AGNs in the center. This implies that the starburst in class B HLIRGs is younger than that in class A HLIRGs. On the other hand, Figure 9(c) also shows an anti-correlation between the optical depth of clumps in AGN torus and the fraction of OB stars in the starburst region. This may imply that the AGN torus in class A HLIRGs is more dusty than that in class B HLIRGs. Furthermore, the results in Figure 9(d) show that the starburst region in class B HLIRGs seems more compact than that in class A HLIRGs.

## 5. SUMMARY

Dust-obscured starburst–AGN composite galaxies, such as ULIRGs and HLIRGs, represent important phases in the formation and evolution of galaxies. It is still very challenging to understand the nature of these interesting but complex galaxies from their SEDs. This can be achieved from the interplay between the modeling and fitting of their SEDs. However, a self-consistent multi-wavelength SED model for such complex systems must contain many parameters, and can only be established step by step. Therefore, a flexible, efficient, and robust SED-fitting tool is necessary. In light of this, we developed a suite of general purpose programs, called BayeSED, for Bayesian analysis of SEDs. The PCA and ANN techniques are employed to allow accurate and efficient generation of model SEDs. Meanwhile, the state-of-art Bayesian inference tool, MultiNest, is interfaced with ANN to allow highly efficient sampling of posterior distributions and calculation of Bayesian evidence.

As a demonstration, we apply this code to an HLIRG sample. By employing three models, we present a complete Bayesian analysis of their SEDs, including model comparison and parameter estimation. According to the computed Bayesian

**Figure 9.** Some relations between starburst and AGN parameters. The red solid and green dashed error bars are results for class A and B HLIRGs, respectively. (a) The IR luminosity of starbursts and AGNs for all HLIRGs in the sample. The black solid line represents the position where the IR luminosity of starbursts and AGNs is equal. (b) The relation between the IR luminosity of AGNs and the fraction of OB stars in the starburst region. (c) The relation between the optical depth of clumps in the AGN torus and the fraction of OB stars in the starburst region. (d) The relation between the IR luminosity of AGNs and the size of the starburst region. (A color version of this figure is available in the online journal.)

evidence of different models and the estimated IR luminosity of starbursts and AGNs, we found that the class A and B HLIRGs as defined by Ruiz et al. (2010) essentially differ in their dominating emission source. We also found some relations between the estimated starburst and AGN parameters. For example, the AGN torus of the HLIRGs dominated by AGNs tends to be more dusty than that of HLIRGs dominated by starbursts. The starburst region of the HLIRGs dominated by starbursts tends to be more compact and has a higher fraction of OB stars than that of HLIRGs dominated by AGNs.

These results are understandable in the context of a galaxy merger driving starburst and delayed AGN activity (Genzel et al. 1998; Sanders et al. 1988; Kauffmann & Haehnelt 2000; Di Matteo et al. 2005; Springel et al. 2005; Hopkins et al. 2006, 2008). There may be an evolution path from class B HLIRGs to class A HLIRGs. The class B HLIRGs may represent the stage where a powerful AGN has already been triggered but still does not outshine the starbursts, while in the state represented by class A HLIRGs, the powerful AGN in the center dominates the output of energy. However, the sample studied here is still very small. Further studies based on more complete samples of HLIRGs and more theoretical models are needed to verify this hypothesis.

Generally, we believe BayeSED can be a useful tool for understanding the nature of complex systems, such as dustobscured starburst–AGN composite galaxies, from decoding their SEDs. In future works, we will apply this code to other larger samples to explore the interplay between starburst and AGN activity in these interesting galaxies. There is still no well-established SED models specifically for starburst–AGN composite galaxies. It would be interesting to explore if a selfconsistent SED model specifically for composite galaxies can have higher Bayesian evidence than a linear combination of starburst+AGN models.

We thank the anonymous referee for his*/*her valuable comments which helped to improve the paper. We thank A. Asensio Ramos for sending their PCA and ANN routines to be used as reference for our work. This work is supported by the National Natural Science Foundation of China (grant Nos. 11033008 and 11103072), and the Chinese Academy of Sciences (grant No. KJCX2-YW-T24).

## REFERENCES

- Almeida, C., Baugh, C. M., Lacey, C. G., et al. 2010, MNRAS, 402, 544
- Andreon, S., Gargiulo, G., Longo, G., Tagliaferri, R., & Capuano, N. 2000, MNRAS, 319, 700
- Asensio Ramos, A., & Ramos Almeida, C. 2009, ApJ, 696, 2075
- Auld, T., Bridges, M., & Hobson, M. P. 2008, MNRAS, 387, 1575
- Baes, M., Davies, J. I., Dejonghe, H., et al. 2003, MNRAS, 343, 1081
- Bailer-Jones, C. A. L. 2011, MNRAS, 411, 435
- Ben´ıtez, N. 2000, ApJ, 536, 571
- Bertin, E., & Arnouts, S. 1996, A&AS, 117, 393
- Bruzual, A. G. 1983, ApJ, 273, 105
- Bruzual, G., & Charlot, S. 2003, MNRAS, 344, 1000
- Budavari, T., Wild, V., Szalay, A. S., Dobos, L., & Yip, C.-W. 2009, ´ MNRAS, 394, 1496
- Carballo, R., Gonzalez-Serrano, J. I., Benn, C. R., & Jim ´ enez-Luj ´ an, F. ´ 2008, MNRAS, 391, 369
- Chakrabarti, S., & Whitney, B. A. 2009, ApJ, 690, 1432
- Collister, A. A., & Lahav, O. 2004, PASP, 116, 345
- Conroy, C., Gunn, J. E., & White, M. 2009, ApJ, 699, 486
- Cybenko, G. 1989, Math. Control Signals Syst., 2, 303
- da Cunha, E., Charlot, S., & Elbaz, D. 2008, MNRAS, 388, 1595
- Devriendt, J. E. G., Guiderdoni, B., & Sadat, R. 1999, A&A, 350, 381
- Di Matteo, T., Springel, V., & Hernquist, L. 2005, Nature, 433, 604
- Dullemond, C. P., & Dominik, C. 2004, A&A, 417, 159
- Efstathiou, A., Rowan-Robinson, M., & Siebenmorgen, R. 2000, MNRAS, 313, 734
- Feldmann, R., Carollo, C. M., Porciani, C., et al. 2006, MNRAS, 372, 565
- Feroz, F., & Hobson, M. P. 2008, MNRAS, 384, 449
- Feroz, F., Hobson, M. P., & Bridges, M. 2009, MNRAS, 398, 1601
- Fioc, M., & Rocca-Volmerange, B. 1997, A&A, 326, 950
- Firth, A. E., Lahav, O., & Somerville, R. S. 2003, MNRAS, 339, 1195
- Francis, P. J., Hewett, P. C., Foltz, C. B., & Chaffee, F. H. 1992, ApJ, 398, 476
- Genzel, R., Lutz, D., Sturm, E., et al. 1998, ApJ, 498, 579
- Glazebrook, K., Offer, A. R., & Deeley, K. 1998, ApJ, 492, 98
- Granato, G. L., Lacey, C. G., Silva, L., et al. 2000, ApJ, 542, 710
- Groves, B., Dopita, M. A., Sutherland, R. S., et al. 2008, ApJS, 176, 438
- Han, Z., Podsiadlowski, P., & Lynas-Gray, A. E. 2007, MNRAS, 380, 1098
- Haykin, S. 1999, Neural Networks: A Comprehensive Foundation (2nd ed.; Upper Saddle River, NJ: Prentice Hall)
- Hopkins, P. F., Hernquist, L., Cox, T. J., & Keres, D. 2008, ˇ ApJS, 175, 356
- Hopkins, P. F., Hernquist, L., Cox, T. J., et al. 2006, ApJS, 163, 1
- Hornik, K. 1991, Neural Netw., 4, 251
- Jonsson, P. 2006, MNRAS, 372, 2
- Kauffmann, G., & Haehnelt, M. 2000, MNRAS, 311, 576
- Kauffmann, G., Heckman, T. M., White, S. D. M., et al. 2003, MNRAS, 341, 33
- Lacey, C., & Silk, J. 1991, ApJ, 381, 14
- Lahav, O., Naim, A., Sodre, L., Jr., & Storrie-Lombardi, M. C. 1996, MNRAS, ´ 283, 207
- Larson, R. B., & Tinsley, B. M. 1978, ApJ, 219, 46
- Leitherer, C., Schaerer, D., Goldader, J. D., et al. 1999, ApJS, 123, 3
- Maraston, C. 2005, MNRAS, 362, 799
- Mukherjee, P., Parkinson, D., & Liddle, A. R. 2006, ApJ, 638, L51
- Nenkova, M., Sirocky, M. M., Ivezic,´ Z., & Elitzur, M. 2008a, ˇ ApJ, 685, 147
- Nenkova, M., Sirocky, M. M., Nikutta, R., Ivezic,´ Z., & Elitzur, M. 2008b, ˇ ApJ, 685, 160
- Noll, S., Burgarella, D., Giovannoli, E., et al. 2009, A&A, 507, 1793
- Rowan-Robinson, M. 2000, MNRAS, 316, 885
- Ruiz, A., Carrera, F. J., & Panessa, F. 2007, A&A, 471, 775
- Ruiz, A., Miniutti, G., Panessa, F., & Carrera, F. J. 2010, A&A, 515, A99
- Salim, S., Rich, R. M., Charlot, S., et al. 2007, ApJS, 173, 267
- Sanders, D. B., Soifer, B. T., Elias, J. H., et al. 1988, ApJ, 325, 74
- Searle, L., Sargent, W. L. W., & Bagnuolo, W. G. 1973, ApJ, 179, 427
- Shaw, J. R., Bridges, M., & Hobson, M. P. 2007, MNRAS, 378, 1365
- Siebenmorgen, R., & Krugel, E. 2007, ¨ A&A, 461, 445
- Silva, L., Granato, G. L., Bressan, A., & Danese, L. 1998, ApJ, 509, 103
- Silva, L., Schurer, A., Granato, G. L., et al. 2011, MNRAS, 410, 2043
- Skilling, J. 2004, in AIP Conf. Proc. 735, Bayesian Inference and Maximum Entropy Methods in Science and Engineering, ed. R. Fischer, R. Preuss, & U. V. Toussaint (Melville, NY: AIP), 395
- Springel, V., Di Matteo, T., & Hernquist, L. 2005, MNRAS, 361, 776
- Tinsley, B. M. 1972, A&A, 20, 383
- Tinsley, B. M. 1978, ApJ, 222, 14
- Trotta, R. 2008, Contemp. Phys., 49, 71
- Tuffs, R. J., Popescu, C. C., Volk, H. J., Kylafis, N. D., & Dopita, M. A. ¨ 2004, A&A, 419, 821
- Vanzella, E., Cristiani, S., Fontana, A., et al. 2004, A&A, 423, 761
- Walcher, J., Groves, B., Budavari, T., & Dale, D. 2011, ´ Ap&SS, 331, 1
- White, S. D. M., & Frenk, C. S. 1991, ApJ, 379, 52
- White, S. D. M., & Rees, M. J. 1978, MNRAS, 183, 341
- Wild, V., & Hewett, P. C. 2005, MNRAS, 358, 1083
- Wild, V., Kauffmann, G., Heckman, T., et al. 2007, MNRAS, 381, 543
- Zhang, F., Han, Z., Li, L., & Hurley, J. R. 2005a, MNRAS, 357, 1088
- Zhang, F., Li, L., & Han, Z. 2005b, MNRAS, 364, 503

