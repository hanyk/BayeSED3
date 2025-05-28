# A Comprehensive Bayesian Discrimination of the Simple Stellar Population Model, Star Formation History, and Dust Attenuation Law in the Spectral Energy Distribution Modeling of Galaxies

Yunkun Han1,2,3 and Zhanwen Han1,2,3

1 Yunnan Observatories, Chinese Academy of Sciences, 396 Yangfangwang, Guandu District, Kunming, 650216, People's Republic of China; hanyk@ynao.ac.cn 2 Center for Astronomical Mega-Science, Chinese Academy of Sciences, 20A Datun Road, Chaoyang District, Beijing, 100012, People's Republic of China

zhanwenhan@ynao.ac.cn 3 Key Laboratory for the Structure and Evolution of Celestial Objects, Chinese Academy of Sciences, 396 Yangfangwang, Guandu District, Kunming, 650216, People's Republic of China

Received 2018 May 11; revised 2018 October 29; accepted 2018 November 7; published 2018 December 26

# Abstract

When modeling and interpreting the spectral energy distributions (SEDs) of galaxies, the simple stellar population (SSP) model, star formation history (SFH), and dust attenuation law (DAL) are three of the most important components. However, each of them carries significant uncertainties that have seriously limited our ability to reliably recover the physical properties of galaxies from the analysis of their SEDs. In this paper, we present a Bayesian framework to deal with these uncertain components simultaneously. Based on the Bayesian evidence, a quantitative implement of the principle of Occam's razor, the method allows a more objective and quantitative discrimination among the different assumptions about these uncertain components. With a Ks-selected sample of 5467 low-redshift (mostly with z 1) galaxies in the COSMOS/UltraVISTA field and classified into passively evolving galaxies (PEGs) and star-forming galaxies (SFGs) with the UVJ diagram, we present a Bayesian discrimination of a set of 16 SSP models from five research groups (BC03 and CB07, M05, GALEV, Yunnan-II, BPASS V2.0), five forms of SFH (Burst, Constant, Exp-dec, Exp-inc, Delayed-τ), and four kinds of DAL (Calzetti law, MW, LMC, SMC). We show that the results obtained with the method are either obvious or understandable in the context of stellar/galaxy physics. We conclude that the Bayesian model comparison method, especially that for a sample of galaxies, is very useful for discriminating the different assumptions in the SED modeling of galaxies. The new version of the BayeSED code, which is used in this work, is publicly available at https://bitbucket.org/hanyk/bayesed/.

Key words: galaxies: fundamental parameters – galaxies: statistics – galaxies: stellar content – methods: data analysis – methods: statistical

# 1. Introduction

Understanding the formation and evolution of galaxies is one of the biggest challenges in modern astrophysics (Mo et al. 2010; De Lucia et al. 2014; Somerville & Davé 2015; Naab & Ostriker 2017). Various complex and not well understood baryonic processes, such as the formation and evolution of stars (Kennicutt 1998; McKee & Ostriker 2007; Heber 2009; Kennicutt & Evans 2012; Duchêne & Kraus 2013), the accretion and feedback of supermassive black holes (Melia & Falcke 2001; Merloni 2004; Fabian 2012; Kormendy & Ho 2013), and the chemical enrichment of interstellar medium (ISM; McKee & Ostriker 1977; Spitzer 1978; Li & Greenberg 1997; Draine 2003; De Lucia et al. 2004; Scannapieco et al. 2005; Draine 2010; Nomoto et al. 2013), are involved. What makes the problem even more challenging is the fact that all of these complex baryonic processes are also tightly entangled (Hamann & Ferland 1993; Timmes et al. 1995; Ferrarese & Merritt 2000; Hopkins et al. 2008a, 2008b; Marulli et al. 2008; Bonoli et al. 2009; Heckman & Best 2014). It is often not trivial to decouple any one of them from the others to allow a complete independent study. To disentangle these complex and highly related baryonic processes involved in the formation and evolution of galaxies, we need to make use of all available sources of information (Bartos & Kowalski 2017).

Despite the recent progress in the detection of cosmic rays (Murase et al. 2008; Adriani et al. 2009), neutrinos (Ahmad et al. 2002; Becker 2008), and gravitational waves (Abbott et al. 2016, 2017), electromagnetic emissions are still the main source of information for our understanding of galaxies. All of those complex baryonic processes involved in the formation and evolution of galaxies leave their imprint on the spectral energy distributions (SEDs) of the electromagnetic emissions from galaxies. In recent decades, large photometric and spectroscopic surveys, such as the Two Micron All Sky Survey (2MASS; Skrutskie et al. 2006), the Sloan Digital Sky Survey (SDSS; York et al. 2000), COSMOS (Scoville et al. 2007), UltraVISTA (McCracken et al. 2012; Muzzin et al. 2013b), CANDELS (Grogin et al. 2011; Koekemoer et al. 2011), and 3D-HST (Brammer et al. 2012; Skelton et al. 2014), have provided us with rich multiwavelength observational data for millions of galaxies covering a large range of redshift. These massive data sets present a tremendous opportunity and challenge for us to understand the formation and evolution of galaxies from the analysis of their SEDs.

The process of solving the inverse problem of deriving the physical properties of galaxies from their observational SEDs is known as SED fitting (Bolzonella et al. 2000; Massarotti et al. 2001; Ilbert et al. 2006; Salim et al. 2007; Walcher et al. 2011). In principle, an SED fitting method that is capable of effectively extracting all the information encoded in these SEDs of galaxies would allow us to fully understand their physical properties. Traditionally, SED fitting is considered an optimization problem, where some χ2 minimization techniques are employed to find the best-fit model and corresponding value of parameters (Arnouts et al. 1999; Bolzonella et al. 2000; Cid Fernandes et al. 2005; Koleva et al. 2009; Kriek et al. 2009; Sawicki 2012; Gomes & Papaderos 2017). However, due to the large number of often degenerated free parameters, it would be more reasonable to consider the problem of SED fitting as a Bayesian inference problem (Benítez 2000; Kauffmann et al. 2003). Recently, it has become quite popular to employ the Markov chain Monte Carlo (MCMC) sampling method to efficiently obtain not only the best-fit results but also the detailed posterior probability distribution of all parameters (Benítez 2000; Kauffmann et al. 2003; Acquaviva et al. 2011; Serra et al. 2011; Pirzkal et al. 2012; Johnson et al. 2013; Calistro Rivera et al. 2016; Leja et al. 2017).

Despite the popularity of the Bayesian parameter estimation method, the Bayesian model comparison/selection method, which is based on the computation of the Bayesian evidences of different models, has not yet been widely used in the field of SED fitting of galaxies. The Bayesian evidence quantitatively implements the principle of Occam's razor, according to which a simpler model with compact parameter space should be preferred over a more complicated one with a large fraction of useless parameter space, unless the latter can provide a significantly better explanation of the data (MacKay 1992, 2003). Based on the Bayesian framework initially introduced by Suyu et al. (2006) for solving the gravitational lensing problem, Dye (2008) presented an approach to determine the star formation history (SFH) of galaxies from multiband photometry, where the most probable model of SFH is obtained by the maximization of the Bayesian evidence. In Han & Han (2012), we have presented a Bayesian model comparison for the SED modeling of hyperluminous infrared galaxies (HLIRGs), where the multimodal-nested sampling (MultiNest) techniques (Feroz & Hobson 2008; Feroz et al. 2009, 2013) have been employed to allow a more efficient calculation of the Bayesian evidence of different SED models. Salmon et al. (2016) presented a Bayesian approach based on Bayesian evidence to check the universality of the dust attenuation law (DAL). For a sample of z ∼ 1.5–3 galaxies from CANDELS with rest-frame UV to near-IR photometric data, they found that some galaxies show strong Bayesian evidence in favor of one particular DAL over another, and this preference is consistent with their observed distribution on the infrared excess (IRX) and UV slope (β) plane. Dries et al. (2016, 2018) presented a hierarchical Bayesian approach to reconstructing the initial mass function (IMF) in simple and composite stellar populations (SSPs and CSPs), where the Bayesian evidence is employed to compare different choices of the IMF prior parameters and to determine the number of SSPs required in CSPs by the maximization of the Bayesian evidence.

In Han & Han (2014), with the first publicly available version of our BayeSED code, we have presented a Bayesian model comparison between two of the most widely used stellar population synthesis (SPS) model (Bruzual & Charlot 2003, hereafter BC03; Maraston 2005, hereafter M05) for the first time. With the distribution of Bayes's factor (the ratio of Bayesian evidence) for a Ks-selected sample of galaxies in the COSMOS/UltraVISTA field (Muzzin et al. 2013b), we found that the BC03 model statistically has larger Bayesian evidence than the M05 model. In Han & Han (2014), the reliability of the BayeSED code for physical parameter estimation has also been systematically tested. The internal consistency of the code has

been tested with a mock sample of galaxies, while its external consistency has been tested by the comparison with the results of the widely used FAST code (Kriek et al. 2009). However, the work still has many limitations. For example, a fixed exponentially declining SFH and the Calzetti et al. (2000) DAL have been assumed to be universal for all galaxies. However, from either an observational or a theoretical point of view, the form of SFH and the DAL of different galaxies are not likely to be the same (Witt & Gordon 2000; Maraston et al. 2010; Wuyts et al. 2011; Simha et al. 2014). Besides, the numerous uncertainties carried by almost all the components involved in the process of SPS (Conroy et al. 2009, 2010; Conroy & Gunn 2010; Conroy 2013) have resulted in the diversity of SPS models. Except for the BC03 and M05 model, there are numerous SPS models from many other groups, which have employed different stellar evolution tracks, stellar spectral libraries, IMFs, and/or synthesis methods (Buzzoni 1989; Fioc & Rocca-Volmerange 1997; Leitherer et al. 1999; Zhang et al. 2005a; Conroy et al. 2009; Eldridge & Stanway 2009; Kotulla et al. 2009; Vazdekis et al. 2010).

As three of the most important components in modeling and interpreting the SEDs of galaxies, the SSP model, SFH, and DAL all carry significant uncertainties. The existence of these uncertainties would seriously limit the possibility of reliably recovering the physical properties of galaxies from the analysis of their SEDs. Besides, it is not easy to reasonably quantify the impact of any one of them without mention of the other two. Hence, it is very important to find a unitized method to quantify the propagation of these uncertainties into the estimation of the physical parameters of galaxies and to quantitatively discriminate their different choices. In this work, we present a unitized Bayesian framework to deal with all of these uncertain components simultaneously.

This paper is structured as follows. In Section 2, we introduce the new SED modeling module of the BayeSED code, including the CSP synthesis method in Section 2.1, the SED modeling of an SSP in Section 2.2, the form of SFH in Section 2.3, and the DAL in Section 2.4. We then briefly review the Bayesian inference methods in Section 3, including the Bayesian parameter estimation in Section 3.1 and the Bayesian model comparison in Section 3.2. In the next two sections, we introduce our new methods for calculating the Bayesian evidence and associated Occam factor for the SED modeling of an individual galaxy (Section 4) and a sample of galaxies (Section 5). In Section 6, we present the results of applying our new methods to a Ks-selected sample in the COSMOS/UltraVISTA field for discriminating among the different choices of SSP model, SFH, and DAL when modeling the SEDs of galaxies. Some discussions about the different SSPs, SFHs, and DALs are presented in Section 7. Finally, a summary of our new methods and results is presented in Section 8.

# 2. The Spectral Energy Distribution Modeling of Galaxies in BayeSED

For a detailed Bayesian analysis of the observed multiwavelength SED of a galaxy, the modeling of its SED is often the most computationally demanding. Hence, the efficiency of the whole Bayesian analysis process strongly depends on the efficiency of the SED modeling method. In the previous version of BayeSED (Han & Han 2012, 2014), some machine learning methods, such as the artificial neural network (ANN) and K-nearest neighbor searching (KNN) algorithm, have been employed. After the training with a precomputed library of model SEDs, the machine learning methods allow a very efficient computation of a massive number of model SEDs during the sampling of an often high-dimensional parameter space of an SED model. By using the machine learning methods, very different SED models can be easily integrated into the BayeSED code with the same procedure. Therefore, the BayeSED code can be easily extended to solve the SED fitting problem in different fields.

Despite these interesting benefits, the machine-learningbased SED modeling methods are not so convenient during the development of an SED model, since any modification to the model components requires a new and often time-consuming machine learning procedure. To explore the effects of assuming different SSP models, SFHs, and DALs in the SED modeling of galaxies, we have built an SED modeling module into the new version (V2.0) of our BayeSED code (see the flowchart in Figure 1). Currently, we do not intend to build a very sophisticated SED modeling procedure into the BayeSED code. To be consistent with the principle of Occam's razor, according to which "entities should not be multiplied unnecessarily," we prefer to start with a simple but still useful SED modeling procedure and gradually increase its complexity.

# 2.1. Composite Stellar Population Synthesis

The SED of a galaxy as a complex stellar system can be obtained with composite SPS as

$$L_{\lambda}(t)=\int_{0}^{t}\,dt^{\prime}\,\psi(t-t^{\prime})\,S_{\lambda}[t^{\prime},Z(t-t^{\prime})]\,T_{\lambda}^{\rm LSM}(t,t^{\prime})\tag{1}$$

$$=T_{\lambda}^{\rm ISM}\int_{0}^{t}dt^{\prime}\;\psi(t-t^{\prime})\,S_{\lambda}[t^{\prime},Z_{0}],\tag{2}$$

where *y*( ) *t t* - ¢ is the star formation rate (SFR) at time t − *t*¢ (SFH), *Sl* [ ( )] *t Zt t* ¢, - ¢ is the luminosity emitted per unit wavelength per unit mass by an SSP of age t′ and chemical composition Z(t − t′), and *T tt* , ISM ¢ *l* ( ) is the transmission function of the ISM. We assume a time-independent metallicity Z0 and DAL *T*ISM *l* for the entire composite population.

# 2.2. The SED Modeling of a Simple Stellar Population

According to the most widely used isochrone synthesis approach (Charlot & Bruzual 1991; Bruzual & Charlot 1993, 2003), the SED of an SSP is obtained as

$$S_{\lambda}(t^{\prime},Z)$$
 
$$=\int_{m_{\rm low}}^{m_{\rm exp}}dm\ \phi(m)f_{\lambda}\left[L_{\rm bol}(m,Z,t^{\prime}),\,T_{\rm eff}(m,Z,t^{\prime}),\,Z\right],\tag{3}$$

where m is the stellar mass, *f*( ) *m* is the stellar IMF with lower and upper mass cutoffs mlow and mup, and *f L mZt T mZt Z* bol eff ,, , ,, , ¢ ¢ *l* [ ( ) ( )] is the SED of a star with bolometric luminosity *L*bol(*mZt* , , ¢), effective temperature *T mZt* , , eff ( )¢ , and metallicity Z. Hence, different choices for any of the IMF, stellar isochrone, and stellar spectral library will result in different SSP models.

Alternatively, the fuel consumption theorem (Renzini & Buzzoni 1986; Maraston 1998, 2005) has been used to allow an easier calculation of the luminosity contribution of the shortlived and often less understood post-main-sequence stellar evolution stages, such as the thermally pulsing asymptotic giant branch (TP-AGB) phase. According to the theorem, the luminosity contribution of each stellar evolutionary phase is proportional to the amount of hydrogen and/or helium (the fuel) burned by nuclear fusion within the stars. It also provides analytical relations between the main-sequence and post-mainsequence stellar evolution, and the SEDs can be obtained using the relations between colors/spectra and bolometric luminosities. There are other approaches to obtain the integrated SED of an SSP, such as the use of empirical spectra of star clusters as templates for SSPs (Bica & Alloin 1986; Bica 1988; Cid Fernandes et al. 2001; Kong et al. 2003) and the employment of the Monte Carlo technique (Zhang et al. 2005a; Han et al. 2007; da Silva et al. 2012; Cerviño 2013).

There are many publicly available SSP models (see http:// www.sedfitting.org/Models.html). In this work, we have selected a set of 16 different SSP models from five groups, including the BC03 (Bruzual & Charlot 2003) and CB07 (Bruzual 2007), M05 (Maraston 2005), GALEV (Kotulla et al. 2009), Yunnan-II (Zhang et al. 2005a), and BPASS V2.0 (Eldridge & Stanway 2009) models. Many SSP models from other research groups (e.g., Buzzoni 1989; Fioc & Rocca-Volmerange 1997; Leitherer et al. 1999; Conroy et al. 2009; Vazdekis et al. 2010), many of which have been widely used in several works, are not included in our list. It is straightforward for us to add all of these SSP models to the new version of the BayeSED code. However, the main purpose of this paper is to demonstrate the Bayesian model comparison method and to evaluate its effectiveness. Hence, we try to randomly select a small set of representative models that are as diverse as possible, although they could be biased to those that are popular, easier to obtain, or more familiar to us. The physical considerations about the effectiveness of the SSP models for the galaxy sample have not been used as the criterion for the selection of them. Actually, they are considered to be equally likely a priori (i.e., before the comparison with data). A summary of the 16 SSP models used in this paper is presented in Table 1. As shown clearly in the table, the SSP models that differ in any model component (Track/Spectral library/IMF/ Binary/Nebular) are treated as different SSP models. In the rest of this section, we present a short description of each chosen SSP model, with a focus on their differences.

#### 2.2.1. BC03 and Updated CB07

The BC03 (Bruzual & Charlot 2003) model is the one most widely used in the literature. It is a good choice for a standard model to compare with. Besides, the isochrone synthesis technique first introduced in this model has been employed by many other more recent models. Hence, the BC03 model is also a good representative of the set of models that have employed similar techniques. We have used the version built with the Padova 1994 evolutionary tracks, the BaSeL 3.1 spectral library, and the IMFs of Chabrier (2003), Kroupa (2001), and Salpeter (1955). The model contains the SED of SSPs with log age yr 5, 10.3 ( )[ = ] and log 2.30, 0.70 ( )[ ] *Z Z* = - . The CB07 (Bruzual 2007) model is very similar to the BC03 model, with the former including an updated prescription (Marigo & Girardi 2007) for the TP-AGB evolution of lowand intermediate-mass stars, which produces much redder near-IR colors for young and intermediate-age stellar populations. However, whether this represents a much better treatment of the TP-AGB phase remains an open issue (Kriek et al. 2010; Zibetti et al. 2013; Capozzi et al. 2016).

#### 2.2.2. M05

The M05 (Maraston 2005) model is also very widely used in many works and often used to be compared with the BC03 model. A main feature of this model lies in its treatment of the post-main-sequence stellar evolution stages, such as TP-AGB, based on the fuel consumption theorem. The contribution of TP-AGB stars is expected to be crucial for modeling the SEDs of young and intermediate-age (0.1–2 Gyr) stellar populations, which predominate the 1.5 *z* 3 redshift range (Maraston 2005; Maraston et al. 2006; Henriques et al. 2011). Except for the different treatment of TP-AGB stars, the M05 model has employed the input stellar evolution tracks/isochrones of Cassisi et al. (1997a, 1997b, 2000), which are different from those used in the BC03 and CB07 models. The public version of the M05 model contains the SED of SSPs with log age yr 3 10.2 ( )[ = ] and log 2.25 0.67 ( )[ ] *Z Z* = - . In this work, we have used the version with a red horizontal branch morphology and the IMF of Kroupa (2001) and Salpeter (1955).

#### 2.2.3. GALEV

The GALEV (GALaxy EVolution) evolutionary synthesis model (Kotulla et al. 2009) has many properties that are in common with the BC03 model. What makes the GALEV model special is its consistent treatment of the chemical evolution of the gas and the spectral evolution of the stellar content. However, to be more easily compared with the SSPs from other groups, we prefer to use the version with metallicity fixed to some specific values, instead of that obtained with a chemically consistent treatment. Actually, we just want to select an SSP model that has nebular emission included, while the GALEV model is the only one that we found to be publicly available and is much easier to obtain. Although the treatment of nebular emission in the GALEV model is relatively simple, it is still useful to test the importance of including nebular emission in the SED model of galaxies. We have used the web interface at http://model.galev.org/model_1.php to generate the SED of SSPs with log age yr 6.6 10.2 ( )[ = ] and log 1.7 0.4 ( )[ *Z Z* = - ]. Both the versions with and without the contribution of nebular emission have been used in this work.

#### 2.2.4. Yunnan-II

The Yunnan models have been built by our binary population synthesis (BPS) team at Yunnan Observatory (Zhang et al. 2004, 2005a, 2005b; Han et al. 2007). The main feature of these models is the consideration of various binary interactions, which is implemented with the help of a Monte Carlo technique. The Yunnan models have employed the Cambridge stellar evolutionary tracks in the form given in the rapid stellar evolution code of Hurley et al. (2000, 2002) as a set of analytic evolution functions fitted to the model tracks of Pols et al. (1998). In this work, we have used the Yunnan-II version (Zhang et al. 2005a) with the BaSeL 2.0 spectral library and the IMF of Miller & Scalo (1979). The model contains the SED of SSPs with log age yr 5.0 10.2 ( )[ = ] and log 2.3 0.18 ( )[ *Z Z* = - ]. To test the importance of considering the effects of binary interactions, both the versions with and without binary interactions have been used in this work.

#### 2.2.5. BPASS

The Binary Population and Spectral Synthesis (BPASS) code is another publicly available population synthesis model that has considered the effects of binary evolution in the SED modeling of stellar populations. Instead of an approximate rapid population synthesis method, detailed stellar evolution models, which are obtained with a custom version of the longestablished Cambridge STARS stellar evolution code, have been used in the code. The authors of the model also try to only use theoretical model spectra with as few empirical inputs as possible in the population syntheses to create synthetic models as pure as possible to compare with observations. In this work, we have used the V2.0 fiducial models, which have assumed a broken power-law IMF with a slope of −1.30 from 0.1 to 0.5*M* and −2.35 from 0.5 to 300 *M*. The model contains the SED of SSPs with log age yr 6.0 10.0 ( )[ = ] and log 1.3 0.30 ( )[ *Z Z* = - ].

The BPASS model is undergoing a rather rapid development. During the writing of this paper, the BPASS team released their V2.1 (Eldridge et al. 2017) and V2.2 (Stanway & Eldridge 2018) models. The BPASS V2.0 model, which is used in this paper, was released in 2015 and has been widely used in many stellar and extragalactic works. However, it was not formally described in detail until the V2.1 data release paper of Eldridge et al. (2017). There are a few refinements in the V2.1 models, but no major changes to the V2.0 results. In Eldridge et al. (2017), the authors also discussed some key caveats and uncertainties in their current models. Especially, they identified several aspects of the old stellar populations (>1 Gyr), such as the binary fraction in lower-mass stars, as problematic in their current model set. In Stanway & Eldridge (2018), the authors stated that some of these issues have been partly addressed in their recently released V2.2 models.

Given the limitations of the BPASS V2.0 model and the improved V2.1 and V2.2 of the same model, it may seem nonsensical to still use the older one. However, in addition to those regarding binary evolution, there are still many other uncertainties involved in the SSP model. Given this, the model will be undergoing an intensive development for a long time, during which the older version of the same model will be rapidly replaced by the newer ones. Actually, many of the models from other groups also have their more updated version (e.g., Bruzual 2011; Maraston & Strömbäck 2011; Zhang et al. 2013). Here we need to point out that it is by no means the aim of this paper to find out the most cutting-edge SSP model. In this paper, we aim at evaluating the effectiveness of applying the Bayesian model comparison method to the SSP models. Hence, we prefer to use the more stable version of those models that have been used for a relatively long time and for which their performance has been known to some extent. Certainly, in the future, we would like to compare these more updated models using the Bayesian methods developed in this paper.

# 2.3. The Form of Star Formation History

Due to its complex formation and evolution history, the detailed SFH of a real galaxy could be arbitrarily complex. However, to derive the physical parameters, such as SFR and stellar mass, from the multiwavelength photometric SED from a very limited number of filter bands, we need to make a priori simple assumptions about its SFH.

The exponentially declining (Exp-dec for short) SFH of the form *t e t y* µ - *t* ( ) , the so-called τ model, is the most widely used assumption. However, some works suggest that it leads to biased estimation of the stellar mass of individual galaxies and the stellar mass functions (Wuyts et al. 2011; Simha et al. 2014). The exponentially increasing (Exp-inc for short) SFH of the form *t et y* µ *t* ( ) , the so-called inverted-τ model (Maraston et al. 2010; Pforr et al. 2012), and the delayed-τ (Delayed for short) model of the form *t te t y* µ - *t* ( ) (Lee et al. 2010) have been suggested to explain the SEDs of high-redshift starforming galaxies. Besides, we also considered the simpler single-burst (Burst for short) and constant SFH for reference. Thus, in total, we have considered five analytical forms of SFHs.

Recently, some authors have suggested some more complicated parameterizations of SFH (Gladders et al. 2013; Abramson et al. 2016; Ciesla et al. 2017; Diemer et al. 2017; Carnall et al. 2018) and physically motivated prescriptions of SFHs drawn from either the hydrodynamic or the semianalytic models of galaxy formation (Finlator et al. 2007; Pacifici et al. 2012; Iyer & Gawiser 2017). Besides, it is also possible to directly employ the nonparametric form of SFH, an approach that has been employed by many works (Heavens et al. 2000; Cid Fernandes et al. 2005; Ocvirk et al. 2006; Tojeiro et al. 2007; Koleva et al. 2009; Díaz-García et al. 2015; Magris et al. 2015; Leja et al. 2017; Zhang et al. 2017). However, the aim of this paper is to evaluate the effectiveness of the Bayesian model comparison method and build a reference for future study, and it is better to start with much simpler forms of SFH. We leave the exploration of these more complicated forms of SFH for future study.

# 2.4. Dust Attenuation Curve

The existence of the ISM (Draine 2010) can significantly change the SED of the stellar populations. For example, the UV–optical starlight can be absorbed by the interstellar dust and reemitted in the infrared. Besides, the UV and ionizing photon flux from the stellar populations can be reduced by the interstellar nebular gas and reemitted as a nebular continuum component and strong emission lines in the optical and infrared. In this paper, we only consider the effect of dust attenuation as a uniform dust screen with different DALs, while leaving the consideration of dust emission for future study.

The DALs of different galaxies are likely to be different owing to different star–dust geometry and/or composition (Witt & Gordon 2000; Reddy et al. 2015; Cullen et al. 2018, 2017b). In this work, we have selected four widely used attenuation curves, including the Calzetti et al. (2000) DAL for starburst galaxies (SB for short) and the MW, LMC, and SMC attenuation laws.4 As for the nebular emission, we have selected the SSP models from GALEV, which has included a self-consistent treatment of this, to test the importance of including nebular emission in the SED modeling of galaxies. We leave the consideration of the physically motivated timedependent attenuation model (Charlot & Fall 2000) and more complicated parameterizations (Witt & Gordon 2000) and the more sophisticated modeling of the nebular emission with the photoionization codes, such as CLOUDY (Ferland et al. 1998, 2013, 2017) and MAPPINGS (Sutherland & Dopita 1993; Groves et al. 2004), for future study.

# 3. Bayesian Inference Methods

In BayeSED, the Bayesian inference methods are employed to interpret the SEDs of galaxies. The base for all these methods is Bayes's theorem. It can be used to solve both the parameter estimation problem and model comparison/selection problems.

# 3.1. Bayesian Parameter Estimation

With the application of Bayes's theorem in the parameter space, the posterior probability of the parameters *q* of a model *M* given a set of observational data *d*, the model *M* itself, and all the other background assumptions *I* is related to the prior probability *p*(∣ ) *q M I*, and the likelihood function *p*(∣ ) () *d MI q q* , , º such that

$$p(\theta|d,M,I)=\frac{p(d|\theta,M,I)p(\theta|M,I)}{p(d|M,I)},\tag{4}$$

where *p*(∣ ) *dM I*, is a normalization factor called Bayesian evidence, or model likelihood. With the joint posterior parameter probability distribution in Equation (4), the marginalized posterior probability distribution for each parameter *qX* can be obtained as

$$p(\theta_{X}|d,M,I)$$
 
$$=\int p(\theta|d,M,I)\mathrm{d}\theta_{1}\cdots\mathrm{d}\theta_{X-1}\mathrm{d}\theta_{X+1}\cdots\mathrm{d}\theta_{N}.\tag{5}$$

The mean, median, or maximum of the marginalized posterior probability distribution can be used as an estimation of the value of a parameter, while the typical width of the distribution can be used as an estimation of the associated uncertainty.

Assuming a Gaussian form of noise, we define the likelihood function for the nth independent wavelength band as

$$\mathcal{L}(\theta)\equiv p(d|\theta,M,I)$$
 
$$=\prod_{i=1}^{n}\ \frac{1}{\sqrt{2\pi}\,\sigma_{i}}\exp\Biggl{(}-\frac{1}{2}\frac{\left(F_{\text{obs},\text{i}}-F_{M(\theta),i}\right)^{2}}{\sigma_{i}^{2}}\Biggr{)},\tag{6}$$

where *F*obs,i and *si* represent the observational flux and associated uncertainty in each band, respectively, and *FM* ( ) *q* ,*i* represents the value of flux for the ith band predicted by the model M, which has a set of free parameters (as indicated by the vector *q*). The uncertainty σi for the ith band is not just the observational error, which is often an underestimation. It is a common practice to additionally consider the potential systematic uncertainties in the observed fluxes and the systematic uncertainties in the employed model itself. Hence, σi should contain three terms such that

$$\sigma^{2}_{i}=\sigma^{2}_{\rm obs,i}+\sigma^{\rm obs,i2}_{\rm sys}+\sigma^{\rm model,i2}_{\rm sys},\tag{7}$$

where *s*obs,i is the purely observational error, sys obs,i *s* represents the systematic uncertainties regarding the observational procedure, and sys model,i *s* represents the systematic uncertainties regarding the modeling procedure.

In principle, *s*obs,i should be considered as a function of the observer-frame wavelength, while sys model,i *s* should be considered as a function of the rest-frame wavelength. For example,

<sup>4</sup> We have used the version of these attenuation curves as implemented in the HyperZ code (Bolzonella et al. 2000).

Brammer et al. (2008) have introduced a rest-frame template error function to account for the systematic uncertainties in the SED model. However, the form of the rest-frame template error function, which is likely to be model dependent, must be determined in advance, instead of during the SED fitting. Besides, the definition of a flexible form of wavelengthdependent *s*obs,i and sys model,i *s* would require too many free parameters, which cannot be well constrained by the limited number of photometric data. Hence, in BayeSED V2.0, the two additional terms are simply defined as

and

$$\sigma_{\rm sys}^{\rm model,i}=\epsilon\Gamma_{\rm sys}^{\rm model}\quad\ast\quad F_{\rm model,i},\tag{9}$$

sys err 8 *F* obs,i

obs *s* = * obs,i ( )

where errsys obs and errsys model are two wavelength-independent free parameters.

sys

In the literature (e.g., Dale et al. 2012; Dahlen et al. 2013), only one of errsys obs and errsys model is usually used and fixed to a predetermined value (typically, 0.02–0.2). Hence, to start from a simpler assumption and not go too much beyond the common practice, in this work only errsys obs is considered as a free parameter spanning (0, 1), while errsys model is fixed to be zero. Due to the simple definition in Equations (8) and (9), the two free parameters errsys obs and errsys model are likely to be degenerated with each other to some extent. In practice, we found that the reduced 2 *c* tend to be closer to 1 in most cases if only errsys obs is considered as a free parameter. Besides, we found that the free parameter errsys obs can be well constrained by the data and very close to the typical value (see Table 3 and Figures 9, 10). On the other hand, if errsys model is left to vary as a free parameter, the model deficiencies would be deposited in this free parameter, and it is potentially possible to use the value of errsys model as an indicator of the quality of a certain model. However, if errsys model is also considered as a free parameter, the difference between different SED models as shown in the Bayesian evidence, which is the focus of this paper, would likely be diluted. We leave the exploration of the effects of errsys model and more complicated forms of both errsys model and errsys obs for future study.

# 3.2. Bayesian Model Comparison

Bayesian model comparison tries to compare competing models, which may have similar or different parameters, by calculating the probability of each model as a whole. Similar to Bayesian parameter estimation, Bayesian model comparison can be achieved by the application of Bayes's theorem in the model space

$$p(M|d,I)=\frac{p(d|M,I)p(M|I)}{p(d|I)}.\tag{10}$$

Here the a priori probability distribution of models in the model space, *p*( ∣) *M I* , can be used to represent our aesthetically and/ or empirically motivated like or dislike of a model, although it is often assumed to be uniform in practice. The Bayesian evidence, or model likelihood of the model *M*, *p*(∣ ) *dM I*, , which is just a normalization factor in Equation (4) and not relevant to parameter estimation, is crucial for Bayesian model comparison. The Bayesian evidence of a model *p*(∣ ) *dM I*, can be obtained by the marginalization (integration) over the entire

parameter space

$$p(d|M,I)\equiv\int p(d|\theta,M,I)p(\theta|M,I)d^{N}\theta.\tag{11}$$

In Equation (10), *p*(∣) *d I* is a normalization factor, which is not relevant to the Bayesian comparison of different models *M* but could be crucial for the Bayesian comparison of different background assumptions *I* in an even higher level of Bayesian inference.

Two models, *M*2 and *M*1, can be formally compared with the ratio of their posterior probabilities given the same observational data set *d* and the background knowledge and assumptions *I*:

$$p(M_{2}|d,I)=\frac{p(d|M_{2},I)p(M_{2}|I)}{p(d|M_{1},I)p(M_{1}|I)},\tag{12}$$

where *p p* ( ∣) ( ∣) *MI MI* 2 1 is the prior odds ratio of the two models. If neither of the two models is more favored a priori, the Bayes factor, which is defined as

$B_{2,1}=\frac{p(d|M_{2},I)}{p(d|M_{1},I)}$, (13)

can be directly used for the Bayesian model comparison. In practice, the empirically calibrated Jeffrey's scale (Jeffreys 1961; Trotta 2008) suggests that ln 0, 1, 1.5 ( ) *B*2,1 > , and 5 (corresponding to the odds of about 1:1, 3:1, 12:1, and 150:1) can be used to indicate inconclusive, weak, moderate, and strong evidence in favor of M2, respectively (see also Jenkins 2014). If more than two models need to be compared, it would be convenient to define a standard model M0 and compute the Bayes factors *Bi*,0 of all models with respect to the standard model. When comparing models with their Bayes factors, it is important to make sure that the data *d* and all of the background knowledge/assumptions *I* used in all models are the same, or the results of comparison would be meaningless.

# 3.3. Occam Factor

As the prior-weighted average of likelihood over the entire parameter space, the Bayesian evidence obtained with Equation (11) automatically implements the principle of Occam's razor. Actually, the Bayesian evidence is determined by the trade-off between the complexity of a model and its goodness of fit to the data. The Occam factor (see, e.g., MacKay 2003; Gregory 2005), which represents a penalty to a model for having to finely tune its free parameters to match the observations, is related to the variety of the predictions that a model makes in the data space. By adopting the suggestion of Gregory (2005), we define the Occam factor of a model as

$\Omega_{\theta}\triangleq\frac{p(d|M,I)}{\mathcal{L}_{\max}(\theta)}$, (14)

where max(*q*ˆ) is the maximum of the likelihood function at *q*ˆ. Hence, the Occam factor defined here is just the ratio of average likelihood and maximum likelihood, which is never greater than 1. It ensures that

$p(d|M,I)=\mathcal{L}_{\max}(\hat{\theta})\Omega_{\theta}$. (15)

A complex model would require a fine-tuning of its parameters to give a better fit to the observational data. Then, a large fraction of its parameter space would be useless and consequently wasted. In that case, its average likelihood will be much smaller than its maximum likelihood, which leads to a smaller Occam factor. The Occam factor defined in Equation (14) is not directly related to the algorithmic complexity of a model and cannot be obtained independently of the observational data. Thus, it would be interesting to find out whether this simple quantification of the complexity of a model is consistent with our intuition about the complexity of the model. Some examples about this will be presented in Section 6.

# 4. The Bayesian Evidence for the SED Modeling of an Individual Galaxy

When modeling the SED of a galaxy, it is clear from Section 2 that we need to make assumptions about the SSP model, the form of SFH, and the properties of the ISM given by the DAL. Since our understandings of these physical processes are far from complete, we have many possible choices for each one of them. Apparently, different choices of these components would result in very different SED modelings. In this section, we introduce the methods for computing the Bayesian evidence for the different SED modelings.

In practice, it is meaningful to distinguish between two kinds of SED modelings: the SED modelings with the SSP, SFH, and DAL all being fixed and the SED modelings with one of the SSP, SFH, and DAL being fixed and the other two being free to vary. The Bayesian model comparison of the first kind of SED modelings can be used to ask the question, which specific combination of SSP, SFH, and DAL is the best? Differently and more interestingly, the Bayesian model comparison of the second kind of SED modelings can be used to ask the question, which SSP/SFH/DAL is the best regardless of the choices of the other two? In Sections 4.1 and 4.2, we will introduce our method to compute the Bayesian evidence for the two different kinds of SED modelings, respectively.

# 4.1. The SED Modeling of a Galaxy with SSP, SFH, and DAL All Being Fixed

Since we have many possible choices for the SSP, SFH, and DAL when modeling the SED of galaxies, it would be interesting to ask, within all the possible choices, which combination of the SSP, SFH, and DAL is the best for the interpretation of given observational data? This question can be answered by the Bayesian model comparison of the *M ssp sfh dal* , , ( ) 0 0 0 -like SED model, which has assumed a specific SSP model ssp0, SFH sfh0, and DAL dal0. The hierarchical (or multilevel) structure of this kind of SED modeling of a galaxy is shown in Figure 2.

As mentioned above, the computation of Bayesian evidence is crucial for the Bayesian model comparison. The Bayesian evidence for an *M ssp sfh dal* , , ( ) 0 0 0 -like SED model can be obtained as

$$p(d_{1}|M\,(ssp_{0},\,s\!h_{0},\,dal_{0}),\,I)$$
 
$$=\int p(d_{1}|\theta_{1},\,M\,(ssp_{0},\,s\!h_{0},\,dal_{0}),\,I)$$
 
$$\times p\,(\theta_{1}|M\,(ssp_{0},\,s\!h_{0},\,dal_{0}),\,I)\,{\rm d}\theta_{1}\tag{16}$$

$$\begin{array}{l}\mbox{\cal L}_{\rm max}(\hat{\theta}_{1})\Omega_{\theta_{1}},\\ \mbox{\cal L}_{\rm max}(\hat{\theta}_{1})\Omega_{\theta_{1}},\end{array}\tag{17}$$

where

$${\cal L}_{\rm max}(\hat{\theta}_{1})\equiv\max_{\theta_{1}}[p(d_{1}|\theta_{1},\,M\,(ssp_{0},\,sfh_{0},\,dal_{0}),\,I)]\tag{18}$$

is the maximum of the likelihood function at *q*1 ˆ , and W*q*1 is the defined Occam factor associated with the free parameters *q*1 of the *M ssp sfh dal* , , ( ) 0 0 0 -like SED model. If we use the shorthand " *M ssp sfh dal* , , ∣∣( ) 0 0 0 " to indicate that *M ssp sfh dal* , , ( ) 0 0 0 is the conditioning information common to all displayed probabilities in the equation, then Equation (16) can be significantly shortened as

$$p(d_{1}|I)$$
 
$$=\int p(d_{1}|\theta_{1},I)p(\theta_{1}|I)\mathrm{d}\theta_{1}\quad||\,M\,(ssp_{0},\,s\!h_{0},\,dal_{0}).\tag{19}$$

Similar shorthand will be used throughout this paper.

# 4.2. The SED Modeling of a Galaxy with One of SSP, SFH, and DAL Being Fixed and the Other Two Being Free to Vary

#### 4.2.1. The Case for a Fixed SSP but Free SFH and DAL

Given the observational data of a galaxy, it is even more interesting to ask the question, which SSP model is the best regardless of the choices of the SFH and DAL? To answer this question, it is useful to define an SED model *M ssp sfh dal* , , 0 ( ), where the SSP model is fixed to the specific choice ssp0, while the SFH and the DAL are free to vary. The hierarchical structure of this kind of SED modeling of a galaxy is shown in Figure 3.

Hence, sfh and dal are considered as two free parameters in addition to *q*1, while ssp0 represents a given SSP model.

The Bayesian evidence for the *M ssp sfh dal* , , 0 ( )-like SED model can be obtained as

$$p(d_{1}|I)\sum_{j,k}\int p(d_{1}|\theta_{1},\,\textit{sfn}_{j},\,\textit{dal}_{k},\,I)$$
 
$$\times p(\theta_{1},\,\textit{sfn}_{j},\,\textit{dal}_{k}|I)d\theta_{1}||\,M(\textit{ssp}_{0},\,\textit{sfn},\,\textit{dal})\tag{20}$$

$$\begin{array}{l}\mbox{\cal L}_{\rm max}(\hat{\theta}_{1},\,\hat{\pi}h,\,\hat{d}al)\Omega_{\theta_{1}}\Omega_{\hat{\pi}h}\Omega_{\hat{d}al}\end{array}\tag{21}$$

$\mathbb{C}_{\max}(\hat{\theta}_{1},\hat{\theta}_{n},\hat{\theta}_{n})\Omega_{\rm Total}$, (22)

where

$$\mathcal{L}_{\max}(\hat{\theta}_{1},\,\hat{s}\hat{m},\,\hat{d}al)$$
 
$$\equiv\max_{\theta_{1},j,k}[p(\mathbf{d}_{1}|\theta_{1},\,\hat{s}\hat{m}_{j},\,\hat{d}al_{k},\,\mathbf{M}\,(ssp_{0},\,\hat{s}\hat{m},\,\hat{d}al),\,\mathbf{I})]\tag{23}$$

is the maximum of the likelihood function at (*q*1, , *sfh dal* ˆ ˆ ˆ ), and W*q*1 , Ωsfh, and Ωdal are the defined Occam factors associated with the free parameters of this SED model. The additional Occam factors Ωsfh and Ωdal imply that the *M ssp sfh dal* , , 0 ( )-like SED model will be further punished for having to freely select the SFH and DAL to match the observations.

Using the product rule of probability, we can obtain the identity equation:

$p(\theta_{1},\,\textit{sfl}_{j},\,\textit{dal}_{k}|I)=p(\theta_{1}|\textit{sfl}_{j},\,\textit{dal}_{k},\,I)$  
  
$\times p(\textit{sfl}_{j},\,\textit{dal}_{k}|I)$  
  
$||\,M\,(\textit{ssp}_{0},\,\textit{sfl},\,\textit{dal}).$ (24)

Hence, Equation (20) can be rewritten as

*d I dI I I M Id I I M M I dM I p p sfh dal p sfh dal p sfh dal d ssp sfh dal p sfh dal p sfh dal p sfh dal d ssp sfh dal p sfh dal ssp sfh dal p ssp sfh dal* ,, , , , , ,, , ,, , , , ,, , ,, , , , , . 25 *j k j k j k j k j k j k j k j k j k j k j k* 1 , 11 1 1 0 , 1 1 1 1 0 , 0 1 0 ò ò å å å *q q q q q q* = = = ( ∣) (∣ )( ∣ ) ( ∣ ) ∣∣ ( ) ( ∣) ( ∣ ) ( ∣ ) ∣∣ ( ) ( ∣( )) (∣ ( ) ) ( )

With the assumptions that the SSP, SFH, and DAL are independent of each other and the Nssp of SSP, the Nsfh of SFH, and the Ndal of DAL are equally likely a priori, Equation (25) can be further simplified as

$$p(d_{1}|I)$$
 
$$=\sum_{j,k}p(sfl_{j}|I)p(dal_{k}|I)p(d_{1}|I)$$
 
$$\times\ ||\ M\left(ssp_{0},\ sfl,\ dal\right)$$
 
$$=\frac{1}{N_{sfl_{k}}N_{dal}}\sum_{j,k}p(d_{1}|M\left(ssp_{0},\ sfl_{j},\ dal_{k}\right),\ I).\tag{26}$$

The method of calculating the Bayesian evidence for the *M ssp sfh dal* , , 0 ( )-like SED modeling presented above can be similarly applied to the *M ssp sfh dal* , , 0 ( )- and *M*( ) *ssp sfh dal* , , 0 -like SED modelings. The hierarchical structures of the last two kinds of SED modelings of a galaxy are shown in Figures 4 and 5, respectively.

The Bayesian evidence of an *M ssp sfh dal* , , 0 ( )-like SED can be obtained as

*dM I dM I p ssp sfh dal N N p ssp sfh dal* ,,, 1 , , , . 27 *ssp dal i k i k* 1 0 , = å 1 0 (∣ ( ) ) (∣ ( ) ) ( )

It can be used to answer the question, given the observational data of a galaxy, which SFH model is the best regardless of the choices of SSP and DAL? Similarly, the Bayesian evidence of the *M*( ) *ssp sfh dal* , , 0 -like SED modeling can be obtained as

$$p(\mathbf{d}_{1}|\mathbf{M}\left(\mathbf{s}sp,\,\mathbf{s}\mathbf{h},\,\mathbf{d}al_{0}\right),\,\mathbf{I})$$
 
$$=\frac{1}{N_{\mathbf{s}sp}N_{\mathbf{s}\mathbf{h}}}\sum_{i,j}p(\mathbf{d}_{1}|\mathbf{M}\left(\mathbf{s}sp_{i},\,\mathbf{s}\mathbf{h}_{j},\,\mathbf{d}al_{0}\right),\,\mathbf{I}).\tag{28}$$

It can be used to answer the question, given the observational data of a galaxy, which DAL is the best regardless of the choices of SSP and SFH?

# 5. The Bayesian Evidence for the SED Modeling of a Sample of Galaxies

When modeling and interpreting the SEDs of a sample of galaxies, we need to make assumptions about the SSP, the form of SFH, and the DAL for all galaxies in the sample. In many works in the literature, a common SSP, SFH, and DAL (e.g., the BC03 SSP with a Chabrier03 IMF, exponentially declining SFH, and Calzetti law) are often assumed for all galaxies in their sample. However, we cannot make sure that the SFH and DAL for different galaxies must be the same. Generally, the different assumptions about the universality of SSP, SFH, and DAL result in different SED modelings of a sample of galaxies, and the correctness of them needs to be properly justified. This can be achieved by the Bayesian model comparison of the SED modelings of a sample of galaxies with different assumptions about the universality of SSP, SFH, and DAL. The foundation for this kind of study is the computation of the Bayesian evidences for the different cases. In this paper, we limit ourselves to two kinds of SED modelings of a sample of galaxies: the one with SSP, SFH, and DAL all being assumed to be universal, and the one with one of the SSP, SFH, and DAL being assumed to be universal while the other two are object dependent. We introduce our method for computing the Bayesian evidence for them in Sections 5.1 and 5.2, respectively.

# 5.1. The SED Modeling of a Sample of Galaxies with SSP, SFH, and DAL All Being Assumed to Be Universal

As a widely used approach when modeling and interpreting the SEDs of a sample of galaxies, the same SSP, SFH, and DAL are often assumed for all galaxies in a sample, especially when the size of the sample is very large. This is a natural choice, since it would be much more computationally demanding to use different SSP, SFH, and/or DAL for different galaxies when we have a large sample. In this subsection, we introduce the method for computing the Bayesian for this case. Although the SSP, SFH, and DAL are all assumed to be universal for all galaxies in a sample, we still have many possible choices for each one of them. This is very similar to the case for an individual galaxy in Section 4. In Sections 5.1.1 and 5.1.2, we introduce our method for computing the Bayesian evidence for the different cases, respectively.

#### 5.1.1. The Case for a Fixed SSP, SFH, and DAL

As the most widely used approach for the SED modeling of a sample of galaxies, the *M ssp sfh dal* , , ( ) 0 0 0 -like SED modeling assumes a particular SSP, SFH, and DAL for all galaxies in a sample. The hierarchical structure of this kind of SED modeling of a sample of N galaxies is shown in Figure 6. The Bayesian evidence of this kind of SED modeling for a sample of galaxies can be obtained as

$$p(d_{1},\,d_{2},\,...,d_{N}|I)$$
 
$$=\int p(d_{1},\,d_{2},\,...,d_{N}|\theta_{1},\,\theta_{2},\,...,\,\theta_{N},\,I)$$
 
$$p(\theta_{1},\,\theta_{2},\,...,\,\theta_{N}|I)\,\mathrm{d}\theta_{1}\,\mathrm{d}\theta_{2}\,...\,\mathrm{d}\theta_{N}\,\,||\,M(\mathit{ssp}_{0},\,\mathit{sfn}_{0},\,\mathit{dal}_{0})\tag{29}$$

$$\mathcal{L}_{\max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,...,\,\hat{\theta}_{N})\Omega_{\theta_{1}},\,\Omega_{\theta_{2}},\,...,\,\Omega_{\theta_{N}}\tag{30}$$

$$\equiv{\cal L}_{\rm max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,...,\,\hat{\theta}_{N})\Omega_{\rm Total},\tag{31}$$

where

$$\mathcal{L}_{\max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,\ldots,\,\hat{\theta}_{N})$$
 
$$\equiv\max_{\theta_{1},\theta_{2},\,\ldots,\,\theta_{N}}[p(\mathbf{d}_{1},\,\mathbf{d}_{2},\,\ldots,\,\mathbf{d}_{N}|\theta_{1},\,\theta_{2},\,\ldots,\,\theta_{N},$$
 
$$\mathbf{M}\left(\mathit{ssp}_{0},\,\mathit{sfh}_{0},\,\mathit{dal}_{0}\right),\,\mathbf{I})]\tag{32}$$

is the maximum of the likelihood function at (*qq q* 1 2 ,,, ¼ *N* ˆˆ ˆ ), and , ,, W W ¼W *qq q* 1 2 *N* is the defined Occam factor associated with the free parameters of N galaxies.

As shown in Figure 6, we assume that the observational data *di* of different galaxies are independent of each other and that the parameters of a galaxy *qi* tell nothing about the observational data *dj* of any other galaxy. With these assumptions, the Bayesian evidence of a *M ssp sfh dal* , , ( ) 0 0 0 -like SED model in Equation (29) can be obtained as

$p(d_{1},d_{2},...,d_{N}|M\left(ssp_{0},\,sfh_{0},\,dal_{0}\right),\,I)$  
  
$=\prod\limits_{g\ =\ 1}^{N}\int p(d_{g}|\theta_{g},\,M\left(ssp_{0},\,sfh_{0},\,dal_{0}\right),\,I)$  
  
$p(\theta_{g}|M\left(ssp_{0},\,sfh_{0},\,dal_{0}\right),\,I)\mbox{d}\theta_{g}$  
  
$=\prod\limits_{g\ =\ 1}^{N}\ p(d_{g}|M\left(ssp_{0},\,sfh_{0},\,dal_{0}\right),\,I).$ (33)

#### 5.1.2. The Case for a Fixed SSP but Free SFH and DAL

It is interesting to check the performance of a particular SSP model for a sample of galaxies and independently of the selection of SFH and DAL. This can be achieved by defining an *M ssp sfh dal* , , 0 ( )-like SED modeling for a sample of N galaxies, where a particular SSP model ssp0 and a free SFH and DAL are assumed for all galaxies in the sample. The hierarchical structure of this kind of SED modeling of a sample of N galaxies is similar to Figure 6, but with the nodes for SFH and DAL being empty. With the Bayesian evidence for this case, we can answer the question, given the observational data set of a sample of N galaxies, which SSP model is the best regardless of the choices of the SFH and DAL? The Bayesian evidence for this case can be obtained as

$p(\mathbf{d}_{1},\,\mathbf{d}_{2},\,...,\,\mathbf{d}_{N}|\mathbf{I})$  
  

$$=\sum_{j,k}\int p(\mathbf{d}_{1},\,\mathbf{d}_{2},\,...,\,\mathbf{d}_{N}|\mathbf{\theta}_{1},\,\mathbf{\theta}_{2},\,...,\,\mathbf{\theta}_{N},\,\mbox{\it sfl}_{j},\,\mbox{\it dal}_{k},\,\mathbf{I})$$
 
$$p(\mathbf{\theta}_{1},\,\mathbf{\theta}_{2},\,...,\,\mathbf{\theta}_{N},\,\mbox{\it sfl}_{j},\,\mbox{\it dal}_{k}|\mathbf{I})d\mathbf{\theta}_{1}d\mathbf{\theta}_{2}\,...\,d\mathbf{\theta}_{N}$$
 
$$||\,\mathbf{M}\,(\mbox{\it ssp}_{0},\,\mbox{\it sfl}_{j},\,\mbox{\it dal})\tag{34}$$

$$\equiv{\cal L}_{\rm max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,...,\,\hat{\theta}_{N},\,\hat{\theta}_{N},\,\hat{\theta}_{N},\,\hat{\theta}_{N})$$
 
$$\ast\Omega_{\theta_{1}}\Omega_{\theta_{2}}\,...\,\Omega_{\theta_{N}}\Omega_{\hat{\theta}_{N}}\Omega_{\hat{\theta}_{N}}\Omega_{\hat{\theta}_{N}}\Omega_{\hat{\theta}_{N}}\Omega_{\hat{\theta}_{N}}\tag{35}$$

$$\mathcal{L}_{\max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,...,\,\hat{\theta}_{N},\,\hat{\theta}_{N},\,\hat{\theta}_{1})\Omega_{\rm Total},\tag{36}$$

where

$$\mathcal{L}_{\max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,...,\,\hat{\theta}_{N},\,\hat{\mathit{s}}\hat{\mathit{h}},\,\hat{\mathit{d}}\mathit{al})

\equiv\max[p(\mathbf{d}_{1},\,\mathbf{d}_{2},\,...,\,\mathbf{d}_{N}|\theta_{1},\,\theta_{2},\,...,\,\theta_{N},\,\mathit{s}\hat{\mathit{h}}_{j},\,\mathit{d}\mathit{al}_{k},

\theta_{1,\,\theta_{2},\,...,\,\theta_{N},\,\mathit{s}\hat{\mathit{h}},\,\mathit{d}\mathit{al}),\,\mathbf{I})]$$

is the maximum of the likelihood function at (*qq q* 1 2 ,,,, , ¼ *N sfh dal* ˆˆ ˆ ˆ ˆ ), and , ,, W W ¼W *qq q* 1 2 *N* is the defined Occam factor associated with the free parameters of N galaxies. Since the SFH and DAL are assumed to be universal for all galaxies in the sample, we only need to add two free parameters (sfh and dal) to represent the selection of the form of SFH and DAL. The associated two additional Occam factors Ωsfh and Ωdal imply that the *M ssp sfh dal* , , 0 ( )-like SED modeling for a sample of galaxies will be further punished for having to freely select the SFH and DAL to match the observations.

As in Equation (24), we can obtain a similar identity equation for N galaxies as

$$p(\theta_{1},\,\theta_{2},\,...,\,\theta_{N},\,\mathit{sfl}_{j},\,\mathit{dal}_{k}|I)$$
 
$$=p(\theta_{1},\,\theta_{2},\,...,\,\theta_{N}|\mathit{sfl}_{j},\,\mathit{dal}_{k},\,I)p(\mathit{sfl}_{j},\,\mathit{dal}_{k}|I)$$
 
$$||\,\mathit{M}\,(\mathit{ssp}_{0},\,\mathit{sfl},\,\mathit{dal}).\tag{38}$$

Hence, the Bayesian evidence in Equation (34) can be rewritten as

*dd dI dd d I I I M I dd d I I M I dd d I M p p sfh dal p sfh dal p sfh dal d d d ssp sfh dal p sfh dal p sfh dal p sfh dal d d d ssp sfh dal p sfh dal p sfh dal ssp sfh dal* ,,, ,,, ,,, , , , ,,, , , , , ... , , , ,,, ,,, , , , , , , , , ... , , , ,,, , , , , . 39 *N j k N N j k N j k j k N j k j k N N j k N j k N j k j k N j k* 1 2 , 12 12 1 2 1 2 0 , 12 12 12 12 0 , 1 2 0 ò ò å å å *qq q qq q qq q qq q qq q qq q* ¼ = ¼¼ ¼ = ¼ ¼ ¼ = ¼ ( ∣) (∣ ) ( ∣ )( ∣ ) ∣∣ ( ) ( ∣) (∣ ) ( ∣) ∣∣ ( ) ( ∣) ( ∣ ) ∣∣ ( ) ( )

As in Equation (26), we assume that the SSP, SFH, and DAL are independent of each other and that the Nssp kinds of SSP model, the Nsfh forms of SFH model, and the Ndal kinds of DAL are equally likely a priori. In addition, we assume that the physical properties of different galaxies are independent of each other. With these assumptions, Equation (39) can be further simplified as

*dd dI I I dd d I M II d I M dM I p p sfh p dal p sfh dal ssp sfh dal p sfh p dal p sfh dal ssp sfh dal N N p ssp sfh dal* ,,, ,,, , , , , , , , , 1 , , , . 40 *N j k j k N j k j k j k g N g j k sfh dal jk g N g j k* 1 2 , 1 2 0 , 1 0 , 1 0 å å å ¼ = ¼ = = = = ( ∣) ( ∣) ( ∣) ( ∣ ) ∣∣ ( ) ( ∣) ( ∣) ( ∣ ) ∣∣ ( ) (∣ ( ) ) ( )

The above method of calculating the Bayesian evidence for the *M ssp sfh dal* , , 0 ( )-like SED modeling for a sample of N galaxies can also be applied to the *M ssp sfh dal* , , 0 ( )-like and *M*( ) *ssp sfh dal* , , 0 -like SED modeling for a sample of N galaxies. The Bayesian evidence of the *M ssp sfh dal* , , 0 ( )-like SED modeling for a sample of N galaxies can be obtained as

$$p(\mathbf{d}_{1},\mathbf{d}_{2},\,...,\mathbf{d}_{N}|\mathbf{M}\,(\mathbf{s}sp,\,\mathbf{s}\hbar_{0},\,\mathbf{d}al),\,\mathbf{I})$$
 
$$=\frac{1}{N_{\mathbf{s}sp}N_{\mathbf{d}al}}\sum_{i,k}\prod_{g=1}^{N}\,p(\mathbf{d}_{g}|\mathbf{M}\,(\mathbf{s}sp_{i},\,\mathbf{s}\hbar_{0},\,\mathbf{d}al_{k}),\,\mathbf{I}).\tag{41}$$

It can be used to answer the question, given the observational data set of a sample of N galaxies, which SFH model is the best regardless of the choices of the SSP and DAL? Similarly, the Bayesian evidence of the *M*( ) *ssp sfh dal* , , 0 -like SED modeling

37

( )

for a sample of N galaxies can be obtained as

$$p(\mathbf{d}_{1},\,\mathbf{d}_{2},\,...,\,\mathbf{d}_{N}|\mathbf{M}\,(ssp,\,\,\mbox{\it sfh},\,\,\mbox{\it dal}_{0}),\,\mathbf{I})$$
 
$$=\frac{1}{N_{\mbox{\scriptsize{ssp}}}N_{\mbox{\scriptsize{sfh}}}}\sum_{i,j}\prod_{g=1}^{N}\,p(\mathbf{d}_{g}|\mathbf{M}\,(ssp_{i},\,\,\mbox{\it sfh}_{j},\,\,\mbox{\it dal}_{0}),\,\mathbf{I}).\tag{42}$$

It can be used to answer the question, given the observational data set of a sample of N galaxies, which DAL is the best regardless of the choices of the SSP and SFH model?

# 5.2. The SED Modeling of a Sample of Galaxies with One of SSP, SFH, and DAL Being Assumed to Be Universal and the Other Two Being Object Dependent

In Section 5.1, we have introduced the method of calculating the Bayesian evidence for the SED modeling of a sample of galaxies where the SSP, SFH, and DAL are all assumed to be universal. However, this could be too strong an assumption. Hence, in this subsection we introduce the method of calculating the Bayesian evidence for the SED modelings with only one of the SSP, SFH, and DAL being assumed to be universal while the other two are object dependent.

# 5.2.1. The Case for a Universal SSP but Object-dependent SFH and DAL

In practice, it is very interesting to ask, given the observational data set of a sample of N galaxies, which SSP model is the best regardless of the different choices of the SFH and DAL for different galaxies? This question can be answered by calculating the Bayesian evidence for an *M ssp sfh sfh sfh dal dal dal* , , ,, , , ,, ( 012 ¼ ¼ *N* 1 2 *N*)-like SED modeling of a sample of N galaxies where a particular SSP model ssp0 is assumed for all galaxies in the sample but the form of SFH and DAL for different galaxies are allowed to be different. The hierarchical structure of this kind of SED modeling of a sample of N galaxies is shown in Figure 7.

The Bayesian evidence for this case can be defined as

*dd dI dd d I I M p p sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal d d d ssp sfh sfh sfh dal dal dal* ,,, ,,, ,,, , , ,, , , ,, , ,,,, , ,, , , , , ... , , , , , , , , 43 *N jj j kk k N N jj j kk k N jj j kk k N N N* 1 2 ,, , ,, , , 12 12 1 2 1 2 012 1 2 *N N N N N N* 1 2 1 2 1 2 1 2 1 2 1 2 å ò *qq q qq q qq q* ¼ = ¼¼ ¼ ¼ ´¼ ¼ ¼ ¼ ¼ ¼ ¼ ( ∣) ( ∣ ) ( ∣ ) ∣∣ ( ) ( )

$$\equiv{\cal L}_{\rm max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,...,\,\hat{\theta}_{N},\,\hat{s}\hat{h}_{1},\,\hat{s}\hat{h}_{2},\,...,\,\hat{s}\hat{h}_{N},$$
  
  

$$\hat{d}\hat{a}l_{1},\,\hat{a}\hat{a}l_{2},\,...,\,\hat{a}\hat{a}l_{N})\,\Omega_{\theta_{1}},\,\Omega_{\theta_{2}},\,...,\,\Omega_{\theta_{N}}$$
  
  

$$\Omega_{\hat{s}\hat{h}_{1}},\,\Omega_{\hat{s}\hat{h}_{2}},\,...,\,\Omega_{\hat{s}\hat{h}_{N}}\,\Omega_{\hat{d}\hat{a}l_{1}},\,\Omega_{\hat{d}\hat{a}l_{2}},\,...,\,\Omega_{\hat{d}\hat{a}l_{N}}\tag{44}$$

$$\equiv{\cal L}_{\rm max}(\hat{\theta}_{1},\,\hat{\theta}_{2},\,...,\,\hat{\theta}_{N},\,s\hat{h}_{1},\,s\hat{h}_{2},\,...,\,s\hat{h}_{N},$$
  
  
$d\hat{a}l_{1},\,d\hat{a}l_{2},\,...,\,d\hat{a}l_{N})\,\Omega_{\rm Total},$ (45)

where

*dd d M I sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal ssp sfh dal* ,,,, , ,, , , ,, max , , , , , ,, , ,, , , ,, , , , , 46 *N N N jj j kk k N N jj j kk k* max 1 2 1 2 1 2 , , , ,,, , ,, , , 12 12 0 *N N N N N* 1 2 1 2 1 2 1 2 1 2 *qq q q q q* ¼ ¼ ¼ º ¼ ¼¼ ¼ *qq q* ¼¼¼ (ˆˆ ˆ ˆˆ ˆ ˆˆ ˆ ) [( ∣ ( ) )] ( )

is the maximum of the likelihood function at ,,,, , ,, , , ,, *N sfh sfh sfh dal dal dal* (*qq q* 1 2 ¼¼ ¼ 1 2 *N* 1 2 *N* ˆˆ ˆ ˆˆ ˆˆˆ ˆ ), and , ,, W W ¼W *qq q* 1 2 *N*, , ,, *sfh sfh sfh* W W ¼W 1 2 *N*, and *dal* , W 1 *dal dal* , , W ¼W 2 *N* are the defined Occam factors associated with the free parameters of the N galaxies. Since the SFH and DAL are not assumed to be universal for all galaxies in the sample, we need to add two free parameters to represent the selection of the form of SFH and DAL for each galaxy. Hence, the associated 2**N* additional Occam factors , ,, *sfh sfh sfh* W W ¼W 1 2 *N* and *dal dal dal* , ,, W W ¼W 1 2 *N* imply that the *M ssp* , 0 ( *sfh sfh sfh dal dal dal* , ,, , , ,, 1 2 ¼ ¼ *N* 1 2 *N*)-like SED modeling for a sample of N galaxies will be further punished for having to freely select the SFH and DAL for each galaxy in the sample to match the observations.

With the identity equation as

*I I I M p sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal ssp sfh sfh sfh dal dal dal* ,,,, , ,, , , ,, ,,, , ,, , , ,, , , ,, , , ,, , , , , , , , , , 47 *N jj j kk k N j j j kk k jj j kk k N N* 1 2 1 2 012 1 2 *N N N N N N* 1 2 1 2 1 2 1 2 1 2 1 2 *qq q qq q* ¼ ¼ ¼= ¼ ¼ ¼ ¼ ¼ ¼ ¼ ( ∣) ( ∣ ) ( ∣) ∣∣ ( ) ( )

the Bayesian evidence in Equation (43) can be rewritten as

*dd dI dd d I I I M p p sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal d d d ssp sfh sfh sfh dal dal dal* ,,, ,,, ,,, , , ,, , , ,, , ,,, , ,, , , ,, , , ,, , , , , ... , , , , , , , , 48 *N jj j kk k N N jj j kk k N jj j kk k jj j kk k N N N* 1 2 ,, , ,, , , 12 12 1 2 1 2 012 1 2 *N N N N N N N N* 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 å ò *qq q qq q qq q* ¼ = ¼¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ( ∣) ( ∣ ) ( ∣ ) ( ∣ ) ∣∣ ( ) ( )

*I dd d I I M p sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal p sfh sfh sfh dal dal dal d d d ssp sfh sfh sfh dal dal dal* , ,, , , ,, ,,, ,,, , , ,, , , ,, , ,,, , ,, , , , , , ... , , ,, , , ,, 49 *jj j kk k jj j kk k N N jj j kk k N jj j kk k N N N* ,, , ,, , , 12 12 1 2 1 2 012 1 2 *N N N N N N N N* 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 ò å *qq q qq q qq q* = ¼ ¼ ¼¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ( ∣) ( ∣ ) ( ∣ ) ∣∣ ( ) ( )

*M I dd dM I p sfh sfh sfh dal dal dal ssp sfh sfh sfh dal dal dal p ssp sfh sfh sfh dal dal dal* , ,, , , ,, , , ,, , , ,, , ,,, , , ,, , , , , , . 50 *jj j kk k jj j kk k N N N jj j kk k* ,, , ,, , , 012 1 2 1 2 0 *N N N N N N* 1 2 1 2 1 2 1 2 1 2 1 2 = ¼ å ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ( ∣ ( ) ) ( ∣( )) ()

With the assumptions that the SSP, SFH, and DAL are independent of each other and that the physical properties of different galaxies are independent of each other, Equation (50) can be further simplified as

*dd dM I M I M I dd dM I M I M I dM I p ssp sfh sfh sfh dal dal dal p sfh sfh sfh ssp sfh sfh sfh dal dal dal p dal dal dal ssp sfh sfh sfh dal dal dal p ssp sfh sfh sfh dal dal dal p sfh ssp sfh sfh sfh dal dal dal p dal ssp sfh sfh sfh dal dal dal p ssp sfh dal* ,,, , , ,, , , ,, , , ,, , , ,, , , ,, , , ,, , , ,, , , ,, , ,,, , , ,, , , ,, , , , ,, , , ,, , , , ,, , , , , , , , , . 51 *N N N jj j kk k jj j N N kk k N N N jj j kk k jj j kk k g N j N N j N N g j k* 1 2 012 1 2 ,, , ,, , , 0 1 2 1 2 012 1 2 1 2 0 ,, , ,, , , 1 012 1 2 012 1 2 0 *N N N N N N N N g g g g* 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 å å ¼ ¼ ¼ = ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ = ¼ ¼ ¼ ¼ ¼ ¼ ¼ ¼ = ( ∣( ) ) ( ∣( ) ) ( ∣( ) ) ( ∣( ) ) ( ∣( ) ) ( ∣( ) )( ∣ ( ) ) ( )

Then, we assume that the Nssp kinds of SSP, the Nsfh forms of SFH, and the Ndal kinds of DAL are equally likely a priori. Hence,

*dd dM I d I dM I p ssp sfh sfh sfh dal dal dal N N p ssp sfh dal N N p ssp sfh dal* ,,, , , ,, , , ,, , 1 ,, , 1 , , , . 52 *N N N jj j kk k g N sfh dal g j k sfh dal N jj j kk k g N g j k* 1 2 012 1 2 ,, , ,, , , 1 0 ,, , ,, , , 1 0 *N N g g N N g g* 1 2 1 2 1 2 1 2 å å ¼ ¼ ¼ = = ¼ ¼ = ¼ ¼ = ⎛ ⎝ ⎜ ⎞ ⎠ ⎟ ( ∣( ) ) (∣ ) (∣ ( ) ) ( )

The above method of calculating the Bayesian evidence for the *M ssp sfh sfh sfh dal dal dal* , , ,, , , ,, ( 012 ¼ ¼ *N* 1 2 *N*)-like SED modeling for N galaxies can also be applied to the *M ssp ssp ssp sfh dal dal dal* , ,, , , , ,, ( ) 12 0 ¼ ¼ *N* 1 2 *N* - and *M ssp ssp ssp sfh sfh sfh dal* , ,, , , ,, , ( ) 1 2 12 ¼ ¼ *N N* 0 -like SED modelings. The Bayesian evidence for the *M ssp* , 1 ( *ssp ssp sfh dal dal dal* ,, , , , ,, 2 0 ¼ ¼ *N* 1 2 *N*)-like SED modeling of a sample of N galaxies can be obtained as

*dd dM I dM I p ssp ssp ssp sfh dal dal dal N N p ssp sfh dal* ,,, , ,, , , , ,, , 1 , , , . 53 *N N N ssp dal N ii i kk k g N g i k* 1 2 12 0 1 2 ,, , ,, , , 1 0 *N N g g* 12 1 2 å ¼ ¼ ¼ = ¼ ¼ = ⎛ ⎝ ⎜ ⎞ ⎠ ⎟ ( ∣( ) ) (∣ ( ) ) ( )

It can be used to answer the question, given the observational data set of a sample of N galaxies, which SFH model is the best regardless of the choices of the SSP and DAL for different galaxies?

Similarly, the Bayesian evidence for the *M ssp* , 1 ( *ssp ssp sfh sfh sfh dal* ,, , , ,, , 2 12 ¼ ¼ *N N* 0)-like SED modeling of a sample of N galaxies can be obtained as

$p(\mathbf{d}_{1},\,\mathbf{d}_{2},\,...,\,\mathbf{d}_{N}|\mathbf{M}\,(\mathit{ssp}_{1},\,\mathit{ssp}_{2},\,...,\,\mathit{ssp}_{N},\,\mathit{sfh}_{1},$  
  
$\mathit{sfh}_{2},\,...,\,\mathit{sfh}_{N},\,\mathit{dal}_{0}),\,\mathbf{I})$  
  

$$=\left(\frac{1}{N_{\mathit{ssp}}N_{\mathit{sfh}}}\right)^{N}\sum_{i_{1},i_{2},\,...,\,i_{N},i_{1},i_{2},\,...,\,i_{N}}$$
 
$$\prod_{g}^{N}\,\,p(\mathbf{d}_{g}|\mathbf{M}\,(\mathit{ssp}_{i_{g}},\,\mathit{sfh}_{j_{g}},\,\mathit{dal}_{0}),\,\mathbf{I}).\tag{54}$$

It can be used to answer the question, given the observational data set of a sample of N galaxies, which DAL is the best regardless of the choices of the SSP and SFH for different galaxies?

# 6. Application to a Ks-selected Sample in the COSMOS/ UltraVISTA Field

In this section, by using the new methods for calculating the Bayesian evidence, we present a Bayesian discrimination among the different choices of SSP model, SFH, and DAL in the SED modeling of galaxies, with the multiwavelength observational data of an individual galaxy (Sections 6.2, and 6.3) and of a sample of galaxies (Section 6.4).

# 6.1. Sample Selection and Classification of Galaxies

As in Han & Han (2014), from the Muzzin et al. (2013b) Ks-selected catalog in the COSMOS/UltraVISTA field, which provides reliable spectroscopic redshifts and photometries in 30 bands covering the wavelength range 0.15–24 μm, we have selected a sample of 5467 galaxies mostly with *z* 1. The galaxies in the sample are classified into SFGs and PEGs according to the box regions defined in Muzzin et al. (2013a), which are similar but not identical to those defined in other works (Williams et al. 2009; Whitaker et al. 2011; Brammer et al. 2011). Specifically, PEGs are defined as

$U-V>1.3$, $V-J<1.5$ (55)

$$U-V>(V-J)\times0.88\ +\ 0.69.\tag{56}$$

Generally, there are 1159 PEGs and 4308 SFGs in our sample. In the left panel of Figure 8, we show the distribution of galaxies in our sample in the UVJ color–color diagram. The estimated SFRs of these galaxies with the BC03 model as given in the catalog of Muzzin et al. (2013b) are shown color-coded. It is clear that the classification of galaxies into SFGs and PEGs is consistent with the estimation of SFR. In the right panel of Figure 8, we show the distribution of stellar mass for galaxies in the sample. Most of the PEGs in our sample are massive galaxies with stellar mass larger than 1010*M*, while the SFGs span a much wider range of stellar mass from 108 *M* to 1011 *M*. As shown in the figure, the galaxies in our sample are distributed widely in both the color–color and stellar mass space. The diversity of galaxies in the sample ensures that robust conclusions can be obtained with them.

# 6.2. Bayesian Parameter Estimation for Individual Galaxies

In this subsection, we demonstrate the methods of Bayesian parameter estimation with a PEG (ULTRAVISTA114558) and an SFG (ULTRAVISTA99938) by assuming the commonly used BC03 SSP model with a Chabrier (2003) IMF (bc03_ch), an Exp-dec SFH, and the Calzetti et al. (2000) DAL. The six free parameters of the model and the priors for them are given in Table 2. Generally, a uniform prior truncated at a physically reasonable range is assumed for all free parameters. Besides, the age of a galaxy is forced to be less than the age of the universe at the given redshift z. More physically reasonable and informative priors can be provided by assuming a model for the redshift-dependent distribution of physical parameters of galaxies. However, in this work, we are only interested in the comparison of different SED models. Hence, the truncated uniform prior only reflects the fact that the SED model itself tells us nothing about the detailed distribution of any physical parameter of galaxies, except for the allowed range.

As a benefit of the Bayesian parameter estimation, in addition to the best-fit results and associated estimation of parameters, the detailed posterior probability distribution functions (PDFs) for all of the free and derived parameters of a model can be obtained. The posterior PDFs of parameters fully described our current state of knowledge about them. In Figures 9 and 10, we show the obtained posterior PDFs for all parameters of the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938. The degeneracies between free parameters can be recognized as multiple peaks and/or strong correlations in the 2D PDFs. Besides, the peak and width of the 1D PDFs can be directly used to estimate the value and associated uncertainty of all parameters. For example, the results of parameter estimation for the PEG ULTRA-VISTA114558 and the SFG ULTRAVISTA99938 are shown in Table 3. The results suggest that the PEG ULTRA-VISTA114558 is only slightly older than the SFG ULTRA-VISTA99938. However, it is much more massive than the latter, and it has experienced a much shorter duration of active star formation, which was started much earlier.

It is often very hard, if not impossible, to determine the SFH of a galaxy with only the photometric data. However, with the Bayesian parameter estimation, we can at least obtain the posterior PDF for the SFH of a galaxy. In Figure 11, we show the detailed posterior PDF for the SFHs of the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938. It is clear from the figure that the obtained SFHs of the two galaxies are very uncertain, although the same Exp-dec SFH has been assumed for them. However, the posterior PDF of their SFHs still allows us to recognize the very different nature of their SFHs. The PEG ULTRAVISTA114558 has experienced a very intensive (1000 yr *M* ) star formation phase during the first 10 Myr, after which the star formation activity has been quenched very quickly. Differently, the SFG ULTRAVISTA99938 has experienced a stable (1 yr *M* ) star formation phase during the first 1 Gyr, after which the SFR has only been slightly decreased. These results are consistent with the merger-driven formation mechanism for the massive PEGs and the secular evolution of the disk-dominated SFGs.

Finally, in Figure 12, we show the results of SED fitting for the PEG ULTRAVISTA114558 and the SFG ULTRA-VISTA99938. Except for the best-fit SED as can be given by the traditional SED fitting methods, the Bayesian SED fitting method allows us to obtain the detailed posterior PDF of the model SEDs. From the compact credible regions and the similarity between the median SED and the best-fit SED, it is clear that the SED model is well constrained by the data. For the PEG ULTRAVISTA114558, the GALEX near-UV (NUV) data are far beyond the 95% credible region of the posterior model SEDs. This indicates a failure of the employed SED model. Except for the BC03 model, we have also tested the Yunnan-II and BPASS V2.0 models, which include UV contribution by hot stars even at older ages. The last two models cannot explain the data point as well. Hence, this could indicate some contribution to the UV by a nonstellar (e.g., active galactic nucleus) source. For the SFG ULTRA-VISTA99938, the Spitzer IRAC 3.6 and 4.5 μm data are slightly below the 95% credible region of the posterior model SED. Since the nebular and dust emissions are not considered in the SED model, the bands that harbor contributions from emission lines may artificially boost the observed brightness and push the model fit up. However, given the error bar, the data are basically consistent with the model without the contribution from dust emission. Hence, this suggests that the contribution of dust emission to the two IRAC bands could be ignored. This is consistent with the relatively strong UV emission and low dust extinction (*A*v 0.28 0.20 0.42 = - + ) as shown in Table 3.

# 6.3. Bayesian Discrimination of SSP, SFH, and DAL for the SED Modeling of Individual Galaxies

In Section 6.2, we have demonstrated the results that can be obtained with the Bayesian parameter estimation methods by the application to two example galaxies. We have assumed the standard model (*M*0 1 ) with the most widely used BC03 SSP with a Chabrier03 IMF (bc03_ch), Exp-dec SFH, and SB-like DAL. There are many other possible choices for SSP, SFH, and DAL, and they will result in very different estimations for the physical parameters of a galaxy. Hence, it is very important to find out the best choice for SSP, SFH, and DAL when modeling the SED of a galaxy. Here we present a Bayesian discrimination of their different choices when modeling the SED of the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938.

#### 6.3.1. The Case for SSP, SFH, and DAL Being Fixed

We first consider the cases where the SSP, SFH, and DAL are all fixed to a specific choice. The standard model (*M*0 1 ) mentioned above is just a special example of this kind of SED modeling of a galaxy. With the Bayesian comparison of this kind of SED modeling, we can find out the best combination of SSP, SFH, and DAL for an individual galaxy.

In Figure 13, we show the Bayes factors with respect to the standard model (*M*0 1 ) for the SED modelings of the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938 with all possible combinations of the 16 SSPs, 5 SFHs, and 4 DALs. It is clear from the figure that the Bayes factors for the SED modeling of galaxies with different SSP models could be very different, even if the same SFH and DAL are assumed. For the PEG ULTRAVISTA114558, the combination of the M05 model with a Salpeter55 IMF (m05_sa), the Burst SFH, and the SB-like DAL has the highest value (2.71) of Bayes factor. For the SFG ULTRAVISTA99938, the combination of the version of the GALEV model with the consideration of emission lines and a Kroupa01 IMF (galev_kr), the Exp-dec SFH, and the LMC-like DAL has the highest value (1.19) of Bayes factor.

It is also worth noticing that, for the PEG ULTRA-VISTA114558, the SED modeling of it assuming a constant SFH has the lowest Bayes factors for almost all combinations of SSP and SFH. The maximum likelihoods and Occam factors for these models shown in Figure 14 reveal the reason for this trend. The SED modelings of the PEG ULTRAVISTA114558 assuming a constant SFH are mainly located at the lower right of the figure, which represents low goodness of fit to the data and low model complexity. This result indicates that the constant SFH is too simple to be able to explain the observational SED of the PEG ULTRAVISTA114558. Contrarily, most of the modelings assuming a Burst SFH are located at the upper right of the figure, which represents high goodness of fit to the data and low model complexity. For the SFG ULTRAVISTA99938, it is not easy to find out a clear trend in favor of a particular SFH or DAL. However, it can be noticed in the right panel of Figure 13 that the CB07 and M05 set of SSP models are less suitable for the SFG ULTRAVISTA99938 than other SSPs, which seems to be the opposite of what is the case for the PEG ULTRAVISTA114558 in the left panel of Figure 13.

### 6.3.2. The Case for One of SSP, SFH, and DAL Being Fixed

In Section 6.3.1, we present a Bayesian comparison of the SED modelings of a galaxy for the cases where SSP, SFH, and DAL are all being fixed. This is useful for finding out the best combination of SSP, SFH, and DAL for a galaxy. However, it is not very helpful to find out the best SSP, SFH, or DAL. Actually, we are more interested in questions such as which SSP is the best independently of the choices of SFH and DAL, which SFH is the best independently of the choices of SSP and DAL, and which DAL is the best independently of the choices of SSP and SFH. These more interesting questions can be answered by considering the cases where only one of the SSP, SFH, and DAL is fixed to a specific choice while the other two are allowed to change within a given set. For the computation of Bayes factors, we have used the same standard model (*M*0 1 ) as in Section 6.3.1. It is worth mentioning that the structure of the SED modelings considered here (see Figures 3–5) is different from that of the standard model (*M*0 1 , with a structure shown in Figure 2). Hence, the value of Bayes factor is determined not only by the selection of the physical components (SSP, SFH, DAL) but also by the modeling structure. However, only the differences between Bayes factors are meaningful for the comparison of the different selections of the physical components (SSP/SFH/DAL).

In Figure 15, we show the Bayes factors with respect to the standard model (*M*0 1 ) for the SED modelings of the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938, where a fixed SSP and free SFH and DAL are assumed. This can be used to answer the question, which SSP is the best for the particular galaxy independently of the choices of SFH and DAL? It is clear from the figure that the CB07 SSP with a Chabrier03 IMF (cb07_ch) has the highest value of Bayes factor (1.02) for the PEG ULTRAVISTA114558, while the version of the GALEV model with the consideration of emission lines and a Kroupa01 IMF (galev_kr) has the highest value of Bayes factor (−0.02) for the SFG ULTRA-VISTA99938. It is interesting to notice that the more "TP-AGB heavy" SSP models of CB07 and M05 systematically have much larger Bayes factors than others for the PEG ULTRAVISTA114558, while they systematically have much smaller Bayes factors than others for the SFG ULTRA-VISTA99938. On the other hand, for the PEG, the performance of the GALEV models is not sensitive to the consideration of emission lines and the selection of IMF. For the SFG, the performance of the version of the GALEV models with the consideration of emission lines (galev_kr, galev_sa) is not sensitive to the selection of IMF. Contrarily, the performance of the version of the GALEV models without the consideration of emission lines (galev0_kr, galev0_sa) is very sensitive to the selection of IMF.

In Figure 16, we show the maximum likelihoods, Occam factors, and Bayesian evidences for the same set of SED modelings. It is clear from the figure that the more "TP-AGB heavy" SSP models of CB07 and M05 can provide a better fit (as indicated by the much larger value of maximum likelihoods) to the observational data than other SSP models for the PEG ULTRAVISTA114558. This is the main reason why they have much larger Bayesian evidence and Bayes factor than that of other SSP models as shown in Figure 15. Besides, the versions of the BPASS V2.0 model both with and without the consideration of binaries are located at the lower left of the ML-OF-BE diagram (indicating a low goodness of fit to the data and large model complexity), which suggests that the model is not very suitable for this PEG. For the SFG ULTRAVISTA99938, the results in Figure 16 show that the version of the GALEV model with the consideration of emission lines can provide a significantly better explanation to the data than other SSP models that have not included the contribution of emission lines. Given this result, it is clear that the consideration of nebular emission lines is very necessary for the SFG. It is also interesting to notice that the Bayesian evidences and Bayes factors of the BC03 models are only slightly smaller than that of the GALEV models for the SFG, although the former cannot provide similar goodness of fit to the data. This is mainly because the BC03 models have much larger Occam factors than the GALEV models for the SFG, which indicates lower model complexity of the former.

Similarly, in Figure 17, we show the Bayes factors with respect to the standard model (*M*0 1 ) for the SED modelings of the PEG ULTRAVISTA114558 and the SFG ULTRA-VISTA99938, where a fixed SFH and free SSP and DAL are assumed. The comparison of this kind of SED modeling can be used to answer the question, which SFH is the best for the particular galaxy independently of the choices of SSP and DAL? It is very clear from the figure that the Burst SFH has the highest value of Bayes factor (0.56) for the PEG ULTRAVISTA114558, while the constant SFH has the highest value of Bayes factor (−0.35) for the SFG ULTRA-VISTA99938. For the PEG ULTRAVISTA114558, the Bayes factor of the Burst SFH is significantly larger than that of the constant SFH. This is just the opposite of what is the case for the SFG ULTRAVISTA99938, which indicates the very different natures of the two galaxies. Meanwhile, the ML-OF-BE diagram in Figure 18 shows that the Burst SFH also provides the best explanation to the observational data of the PEG ULTRAVISTA114558, while the Exp-dec SFH, instead of the constant SFH, provides the best explanation to the observational data of the SFG ULTRAVISTA99938. This is mainly because Burst SFH has the largest value of Occam factor (i.e., the lowest model complexity) for the PEG. On the other hand, although the Exp-dec SFH can provide the best explanation to the data of the SFG, it also has the lowest value of Occam factor (i.e., the highest model complexity). Since the more model complexity cannot be balanced by the better goodness of fit to the data, it actually has lower Bayesian evidence and Bayes factor than the simpler constant SFH.

Finally, in Figure 19, we show the Bayes factors with respect to the standard model (*M*0 1 ) for the SED modelings of the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938, where a particular DAL is assumed but the SSP and SFH are free to vary. The comparison of this kind of SED modeling can be used to answer the question, which DAL is the best for the particular galaxy independently of the choices of SSP and SFH? It is clear from the figure that the SB-like DAL has the highest value of Bayes factor (0.55) for the PEG ULTRA-VISTA114558, while the MW-like DAL has the highest value of Bayes factor (−0.43) for the SFG ULTRAVISTA99938. Besides, the ML-OF-BE diagram in Figure 20 shows that the SB-like DAL also provides the best explanation to the observational data of the PEG, while the SMC-like and LMC-like DALs, instead of the MW-like DAL, provide the best explanation to the observational data of the SFG. This is simply due to the much larger Bayes factor of the MW-like DAL than that of the SMC-like and LMC-like DALs for the SFG.

# 6.4. Bayesian Discrimination of SSP, SFH, and DAL for the SED Modeling of a Sample of Galaxies

In Section 6.3, we presented a detailed Bayesian discrimination of SSPs, SFHs, and DALs for the SED modeling of a PEG and an SFG. This kind of study is useful for investigating the particular characteristic of a galaxy. However, since only one object is involved in each case, the conclusions obtained for it are not necessarily suitable for other objects of the same type. Hence, in many cases, we are more interested in comparing the performance of different SSPs, SFHs, and DALs for a sample of galaxies. In this subsection, based on the method of calculating the Bayesian evidence for the SED modeling of a sample of galaxies in Section 5, we present a detailed Bayesian discrimination of different assumptions about SSP, SFH, and DAL for the SED modeling of a sample of galaxies for the first time.

# 6.4.1. The Case for All the SSP, SFH, and DAL Being Universal and Fixed

A fundamental difference between the SED modeling of an individual galaxy and the SED modeling of a sample of galaxies is that for the latter we can assume either the same SSP, SFH, and/or DAL for all objects in the sample (the universal case), or different SSPs, SFHs, and/or DALs for different objects (the object-dependent case). Hence, with the Bayesian discrimination of different assumptions in the SED modelings of a sample of galaxies, it is possible to test the universality of different SSPs, SFHs, and DALs. Here we first consider the cases where SSPs, SFHs, and DALs are all assumed to be universal.

With SSP, SFH, and DAL all being assumed to be universal, we still have the freedom of selecting them from the many possible choices. Hence, we first consider the cases where SSP, SFH, and DAL are all fixed to a specific choice. This is the most widely used assumption when modeling and interpreting the SEDs of a sample of galaxies, the structure of which is shown in Figure 6. For example, in many works, people often assume the standard model (*M*0 N) with the bc03_ch SSP, the Exp-dec SFH, and the SB-like DAL for all galaxies in their samples. By the Bayesian comparison of this kind of SED modeling, we can find out the specific combination of SSP, SFH, and DAL with the best universality for a sample of galaxies.

In Figure 21, we show the Bayes factors with respect to the standard model (*M*0 N) for all possible combinations of the 16 SSPs, 5 SFHs, and 4 DALs when modeling the SEDs of a sample of PEGs and SFGs. The combination of the BC03 SSP with a Kroupa01 IMF (bc03_kr), the Exp-dec SFH, and the SMC-like DAL has the highest value (2113.1) of Bayes factor for the PEGs, while the combination of the version of the GALEV SSP with the consideration of emission lines and a Kroupa01 IMF (galev_kr), the Exp-dec SFH, and the SB-like DAL has the highest value (5326.0) of Bayes factor for the SFGs. This is very different from the results for individual galaxies in Figure 13. Since a sample of galaxies, instead of just one object, is involved, the conclusions obtained here are with respect to the sample as a whole. Similar to the results for individual galaxies in Figure 13, the Bayes factors for the SED modeling of a sample of galaxies with different SSP models could be very different, even if the same SFH and DAL are assumed. Besides, for the PEGs, the form of SFH has the lowest value of Bayes factors for almost all combinations of SSP and DAL. For the SFGs, the Burst SFH has the lowest value of Bayes factors for almost all combinations of SSP and DAL. These general trends can be understood from the maximum likelihoods, Occam factors, and Bayesian evidences of these models in Figure 22. It can be noticed that for the PEGs, most of the models assuming a Burst SFH are located at the upper right of the figure, which indicates a low model complexity and high goodness of fit to the data, while most of the models assuming a constant SFH are located at the lower right of the figure, which indicates low model complexity but low goodness of fit to the data. On the other hand, for the SFGs, most of the models assuming a Burst SFH are located at the lower right of the figure, which indicates a low model complexity but low goodness of fit to the data, while most of the models assuming a constant SFH are located at the upper right of the figure, which indicates low model complexity and low goodness of fit to the data.

# 6.4.2. The Case for SSP, SFH, and DAL Being Universal but Only One of Them Being Fixed

In Section 6.4.1, we presented a Bayesian comparison of the SED modelings of a sample of galaxies where SSP, SFH, and

Figure 1. Flowchart for modeling and interpreting the multiwavelength photometric SED of a galaxy with BayeSED V2.0. Most parts of V2.0 are similar to that of V1.0 (see Figure 14 of Han & Han 2014). The major difference between them is the method used for SED modeling. In BayeSED V1.0, some machine learning (ML) techniques (e.g., PCA, ANN, and KNN) have been used for interpolating the model SED grid precomputed with the widely used FAST code. Instead, in BayeSED V2.0, we have built a module for modeling the SEDs of galaxies, which allow the free selection of SSP, SFH, and DAL within a large set. The ML-based methods are not used in this work but have not been abandoned. See the text for a discussion about the advantages and disadvantages of the two methods.

| Short Name | Model Family | Track/Isochrone | Spectral Library | IMF | Binary | Nebular |
| --- | --- | --- | --- | --- | --- | --- |
| bc03_ch | BC03a | Padova94+Charlot97 | BaSeL 3.1 | Chabrier03 | No | No |
| bc03_kr | BC03 | Padova94+Charlot97 | BaSeL 3.1 | Kroupa01 | No | No |
| bc03_sa | BC03 | Padova94+Charlot97 | BaSeL 3.1 | Salpeter55 | No | No |
| cb07_ch | CB07b | Padova94+Marigo07 | BaSeL 3.1 | Chabrier03 | No | No |
| cb07_kr | CB07 | Padova94+Marigo07 | BaSeL 3.1 | Kroupa01 | No | No |
| cb07_sa | CB07 | Padova94+Marigo07 | BaSeL 3.1 | Salpeter55 | No | No |
| m05_sa | M05c | Cassisi et al. (1997a, 1997b, 2000) | BaSeL 3.1 | Salpeter55 | No | No |
| m05_kr | M05 | Cassisi et al. (1997a, 1997b, 2000) | BaSeL 3.1 | Kroupa01 | No | No |
| galev0_sa | GALEVd | Padova94 | BaSeL 2.0 | Salpeter55 | No | No |
| galev0_kr | GALEV | Padova94 | BaSeL 2.0 | Kroupa01 | No | No |
| galev_sa | GALEV | Padova94 | BaSeL 2.0 | Salpeter55 | No | Yes |
| galev_kr | GALEV | Padova94 | BaSeL 2.0 | Kroupa01 | No | Yes |
| ynII_s | Yunnan-IIe | f Pols et al. (1998) | BaSeL 2.0 | Miller & Scalo (1979) | g No | No |
| ynII_b | Yunnan-II | Pols et al. (1998) | BaSeL 2.0 | Miller & Scalo (1979) | Yes | No |
| bpass_s | BPASS V2.0h | i Eldridge et al. (2008) | BaSeL 3.1 | Broken power lawj | No | No |
| bpass_b | BPASS V2.0 | Eldridge et al. (2008) | BaSeL 3.1 | Broken power law | Yes | No |

Table 1 Summary of SSP Models

#### Notes.

a http://www.bruzual.org/bc03/ b http://www.bruzual.org/cb07/ c http://www-astro.physics.ox.ac.uk/~maraston/SSPn/SED/ d http://model.galev.org/ e http://www1.ynao.ac.cn/~zhangfh/YN_SP.html f Based on the Cambridge stellar evolutionary tracks as given by the rapid stellar evolution code of Hurley et al. (2000, 2002). g This IMF is supported by the studies of Kroupa et al. (1993) and Zoccali et al. (2000). h http://www.bpass.org.uk/ i

Based on a detailed stellar evolution with a custom version of the Cambridge STARS stellar evolution code.

j A IMF with a slope of −1.30 from 0.1 to 0.5 *M* and −2.35 from 0.5 to 300 *M*, which is similar to that of Kroupa01 and Chabrier03.

DAL are all assumed to be universal and fixed to a specific choice. This is useful for finding out the combination of SSP, SFH, and DAL with the best universality for a sample of galaxies. However, we are more interested in questions such as which SSP is the best independently of the choices of SFH and DAL, which SFH is the best independently of the choices of SSP and DAL, and which DAL is the best independently of the choices of SSP and SFH. This is somewhat similar to the case for individual galaxies in Section 6.3.2. However, here we want

to obtain the conclusions for a sample of galaxies instead of that for an individual galaxy.

In Figure 23, we show the Bayes factors with respect to the standard model (*M*0 N) for the SED modelings of the PEGs and the SFGs, where a particular SSP is assumed but the SFH and DAL are free to vary. The comparison of this kind of SED modeling can be used to answer the question, which SSP is the best for all galaxies in the sample and independently of the choices of SFH and DAL? It is very clear from the figure that

Figure 2. Hierarchical structure for the *M ssp sfh dal* , , ( 0 0 0)-like SED modeling of a galaxy, where SSP, SFH, and DAL are fixed to *ssp*0, *sfh*0, and *dal*0, respectively. The black nodes indicate certain quantities (or fixed parameters), while the open nodes indicate uncertain quantities (or free parameters). The gray nodes indicate observational data with errors. In the language of Bayesian hierarchical modeling, SSP, SFH, and DAL are called hyperparameters. They are just used to indicate the different selections of the three uncertain components. For the SED modeling of galaxies, they define a 3D model space. Finally, the conditional dependences between nodes are specified with arrow lines. Hereafter, we set the *M ssp sfh dal* , , ( 0 0 0)-like SED modeling of a galaxy with ssp0 = bc03_ch, *sfh*0 = Exp-dec, and *dal*0 = SB-like as the standard model *M*0 1 .

Figure 3. Similar to Figure 2, but for the *M ssp sfh dal* , , 0 ( )-like SED modeling, where a fixed SSP (*ssp*0) and free SFH (sfh) and DAL (dal) are assumed. Here the selections of the form of SFH and DAL are considered as two additional free parameters, which will be marginalized out when comparing the different SSP models.

the BC03 SSP with a Kroupa01 IMF has the highest value of Bayes factor (2110.10) for the PEGs, while the version of the GALEV model with the consideration of emission lines and a Kroupa01 IMF has the highest value of Bayes factor (5323.00) for the SFGs. Besides, the result for all PEGs in the sample is very different from that for the particular PEG ULTRA-VISTA114558, for which the more "TP-AGB heavy" SSP models of CB07 and M05 have much larger Bayes factor than other SSPs as shown in Figure 15. Both the results for PEGs and SFGs suggest that the more "TP-AGB heavy" SSP models of CB07 and M05 are not universally better than other "TP-AGB light" models. For the PEGs, assuming the version of the BPASS V2.0 SSP without the consideration of binaries leads to a Bayes factor that is very close to that of assuming the BC03 SSPs. It can be noticed in Figure 24 that the former actually leads to a better fit to the observational data as shown by the larger maximum likelihood. However, the BC03 SSPs can lead to larger Occam factors, which implies lower model complexity. It is also worth noticing that the version of the BPASS V2.0 SSP with the consideration of binaries has the lowest Bayes factor. As shown in Figure 24, this SSP is located at the lower left of the ML-OF-BE diagram, which implies low goodness of fit to the observational data of PEGs and relatively

Figure 4. Similar to Figure 3, but for the *M ssp sfh dal* , , 0 ( )-like SED modeling, where a fixed SFH (*sfh*0) and free SSP (ssp) and DAL (dal) are assumed. Similarly, the uncertain selection of the SSP model and the form of DAL will be marginalized out when comparing the different forms of SFH.

Figure 5. Similar to Figure 3, but for the *M* (*ssp sfh dal* , , 0)-like SED modeling, where a fixed DAL (*dal*0) and free SSP (ssp) and SFH (sfh) are assumed. Similarly, the uncertain selection of the SSP model and the form of SFH will be marginalized out when comparing the different forms of DAL.

high model complexity. On the other hand, the results for the SFGs are more consistent with that for the particular SFG ULTRAVISTA99938 shown in Figures 15 and 16. However, it becomes even clearer that the version of the GALEV SSP with the consideration of nebular emission lines not only has the highest value of Bayes factor but also provides the best explanation to the observational data of the SFGs. These results suggest that the consideration of nebular emission lines is indispensable for explaining the photometric observations of the SFGs.

In Figures 25 and 26, we present a Bayesian comparison of the different forms of SFHs for the PEGs and SFGs. The results show that the commonly assumed Exp-dec SFH provides the best explanation to the observational data of both PEGs and SFGs and has the highest value of the Bayes factor, although it has the lowest value of the Occam factor and consequently the highest model complexity. Hence, the Exp-dec SFH has the best universality for both PEGs and SFGs in our sample, although it is not necessarily the best for all galaxies. Besides, the performance of the Burst SFH is much better than the constant SFH for the PEGs, while the opposite is true for the SFGs. Similarly, in Figures 27 and 28, we present a Bayesian comparison of the different forms of DALs for the PEGs and SFGs. The results show that the SMC-like DAL provides the best explanation to the observational data of PEGs and has the highest value of Bayes factor (2108.90). For the SFGs, the SBlike DAL provides the best explanation to the observational data and has the highest value of Bayes factor (5322.00). The very different SFH and DAL suggest that formation mechanisms for the PEGs and the SFGs are generally very different.

Figure 6. Hierarchical structure for the *M ssp sfh dal* , , ( 0 0 0)-like SED modeling of a sample of N galaxies, where SSP, SFH, and DAL are assumed to be universal and fixed to *ssp*0, *sfh*0, and *dal*0, respectively. Hereafter, we set the *M ssp sfh dal* , , ( 0 0 0)-like SED modeling of N galaxies with *ssp*0 = bc03_ch, *sfh*0 = Exp-dec, and *dal*0 = SB-like as the standard model *M*0 N.

Figure 7. Hierarchical structure for the *M ssp sfh sfh sfh dal dal dal* , , ,, , , ,, ( 012 ¼ ¼ *N* 1 2 *N*)-like SED modeling of a sample of N galaxies, where a universal and fixed SSP and object-dependent and free SFH and DAL are assumed.

# 6.4.3. The Case for Only One of the SSP, SFH, and DAL Being Universal and Fixed

As demonstrated in Section 6.4.2, by the Bayesian comparison of the SED modelings of a sample of galaxies where the SSP, SFH, and DAL are all assumed to be universal but only one of them is fixed to a specific choice, we can investigate the universality of different SSPs, SFHs, and DALs. However, it is not necessary to assume that SSP, SFH, and DAL are all universal when investigating the universality of only one of them. Actually, it could be even more interesting to find out which SSP model has the best universality for all galaxies in a sample without assuming a universal SFH and DAL, which form of SFH has the best universality for all galaxies in a sample without assuming a universal SSP and DAL, and which form of DAL has the best universality for all galaxies in a sample without assuming a universal SSP and SFH. Hence, by the Bayesian comparison of the SED

modelings of a sample of galaxies where only one of the SSP, SFH, and DAL is assumed to be universal and fixed to a specific choice, we can better understand the universality of different SSPs, SFHs, and DALs.

The Bayesian comparison of the *M ssp sfh sfh sfh* , , ,, , 012 ¼ *N* ( *dal dal dal* 1 2 , ,, ¼ *N*)-like SED modelings of a sample of galaxies can be used to answer the question, which SSP model has the best universality for all galaxies in the sample and independently of the SFH and DAL assumed for different galaxies? In Figure 29, we show the Bayes factors with respect to the standard model (*M*0 N) for the *M ssp sfh sfh* , , ,, ( 012 ¼ *sfh dal dal dal* , , ,, *N* 1 2 ¼ *N*)-like SED modelings of the PEGs and the SFGs where only the SSP is assumed to be universal and fixed to a particular choice while the SFH and DAL are assumed to be object dependent and free. For the PEGs, it is clear that the version of the BPASS V2.0 SSP without the consideration of binaries has the highest value of Bayes factor (1695.60), which is only slightly larger than that for the BC03

Figure 8. Left: classification of galaxies in our sample according to the definition of box regions in the UVJ color–color diagram as given by Muzzin et al. (2013a), and color-coded with SFR. Right: distribution of stellar mass for galaxies in our sample. It is clear that the galaxies in our sample are distributed widely in both the color–color and stellar mass space.

Table 2 Summary of the Free Parameters and Priors

| Parameter | Prior Range | Prior Type |
| --- | --- | --- |
| z | [0 6] | Uniform |
| obs errsys | [0 1] | Uniform |
| log age yr ( ) | [5 10.3] | Uniform and age age z < U( ) |
| log(Z Z) | [−2.30 0.70] | Uniform |
| log yr (t ) | [6 12] | Uniform |
| Av | [0 4] | Uniform |

SSPs. The maximum likelihoods and Occam factors in the left panel of Figure 30 show that the version of the BPASS V2.0 SSP without the consideration of binaries provides a much better explanation to the observational data of PEGs than the BC03 SSPs, while the latter have much larger Occam factors and consequently much lower model complexity. For the SFGs, the version of the GALEV SSP with the consideration of emission lines and a Kroupa01 IMF has the highest value of Bayes factor (3336.00), which is much larger than that of all the other SSPs. The maximum likelihoods and Occam factors in the right panel of Figure 30 show that the version of the GALEV SSP with the consideration of emission lines and a Kroupa01 IMF provides a much better explanation to the observational data of SFGs than the BC03 SSPs, while the latter have much larger Occam factors and consequently much lower model complexity. A more detailed discussion about the performance of different SSP models will be presented in Section 7.

Similarly, the Bayesian comparison of the *M ssp ssp* , , 1 2 ( , , , , ,, *ssp sfh dal dal dal* ¼ ¼ *N* 0 1 2 *N*)-like SED modelings of a sample of galaxies can be used to answer the question, which form of SFH has the best universality for all galaxies in the sample and independently of the SSP and DAL assumed for different galaxies? In Figure 31, we show the Bayes factors with respect to the standard model (*M*0 N) for the *M ssp ssp ssp sfh dal dal dal* , ,, , , , ,, ( ) 12 0 ¼ ¼ *N* 1 2 *N* -like SED modelings of the PEGs and the SFGs where only the SFH is assumed to be universal and fixed to a particular choice while the SSP and DAL are assumed to be object dependent and free.

Table 3 The Estimation of Free Parameters (in Bold Font) and Derived Parameters for the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938 with the Uniform Prior for All Free Parameters

| Parameter | PEG | SFG |
| --- | --- | --- |
| z | + 0.13 0.82 0.05 - | + 0.03 0.60 0.07 - |
| ssys | + 0.02 0.06 0.02 - | + 0.07 0.03 0.04 - |
| log yr (age ) | + 0.12 9.63 0.30 - | + 9.45 0.47 0.24 - |
| log yr (t ) | + 0.90 7.33 0.87 - | + 10.58 1.21 0.94 - |
| log Z (Z ) | + 0.54 - - 1.49 0.36 | + 0.49 - - 1.60 0.38 |
| Av mag | + 1.05 0.56 0.23 - | + 0.28 0.20 0.42 - |
| zform | + 3.86 - 2.65 1.03 | + 1.04 - 1.22 0.49 |
|  | + | + 0.26 |
| log SFR ( [ yr ] M ) | - 67.72 931.28 61.42 - + 0.10 | 0.01 0.15 - + 0.10 |
| log M ( ) M * | 10.78 0.15 - | 9.54 0.21 - |
| log ( Lbol [ erg s ] ) | + 0.06 - 44.60 0.05 | + 0.24 - 43.97 0.12 |

For the PEGs, the Exp-dec SFH has the largest Bayes factor, while the Burst SFH has the second-largest Bayes factor. For the SFGs, the Exp-dec SFH still has the largest Bayes factor, while the constant SFH has the second-largest Bayes factor. The ML-OF-BE diagram in the left panel of Figure 32 shows that the Exp-dec SFH provides a much better explanation to the observational data of PEGs than other form of SFHs. The Burst SFH has a much larger Occam factor and consequently a much lower model complexity, although it is not as good as the Expdec SFH for fitting the observational data of PEGs. Meanwhile, the ML-OF-BE diagram in the right panel of Figure 32 shows that the Exp-dec SFH also provides a much better explanation to the observational data of SFGs than other forms of SFHs. The constant SFH has a much larger Occam factor and consequently a much lower model complexity, although it is not as good as the Exp-dec SFH for fitting the observational data of SFGs.

Finally, the Bayesian comparison of the *M ssp ssp* , ,, ( 1 2 ¼ *ssp sfh sfh sfh dal* , , ,, , *N N* 1 2 ¼ 0)-like SED modelings of a sample of galaxies can be used to answer the question, which form of DAL has the best universality for all galaxies in the sample and independently of the SSP and SFH assumed for different galaxies? In Figure 33, we show the Bayes factors with respect to the standard model (*M*0 N) for the

Figure 9. 1D and 2D posterior PDFs of free parameters for the PEG ULTRAVISTA114558. They represent our state of knowledge about them. The presence of multiple peaks and/or strong correlations in the 2D PDFs indicates the degeneracies between the free parameters of the SED model.

*M ssp ssp ssp sfh sfh sfh dal* , ,, , , ,, , ( ) 1 2 12 ¼ ¼ *N N* 0 -like SED modelings of the PEGs and the SFGs where only the DAL is assumed to be universal and fixed to a particular choice while the SSP and SFH are assumed to be object dependent and free. It is clear from the figure that the SMC-like DAL has the largest Bayes factor for the PEGs, while the SB-like DAL has the largest Bayes factor for the SFGs. The ML-OF-BE diagram in the left panel of Figure 34 shows that the SMC-like DAL provides a better explanation to the observational data of PEGs than other forms of DALs and has the lowest model complexity. Meanwhile, the ML-OF-BE diagram in the right

panel of Figure 34 shows that the SB-like DAL provides a much better explanation to the observational data of SFGs than other forms of DAL, although it has a relatively large model complexity.

# 7. Discussion

As mentioned in Sections 1 and 2, there are many uncertain components in the SED modeling of galaxies. Different considerations of these uncertain components will result in very different SED predictions and very different estimations

Figure 10. Same as Figure 9, but for the SFG ULTRAVISTA99938.

about the physical parameters of galaxies (Conroy et al. 2009; Lee et al. 2009; Longhetti & Saracco 2009; Abrahamse et al. 2011; Magris et al. 2011; Dolphin 2012; Pforr et al. 2012; Kobayashi et al. 2013; Michałowski et al. 2014; Pacifici et al. 2015). Hence, it is very important to find a valid method to discriminate among the different considerations about those uncertain components in the SED modeling of galaxies. In this paper, we have proposed a new Bayesian framework to compare the SED modelings of a sample of galaxies with different assumptions about three of the major uncertain components: SSP, SFH, and DAL. We suggest that the Bayesian evidence, which is determined by the trade-off between the complexity of a model and its goodness of fit to the data, is a more reasonable and useful quantification for the performance of a model. Besides, by calculating the Bayesian evidence for the SED modeling of a sample of galaxies instead of just one galaxy, this new Bayesian framework allows us to investigate the universality of different SSPs, SFHs, and DALs. In this section, we discuss some results obtained with the first application of this new method.

# 7.1. The Universality of Different SSP Models

One of the most important uncertainties in the SED modeling of galaxies is the modeling of an SSP. As mentioned

Figure 11. Posterior PDF for the SFH of the PEG ULTRAVISTA114558 (left) and the SFG ULTRAVISTA99938 (right). Only the median and the 68% and 95% credible regions obtained from the posterior PDF of the SFH for the two galaxies are shown.

Figure 12. Results of SED fitting for the PEG ULTRAVISTA114558 (left) and the SFG ULTRAVISTA99938 (right). Except for the best-fit SED, the median and the 68% and 95% credible regions obtained from the posterior PDFs of the model SEDs are also shown. The GALEX far-UV and NUV and Spitzer IRAC 3.6 and 4.5 μm data have been labeled in the figure.

in Section 2.2, there are many uncertainties about the star formation (e.g., IMF) and evolution, stellar spectral libraries, and synthesis method in this procedure. With the different treatments of these uncertainties, different SSP models may have different limitations. However, for the study of the galaxy formation and evolution, the best SSP model should have the best universality to avoid the bias introduced by the employed SSP model. Here we discuss the results for the SSP models with a focus on their universality.

#### 7.1.1. The Contribution of TP-AGB Stars

While the importance of TP-AGB stars in the SED modeling of an SSP is well established (Maraston 2005; Bruzual 2007; Conroy et al. 2009), the appropriate treatment of them is still an open issue. Maraston et al. (2006) presented a comparison between the performance of the BC03 and M05 models, which are very different in the treatment of the TP-AGB phase, for a sample of seven passively evolving high-z galaxies. They found that the TP-AGB phase is very important for the interpretation of rest-frame near-IR data, and the M05 models give better fits to these galaxies than the BC03 models. In Kriek et al. (2010) and Zibetti et al. (2013), two samples (62 for the former and 16 for the latter) of post-starburst galaxies, where the contribution of TP-AGB stars is thought to be most prominent, have been used to discriminate the SSP models with different considerations for the contribution of TP-AGB stars. They found that the "TP-AGB light" BC03 model is more favored than the "TP-AGB heavy" M05 model, since the former can more consistently fit the rest-frame optical to near-IR parts of the SEDs of these galaxies. Capozzi et al. (2016) presented a comparison of the performance of three SSP models with heavy, mild, and light TP-AGB contribution for a sample of 51 spectroscopically confirmed high-z passive galaxies. They found that the observed SEDs of these galaxies can be best fitted by assuming a significant contribution from TP-AGB stars. Different methods have been used in these works and sometimes lead to discrepant conclusions. However, they are similar in that the performance of different models is mainly compared with their goodness of fit (as quantified by the 2 *c* ) to the observational data of a relatively small sample of galaxies.

The Bayesian evidence, which is determined by the trade-off between the complexity of a model and its goodness of fit to

Figure 13. Bayes factors with respect to the standard model (*M*0 1 , which assumes the BC03 SSP with a Chabrier03 IMF, Exp-dec SFH, and SB-like DAL) for the *M ssp sfh dal* , , ( 0 0 0)-like SED modelings of the PEG ULTRAVISTA114558 (left) and the SFG ULTRAVISTA99938 (right), where SSP (see Table 1 for the meaning of each SSP model), SFH, and DAL are all fixed to a particular choice. The dotted lines show the values of the Bayes factor with a step of 1.5. The value of Bayes factor for the model with the highest Bayes factor is also shown in the figure. For the PEG ULTRAVISTA114558, the combination of the M05 SSP with a Salpeter55 IMF (m05_sa), the Burst SFH, and the SB-like DAL has the highest value (2.71) of Bayes factor. For the SFG ULTRAVISTA99938, the combination of the version of the GALEV SSP with the consideration of emission lines and with a Kroupa01 IMF (galev_kr), the Exp-dec SFH, and the LMC-like DAL has the highest value (1.19) of Bayes factor. The positive value of Bayes factor indicates that the model has higher Bayesian evidence than the standard model *M*0 1 .

Figure 14. Maximum likelihood vs. Occam factor diagram vs. Bayesian evidence diagram (hereafter the ML-OF-BE diagram) for the *M ssp sfh dal* , , ( 0 0 0)-like SED modelings of the PEG ULTRAVISTA114558 (left) and the SFG ULTRAVISTA99938 (right) where SSP, SFH, and DAL are all fixed to a particular choice. The maximum likelihood, which is defined in Equation (18) and directly related to the min 2 *c* , represents the goodness of fit to the data of a model. The Occam factor, which is defined in Equation (17), represents the complexity of a model. The Bayesian evidence, which is defined in Equation (16) and indicated as dotted lines with a step of 1.5, is just the product of the maximum likelihood and Occam factor. The four different colors represent the models with different assumptions about the DAL, while the five different shapes represent the models with different assumptions about the SFH. For a given color and shape, there are 16 points, representing models with different assumptions about the SSP model.

the data, could be a more reasonable and useful quantification for the performance of a model. In Section 6, we have employed the Bayesian evidence to compare the performance of different SSP models for individual galaxies (Section 6.3) and a sample of galaxies (Section 6.4). The results in Figures 15 and 16 show that the more "TP-AGB heavy" models of CB07 and M05 have larger Bayesian evidence and goodness of fit to the data for the specific PEG ULTRA-VISTA114558, which is just the opposite of what is the case for the specific SFG ULTRAVISTA99938. Although this result is robust against the different choices of SFH and DAL for the two galaxies, it may not be representative of a population of galaxies.

In Figures 23 and 24, we have compared the performance of different SSPs for a sample of 1159 PEGs and a sample of 4308 SFGs, where the SSP, SFH, and DAL are all assumed to be universal but only the SSP is fixed to a particular choice. For both the samples of PEGs and SFGs, the results suggest that the more "TP-AGB heavy" models of CB07 and M05 are not universally better than other "TP-AGB light" models either in the sense of the Bayesian evidence or in the goodness of fit to the data alone. Furthermore, in Figures 29 and 30, we have compared the performance of different SSPs for the sample of PEGs and SFGs without assuming a universal SFH and DAL for all galaxies to obtain more robust results. Interestingly, the obtained results are basically the same.

Figure 15. Similar to Figure 13, but for the *M ssp sfh dal* , , 0 ( )-like SED modelings where a fixed SSP and free SFH and DAL are assumed. The CB07 SSP with a Chabrier03 IMF (cb07_ch) has the highest value of Bayes factor (1.02) for the PEG ULTRAVISTA114558, while the version of the GALEV model with the consideration of emission lines and a Kroupa01 IMF (galev_kr) has the highest value of Bayes factor (−0.02) for the SFG ULTRAVISTA99938. The negative Bayes factor indicates that even the best one of the *M ssp sfh dal* , , 0 ( )-like SED modelings is not better than the standard model *M*0 1 , which is an *M ssp sfh dal* , , ( 0 0 0)-like SED modeling. Hence, for the SFG ULTRAVISTA99938, the additional complexity as introduced by the two additional free parameters (sfh and dal) is not justified by a much better fit to the observational data. However, the two additional parameters are still useful to make sure the comparison of SSP models is independent of the selection of SFH and DAL. For the comparison of any two SSP models, only the difference of their Bayes factors is meaningful to us.

Figure 16. Similar to Figure 14, but for the *M ssp sfh dal* , , 0 ( )-like SED modelings where a fixed SSP and free SFH and DAL are assumed. Here the Occam factor and maximum likelihood are defined in Equations (22) and (23), respectively. The Bayesian evidence is defined in Equation (20) and calculated with Equation (26). The more "TP-AGB heavy" SSP models of CB07 (cb07_ch, cb07_kr, cb07_sa) and M05 (m05_kr, m05_sa) provide much better fits to the observational data of the PEG ULTRAVISTA114558, while the version of the GALEV model with the consideration of emission lines (galev_kr, galev_sa) provides much better fits to the observational data of the SFG ULTRAVISTA99938.

Hence, the results of our Bayesian model comparison with a sample of galaxies do not support the more "TP-AGB heavy" model of either CB07 or M05. It is worth noticing that the performances of the CB07 and M05 models are somewhat similar, although the different stellar tracks and synthesis methods have been employed by them. Besides, the BC03 and CB07 models are only different in the treatment of TP-AGB stars, while the BC03 models obviously have better performance than the CB07 models. These results suggest that a universally appropriate treatment of the TP-AGB phase is still not well established in the current SSP models. This may not be so surprising given the large number of uncertainties involved in the modeling of the TP-AGB phase (Conroy et al. 2009; Marigo et al. 2013; Rosenfield et al. 2014, 2016).

It is important to mention that the results obtained with Bayesian model comparison are always data dependent as clearly shown in Equation (12). Hence, the above conclusion could depend on the sample of galaxies used in this paper. Indeed, most galaxies in our sample are at low redshift (mostly with z 1), where the contribution of TP-AGB stars is thought to be less important. However, it is still not easy to understand why the more sophisticated treatments of TP-AGB stars result in SSP models that have a poorer performance for low-redshift galaxies. Since the more "TP-AGB heavy" models of CB07 and M05 are primarily tested for galaxies at higher redshifts, where the contribution of TP-AGB stars is thought to be very important, the models could be overly tuned for those galaxies. We will test this with the comparison of the results obtained for low-redshift and high-redshift galaxies in a future work.

Figure 17. Similar to Figure 15, but for the *M ssp sfh dal* , , 0 ( )-like SED modelings where a fixed SFH and free SSP and DAL are assumed. The Burst SFH has the largest value of Bayes factor (0.56) for the PEG ULTRAVISTA114558 (left), while the constant SFH has the largest value of Bayes factor (−0.35) for the SFG ULTRAVISTA99938 (right).

Figure 18. Similar to Figure 16, but for the *M ssp sfh dal* , , 0 ( )-like SED modelings where a fixed SFH and free SSP and DAL are assumed. For the PEG ULTRAVISTA114558, the Burst SFH provides the best fit to the data and has the highest value of Occam factor (i.e., the lowest model complexity). For the SFG ULTRAVISTA99938, the Exp-dec SFH provides the best fit to the data but has the lowest value of Occam factor (i.e., the highest model complexity).

#### 7.1.2. The Consideration of Binary Star Interaction

The presence of a nearby companion may alter the evolution of a star significantly by their interactions. It is observationally well established that a large fraction of stars, especially the massive ones, are in binary or higher-order multiple systems (Sana et al. 2012; Duchêne & Kraus 2013). Hence, physically, it is very important to consider the effects of binary star interaction in the SED modeling of a stellar population.

We have employed two publicly available SSP models (Yunnan-II and BPASS V2.0), which have included the effects of binary star interactions to test the importance of binaries. Both the versions with and without binaries of the two models have been considered. In Figures 23 and 24, we show the results obtained for the case that the SSP, SFH, and DAL are all assumed to be universal but only the SSP is fixed to a particular choice for a sample of 1159 PEGs and a sample of 4308 SFGs. It is clear from the figures that for both the samples of PEGs and SFGs, the version of the Yunnan-II model with binaries is much better than the version without binaries. Surprisingly, the

version of the BPASS V2.0 model with binaries is even worse than the version without binaries. In Figures 29 and 30, we further considered the case without assuming a universal SFH and DAL for all galaxies in the sample of PEGs or SFGs. It is even clearer that the version of the Yunnan-II model with binaries is much better than the version without binaries, especially for the sample of PEGs. However, the version of the BPASS V2.0 model with binaries is still much worse than the version without binaries. As shown in Figures 24 and 30, the version of the BPASS V2.0 model with binaries is always located at the lower left of the ML-OF-BE diagram, which indicates a low goodness of fit to the data and a high degree of model complexity.

Given the limitation of the BPASS V2.0 model as mentioned in Section 2.2.5, the above results are not so surprising. Eldridge et al. (2017) stated that the BPASS code was initially established for young stellar populations, and they do not recommend the current version of the code for fitting the stellar populations much older than 1 Gyr. Since most galaxies in our

Figure 19. Similar to Figure 15, but for the *M* (*ssp sfh dal* , , 0)-like SED modeling where a fixed DAL and free SSP and SFH are assumed. The SB-like DAL has the highest value of Bayes factor (0.55) for the PEG ULTRAVISTA114558 (left), while the MW-like DAL has the highest value of Bayes factor (−0.43) for the SFG ULTRAVISTA99938 (right).

Figure 20. Similar to Figure 16, but for the *M* (*ssp sfh dal* , , 0)-like SED modelings where a fixed DAL and free SSP and SFH are assumed. For the PEG ULTRAVISTA114558, the SB-like DAL provides a better fit to the data and has the largest Bayesian evidence. For the SFG ULTRAVISTA99938, the SMC-like and LMC-like DALs provide better fits to the data, but the MW-like DAL has the largest Bayesian evidence.

sample are located at z 1, the contribution of the stellar populations much older than 1 Gyr cannot be ignored (see Tables 3 for the two examples). Actually, we obtained the above results long before the publication of the Eldridge et al. (2017) paper, where the limitations of the model were first pointed out in detail. Hence, the states in Eldridge et al. (2017) are really an independent support for the effectiveness of our Bayesian model comparison method. In Stanway & Eldridge (2018), the authors stated that some issues about binary evolution have been partly addressed in their recently released V2.2 models. We would like to check this in a following work.

Meanwhile, it is important to notice in Figures 29 and 30 that the version of the BPASS V2.0 model without binaries is actually better than both the versions of the Yunnan-II model with and without binaries. A possible reason for this result is that the BPASS model is based on a detailed stellar evolution calculation with the Cambridge STARS stellar evolution code instead of the approximate and rapid stellar evolution code of Hurley et al. (2000, 2002) as employed by the Yunnan-II model. Besides, a Monte Carlo binary population synthesis technique has been employed in the Yunnan-II model, which could drive the differences with the BPASS V2.0 model.

#### 7.1.3. The Universality of IMF

Some recent works (Davé 2008; van Dokkum 2008; Conroy & van Dokkum 2012; van Dokkum & Conroy 2012) suggest that the IMF might not be universal but could be evolving with the mass and redshift of the galaxies. By using the Bayesian model comparison method for a sample of galaxies, it is possible to compare the SED modeling assuming a universal IMF and that assuming an evolving IMF. However, all the SSPs employed in this work assume a universal IMF. Hence, here we just want to compare the degree of universality of different IMFs.

The results in Figure 29 show that it is possible to compare the degree of universality of SSPs with different assumptions about the IMF. To make it clearer, in Table 4, we show the

Figure 21. Bayes factors with respect to the standard model (*M*0 N, which assumes the BC03 SSP with a Chabrier03 IMF, Exp-dec SFH, and SB-like DAL for all galaxies in the sample) for the *M ssp sfh dal* , , ( 0 0 0)-like SED modelings of PEGs (left) and SFGs (right), where the SSP, SFH, and DAL are all assumed to be universal and fixed to a particular choice. The dotted lines show the values of the Bayes factor with a step of 10,000. For the PEGs, the combination of the BC03 SSP with a Kroupa01 IMF (bc03_kr), the Exp-dec SFH, and the SMC-like DAL has the highest value (2113.10) of Bayes factor. For the SFGs, the combination of the version of the GALEV SSP with the consideration of emission lines and a Kroupa01 IMF (galev_kr), the Exp-dec SFH, and the SB-like DAL has the highest value (5326.00) of Bayes factor. Since a sample of galaxies, instead of just one object (as in Figure 13), is involved, the conclusions obtained here are with respect to the sample as a whole.

Figure 22. ML-OF-BE diagram for the *M ssp sfh dal* , , ( 0 0 0)-like SED modelings of PEGs (left) and SFGs (right), where SSP, SFH, and DAL are all assumed to be universal and fixed to a particular choice. Here the Occam factor and maximum likelihood are defined in Equations (31) and (32), respectively. The Bayesian evidence is defined in Equation (29) and calculated with Equation (33). The dotted lines show the values of the Bayesian evidence with a step of 10,000.

detailed value of Bayes factors of different SSPs for PEGs and SFGs, which are just the same as in Figure 29. For all the BC03, CB07, and GALEV models, the version of them assuming a Kroupa01 IMF has a much larger Bayes factor than the version assuming a Salpeter55 IMF for both PEGs and SFGs. The only exception is the M05 model, which obviously favors the Salpeter55 IMF for both PEGs and SFGs. A possible reason for this is that the population synthesis method employed by the M05 model is very different from that employed by other models. It is also worth noticing that the M05 model is more sensitive to the selection of IMF than other models and has the lowest value of Occam factor as shown in Figure 30. Generally, our results suggest that the IMFs of stellar population in PEGs and SFGs are not likely to be systematically different and the Kroupa01 IMF is more universal than the Salpeter55 IMF.

#### 7.1.4. The Importance of Nebular Emissions

The importance of including the contribution of nebular emission lines to the broadband fluxes of galaxies with active star formation has been well documented in the literature (Charlot & Longhetti 2001; Zackrisson et al. 2008; Ilbert et al. 2009; Schaerer & de Barros 2009; Schenker et al. 2013; Stark et al. 2013; de Barros et al. 2014). For example, Ilbert et al. (2009) show that the flux of nebular emission lines can change the color by about 0.4 mag, and a reasonable treatment of emission lines can decrease the dispersion of photo-z estimation by a factor of 2.5. Here we discuss the results about nebular emission obtained with the Bayesian model comparison method for a sample of galaxies developed in this paper.

The results in Figure 29 and Table 4 show that the version of the GALEV SSP with the consideration of emission lines has a

Figure 23. Similar to Figure 21, but for the *M ssp sfh dal* , , 0 ( )-like SED modelings, where the SSP, SFH, and DAL are all assumed to be universal, but a fixed SSP and free SFH and DAL are assumed. The dotted lines show the values of the Bayes factor with a step of 1000. The BC03 SSP with a Kroupa01 IMF (bc03_kr) has the highest value of Bayes factor (2110.10) for the PEGs (left), while the version of the GALEV SSP with the consideration of emission lines and a Kroupa01 IMF (galev_kr) has the highest value of Bayes factor (5323.00) for the SFGs (right).

Figure 24. Similar to Figure 22, but for the *M ssp sfh dal* , , 0 ( )-like SED modelings, where the SSP, SFH, and DAL are all assumed to be universal, but a fixed SSP and free SFH and DAL are assumed. Here the Occam factor and maximum likelihood are defined in Equations (36) and (37), respectively. The Bayesian evidence is defined in Equation (34) and calculated with Equation (40). The dotted lines show the values of the Bayesian evidence with a step of 1000. The version of the BPASS V2.0 SSP model without the consideration of binaries (bpass_s) provides the best fit to the observational data of the PEGs (left), while the version of the GALEV SSP models with the consideration of emission lines (galev_kr, galev_sa) provides the best fits to the observational data of the SFGs (right).

significantly larger Bayes factor than all the other SSPs without the consideration of emission lines for the SFGs. The maximum likelihoods in the right panel of Figure 30 show that this model can provide a significantly better fit to the observational data of SFGs than others, although it has a relatively smaller Occam factor and consequently a higher model complexity than most of the others. Hence, it is clear that the nebular emission lines are indeed very important for the SFGs. However, for the PEGs, the Bayes factors of the version of the GALEV SSP with and without the consideration of emission lines are not larger than most of the other models. The results in the left panel of Figure 30 show that the GALEV models provide a poorer fit to the observational data of PEGs than most of the other models, although they have the largest value of Bayes factor. These results suggest that, for the modeling of stellar emission, the GALEV models are not more sophisticated than other models. Hence, it is very likely that

other SSP models would perform much better for SFGs when a reasonable consideration of nebular emission is included in them. Unfortunately, without the version of them with nebular emissions self-consistently included, we cannot test this with the Bayesian model comparison method developed in this work.

# 7.2. The Universality of Different Forms of SFH

In theory, due to the different environmental influences and formation conditions, the detailed SFHs of different galaxies are expected to be very different. However, when the details in the SFHs are smoothed out, the general shape of them could be more similar. In practice, the Exp-dec SFH has been widely employed in many works as if it is universal for all galaxies. This assumption has been doubted in many works (Lee et al. 2010; Maraston et al. 2010; Pforr et al. 2012; Reddy et al. 2012; Lee et al. 2014), and many authors have suggested some

Figure 25. Similar to Figure 23, but for the *M ssp sfh dal* , , 0 ( )-like case, where a fixed SFH and free SSP and DAL are assumed. The commonly assumed Exp-dec SFH has the highest values of Bayes factor for both PEGs (left; 2108.90) and SFGs (right; 5322.00).

Figure 26. Similar to Figure 24, but for the *M ssp sfh dal* , , 0 ( )-like case, where a fixed SFH and free SSP and DAL are assumed. For both PEGs and SFGs, the widely used Exp-dec SFH provides the best explanation to the observational data, although it has the lowest value of Occam factor (i.e., the highest model complexity).

Figure 27. Similar to Figure 23, but for the *M* (*ssp sfh dal* , , 0)-like case, where a fixed DAL and free SSP and SFH are assumed. The SMC-like DAL has the highest value of Bayes factor (2108.70) for the PEGs, while the SB-like DAL has the highest value of Bayes factor (5321.00) for the SFGs.

Figure 28. Similar to Figure 24, but for the *M* (*ssp sfh dal* , , 0)-like case, where a fixed DAL and free SSP and SFH are assumed. The SMC-like DAL provides the best explanation to the observational data of PEGs (left), while the SB-like DAL provides the best explanation to the observational data of SFGs (right).

Figure 29. Similar to Figure 23, but for the *M ssp sfh sfh sfh dal dal dal* , , ,, , , ,, ( 012 ¼ ¼ *N* 1 2 *N*)-like SED modelings where a universal and fixed SSP and objectdependent and free SFH and DAL are assumed. The version of the BPASS V2.0 SSP without the consideration of binaries (bpass_s) has the highest value (1695.60) of Bayes factor for the PEGs (left), while the version of the GALEV SSP with the consideration of emission lines and a Kroupa01 IMF (galev_kr) has the highest value (3336.00) of Bayes factor for the SFGs (right).

more complicated (Gladders et al. 2013; Abramson et al. 2016; Ciesla et al. 2017; Diemer et al. 2017; Carnall et al. 2018) or more physically motivated (Finlator et al. 2007; Pacifici et al. 2012; Iyer & Gawiser 2017) forms of SFHs. In most previous works, the different forms of SFH are mainly compared by their goodness of fit to the observational data. Apparently, a more complicated form of SFH tends to provide a better fit to the data. However, this additional complexity is not necessarily well supported by the data.

Here we discuss the comparison of different forms of SFH with the Bayesian evidence, which is determined by the tradeoff between the complexity of a model and its goodness of fit to the data. In Figure 17, we have compared the different forms of SFH for the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938. The Burst SFH has the largest Bayesian evidence for the PEG ULTRAVISTA114558 as shown in the left panel of Figure 17. However, the maximum likelihoods in the left panel of ML-OF-BE diagram 18 show that its goodness of fit to the data is similar to that of the Exp-dec, Exp-inc, and Delayed SFHs. Actually, it has a much larger Occam factor and consequently much smaller model complexity than the others. The trade-off between its model complexity and goodness of fit to the data finally leads to the largest Bayesian evidence. On the other hand, the constant SFH has the largest Bayesian evidence as shown in the right panel of Figure 17 for the SFG ULTRAVISTA99938. Although it has the largest Occam factor as shown in the right panel of Figure 18, its goodness of fit to the data is much smaller than the Exp-dec SFH, which actually provides the best fit to the data. Interestingly, the trade-off between its model complexity and goodness of fit to the data still leads to the largest Bayesian evidence. These results suggest that a simple definition of Occam factor similar to that in Equation (22) can provide results that are basically consistent with our intuition about the complexity of a model. However, it seems meaningless to talk about the absolute complexity of a model in the sense of this definition without mentioning a particular object.

The above results are obtained for a particular PEG and SFG. They are not necessarily representative of a whole population of galaxies. In Figure 31, we have compared the universality of

Figure 30. Similar to Figure 24, but for the *M ssp sfh sfh sfh dal dal dal* , , ,, , , ,, ( 012 ¼ ¼ *N* 1 2 *N*)-like SED modelings where a universal and fixed SSP and objectdependent and free SFH and DAL are assumed. Here the Occam factor and maximum likelihood are defined in Equations (45) and (46), respectively. The Bayesian evidence is defined in Equation (43) and calculated with Equation (52). The version of the BPASS V2.0 SSP without the consideration of binaries (bpass_s) provides the best fit to the observational data of PEGs (left), while the version of the GALEV SSP models with the consideration of emission lines (galev_kr, galev_sa) provides the best fits to the observational data of SFGs (right).

Figure 31. Similar to Figure 29, but for the *M ssp ssp ssp sfh dal dal dal* , ,, , , , ,, ( 12 0 ¼ ¼ *N* 1 2 *N*)-like case where a universal and fixed SFH and object-dependent and free SSP and DAL are assumed. The commonly assumed Exp-dec SFH has the highest values of Bayes factor for both PEGs (left; 1917.30) and SFGs (right; 3440.00).

different forms of SFH for the sample of PEGs and SFGs. Since the results are obtained without assuming a universal SSP and a universal DAL, they are very robust against the choice of SSP and DAL for different galaxies. Interestingly, the results show that the Exp-dec SFH, which is the most widely used form of SFH in the literature, has the best universality for both PEGs and SFGs in our sample. Besides, the maximum likelihoods in Figure 32 show that the Exp-dec SFH also provides generally the best goodness of fit to the observational data of both PEGs and SFGs, although it has the smallest Bayes factor and consequently the largest model complexity. These results show that the Exp-dec SFH is the most successful at explaining the multiwavelength photometric observations of a relatively large sample of low-redshift galaxies. However, since the results obtained with Bayesian model comparison are always data dependent, as clearly shown in Equation (12), the results for galaxies at higher redshifts could be very different. We will check this in a future work.

# 7.3. The Universality of Different Forms of DAL

An assumption about the effects of dusty ISM on the observed SEDs of galaxies is necessary when deriving the physical properties of galaxies. The most widely used assumption is a uniform empirical or analytical attenuation law as a simple screen. However, some works suggested that the dust laws are likely to be nonuniversal for galaxies with different types and redshifts. For example, Kriek & Conroy (2013) have utilized the stacked photometric SEDs to explore the variation of DAL in 0.5 < z < 2.0 galaxies. They found that the best-fit DAL varies with the spectral type of the galaxy, with more active galaxies having shallower DALs. Salmon et al. (2016) show that some individual galaxies at z ∼ 1.5–3 from CANDELS have strong Bayesian evidence in favor of one particular dust law. Besides, they found that the shallower SBlike DAL is more favored by galaxies with high color excess, while the steeper SMC-like DAL is more favored by galaxies with low color excess. With the CIGALE (Noll et al. 2009)

Figure 32. Similar to Figure 30, but for the *M ssp ssp ssp sfh dal dal dal* , ,, , , , ,, ( 12 0 ¼ ¼ *N* 1 2 *N*)-like case where a universal and fixed SFH and object-dependent and free SSP and DAL are assumed. For both PEGs (left) and SFGs (right), the widely used Exp-dec SFH provides the best explanation to the observational data, although it has the lowest value of Occam factor (i.e., the highest model complexity).

Figure 33. Similar to Figure 29, but for the *M ssp ssp ssp sfh sfh sfh dal* , ,, , , ,, , ( 1 2 12 ¼ ¼ *N N* 0)-like case where a universal and fixed DAL and object-dependent and free SSP and SFH are assumed. The SMC-like DAL has the highest value (2267.60) of Bayes factor for the PEGs (left), while the SB-like DAL has the highest value (2087.00) of Bayes factor for the SFGs (right).

SED fitting code, Salim et al. (2018) studied the dust attenuation curves of 230,000 individual galaxies in the local universe, including PEGs and intensely SFGs. Similar to Salmon et al. (2016), they found a strong correlation between the attenuation curve slope and the optical opacity (Av), with more opaque galaxies having shallower curves. These results are consistent with the predictions based on some radiative transfer models (Chevallard et al. 2013).

An important difference between our method and that of Salmon et al. (2016) is that we have calculated the Bayesian evidence of different DALs with the marginalization over not only the stellar population parameters but also the different choices of the SSP model and the form of SFH. By using this method, more robust results about the DAL can be obtained. In Figure 19, we have compared the performance of different DALs with the Bayesian evidence for the PEG ULTRA-VISTA114558 and the SFG ULTRAVISTA99938. The results show that the SB-like DAL is more favored by the PEG and the MW-like DAL is more favored by the SFG. However, the ML-OF-BE diagram in Figure 20 shows that the more favored

DALs do not necessarily provide a much better fit to the data, although they do have a relatively larger Occam factor, which indicates lower model complexity for the two galaxies.

Another very important difference between our method and that of Salmon et al. (2016) is that we have defined the Bayesian evidence for the SED modeling of a sample of galaxies in addition to that for individual galaxies. In Figure 33, we have compared the performance of different DALs for the SED modeling of a sample of PEGs and SFGs. By using the Bayesian evidence defined for the SED modeling of a sample of galaxies, we find that the steeper SMC-like DAL is systematically more favored by the PEGs, while the shallower SB-like DAL is systematically more favored by the SFGs. Besides, the ML-OF-BE diagram in Figure 34 shows that the SMC-like DAL also provides the best fit to the observational data of PEGs, while the SB-like DAL also provides the best fit to the observational data of SFGs. Since these results are obtained without assuming a universal SSP and SFH, they should be more robust. As shown in Figure 35, for the sample used in this work, the SFGs have a mean value of optical

Figure 34. Similar to Figure 30, but for the *M ssp ssp ssp sfh sfh sfh dal* , ,, , , ,, , ( 1 2 12 ¼ ¼ *N N* 0)-like case where a universal and fixed DAL and object-dependent and free SSP and SFH are assumed. The SMC-like DAL provides the best explanation to the observational data of PEGs (left) and has the lowest model complexity, while the SB-like DAL provides the best explanation to the observational data of SFGs (right) but has a relatively large model complexity.

Table 4 The Detailed Value of Bayes Factor for the 16 SSPs as in Figure 29

| SSP | PEGs(1159) | SFGs(4308) |
| --- | --- | --- |
| bc03_ch | 1494.00 | −1221.00 |
| bc03_kr | 1529.40 | −1143.00 |
| bc03_sa | 1329.60 | −1332.00 |
| cb07_ch | −1953.70 | −6551.00 |
| cb07_kr | −1893.60 | −6336.00 |
| cb07_sa | −2250.50 | −6475.00 |
| galev0_kr | −362.60 | −2131.00 |
| galev0_sa | −837.60 | −12202.00 |
| galev_kr | −489.60 | 3336.00 |
| galev_sa | −938.50 | 3183.00 |
| m05_kr | −1219.40 | −10541.00 |
| m05_sa | −272.40 | −6250.00 |
| ynII_s | −329.80 | −5155.00 |
| ynII_b | 1053.70 | −4961.00 |
| bpass_s | 1695.60 | −4322.00 |
| bpass_b | −3100.80 | −15547.00 |

opacity larger than that of PEGs. Hence, basically, our results are consistent with the findings of Salmon et al. (2016) and Salim et al. (2018) and the prediction of Chevallard et al. (2013) based on radiative transfer models. However, our results are based on the assumption of a universal DAL. We have tried to find out which DAL is better if it is assumed to be universal. Hence, the results may highly depend on the used sample if the attenuation curve is actually object dependent. Salim et al. (2018) have used a much larger sample than us. They show that the average attenuation curve of local star-forming galaxies in their sample is almost as steep as that of SMC. With a parameterized form of DAL, a more detailed investigation of the variation of DAL in different galaxies and its possible evolution with redshift will be the subject of a future work.

# 8. Summary and Conclusions

In this work, we have proposed a new method to define the Bayesian evidence for the SED modeling of an individual galaxy and a sample of galaxies. With the application of the newly defined Bayesian evidences and the new version of our BayeSED code to a Ks-selected, low-redshift (z 1) sample in

Figure 35. Distribution of optical opacity *A*v for the PEGs and SFGs in our sample. The vertical lines indicate the mean of the two distributions. On average, the SFGs have larger optical opacity than PEGs.

the COSMOS/UltraVISTA field, we have demonstrated a comprehensive Bayesian discrimination of the different assumptions about SSP, SFH, and DAL in the SED modeling of galaxies.

We summarize our main results as follows:

- 1. The more "TP-AGB heavy" SSP models of CB07 and M05 are not systematically more favored by both PEGs and SFGs in our sample, although they could be favored by some individual galaxies.
- 2. A reasonable consideration of binaries is important for the SED modeling of both PEGs and SFGs. For the two publicly available SSP models with the consideration of binaries, the Yunnan-II model is more favored than the BPASS V2.0 model by both the PEGs and SFGs in our sample.
- 3. For both the PEGs and SFGs in our sample, the Kroupa01 IMF is systematically more favored than that of Salpeter55.
- 4. A simple but reasonable consideration of nebular emission lines, such as that implemented in the GALEV SSP model, can significantly improve the performance of the SED modeling of SFGs.

Figure 36. 1D posterior PDFs of free parameters assuming the U:Uniform(a,b), N1:Normal *a*, *b a* 2 *m s* = = - ( ), N2:Normal , *ba ba* 2 2 *m s* = = + - ( ), and N3:Normal *b*, *b a* 2 *m s* = = - ( ) priors, for the PEG ULTRAVISTA114558 (left) and the SFG ULTRAVISTA99938 (right).

| Table 5 |
| --- |
| The Parameter Estimation and Bayesian Evidence with Different Priors for the PEG ULTRAVISTA114558 |

| Parameter | U | N1 | N2 | N3 |
| --- | --- | --- | --- | --- |
| z | + 0.13 0.82 0.05 - | + 0.85 0.07 0.12 - | + 0.83 0.06 0.13 - | + 0.03 0.79 0.11 - |
| ssys | + 0.02 0.06 0.02 - | + 0.03 0.05 0.02 - | + 0.03 0.05 0.02 - | + 0.06 0.02 0.03 - |
| log yr (age ) | + 0.12 9.63 0.30 - | + 9.54 0.24 0.18 - | + 9.57 0.27 0.16 - | + 0.11 9.65 0.17 - |
| yr (t ) | + 0.90 | + 0.90 | + | + 0.73 |
| log | 7.33 0.87 - + | 7.12 0.72 - + | 7.42 0.86 0.83 - + | 8.15 1.13 - + |
| log Z (Z ) | 0.54 - - 1.49 0.36 | 0.50 - - 1.35 0.48 | 0.55 - - 1.36 0.42 | 0.50 - - 1.41 0.37 |
| Av mag | + 1.05 0.56 0.23 - | + 0.33 0.93 0.53 - | + 0.31 0.98 0.55 - | + 0.31 1.15 0.32 - |
| zform | + 3.86 - 2.65 1.03 | + 3.14 - 1.98 0.41 | + 3.00 2.16 0.59 - | + 2.82 1.20 2.95 - |
| log SFR ( [ ] M ) yr | + - 67.72 931.28 61.42 - | + 89.06 - 99.41 899.59 - | + 42.50 - 48.00 951.00 - | + 8.99 - - 9.58 159.99 |
| log M ( ) M * | + 0.10 10.78 0.15 - | + 0.14 - 10.72 0.09 | + 0.12 10.74 0.12 - | + 0.09 10.75 0.19 - |
|  | + | + | + | + |
| log ( Lbol [ erg s ] ) | 0.06 - 44.60 0.05 | 0.05 44.60 0.05 - | 0.05 44.60 0.06 - | 0.06 - 44.57 0.11 |
| ln(Evidence) | + 0.18 - - 14.36 0.18 | + 0.17 - - 13.30 0.17 | + 0.18 - - 14.93 0.18 | + 0.20 - - 17.91 0.20 |

Note. The bold values are free parameters.

| Parameter | U | N1 | N2 | N3 |
| --- | --- | --- | --- | --- |
| z | + 0.03 0.60 0.07 - | + 0.04 0.60 0.06 - | + 0.59 0.06 0.04 - | + 0.05 0.58 0.06 - |
| ssys | + 0.04 - 0.07 0.03 | + 0.03 0.07 0.03 - | + 0.03 0.07 0.03 - | + 0.03 0.07 0.03 - |
| log yr (age ) | + 9.45 0.47 0.24 - | + 0.51 9.03 0.70 - | + 9.25 0.49 0.36 - | + 0.29 9.35 0.37 - |
| log yr (t ) | + 10.58 1.21 0.94 - | + 1.37 9.33 1.60 - | + 10.10 1.26 1.14 - | + 0.80 10.85 1.14 - |
| log Z (Z ) | + 0.49 - - 1.60 0.38 | + 0.62 - - 1.57 0.42 | + 0.60 - - 1.52 0.40 | + 0.53 - - 1.42 0.41 |
| Av mag | + 0.42 - 0.28 0.20 | + 0.39 0.46 0.32 - | + 0.42 0.40 0.28 - | + 0.45 0.36 0.25 - |
| zform | + 1.04 - 1.22 0.49 | + 0.71 0.78 0.15 - | + 0.85 0.92 0.25 - | + 0.88 1.02 0.31 - |
| log SFR ( [ ] M ) yr | + 0.26 0.01 0.15 - | + 0.35 - - 0.02 0.42 | + 0.31 0.05 0.20 - | + 0.30 0.07 0.17 - |
| log M ( ) | + 0.10 9.54 0.21 | + 0.16 - 9.44 0.20 | + 0.13 9.49 0.22 | + 0.12 9.50 0.21 |
| M * | - |  | - | - |
| log ( Lbol [ erg s ] ) | + 0.24 - 43.97 0.12 | + 0.27 44.05 0.16 - | + 0.28 - 44.02 0.15 | + 0.29 44.00 0.15 - |
| ln(Evidence) | + 0.17 - - 25.56 0.17 | + 0.16 - - 24.81 0.16 | + 0.17 - - 25.94 0.17 | + 0.18 - - 28.55 0.18 |

Table 6 Same as Table 5, but for the SFG ULTRAVISTA99938

Note. The bold values are free parameters.

- 5. The widely used Exp-dec SFH is the one best supported by the multiwavelength photometric data of both PEGs and SFGs in our sample, although it is not necessarily more physically reasonable than others.
- 6. For the galaxies in our sample, the SMC-like DAL is systematically more favored by the PEGs, while the SB-like DAL is systematically more favored by the SFGs.

The above results are either obvious or understandable in the context of galaxy physics. Hence, we conclude that the Bayesian evidence, which is determined by the trade-off between the complexity of a model (quantified by the Occam factor) and its goodness of fit to the data (quantified by the maximum likelihood), is very useful for discriminating the different assumptions in the SED modeling of galaxies. By using the Bayesian evidence marginalized over not only the normal parameters but also the different choices of all the irrelevant and uncertain components, it is possible to obtain much more robust conclusions. Especially, the Bayesian evidence defined for the SED modeling of a sample of galaxies allows us to compare the universality of any assumption made in the modeling procedure. This opens the door for many interesting investigations. Based on a simple procedure and widely used SSPs, SFHs, and DALs to model the SEDs of galaxies, we have demonstrated the usefulness of the Bayesian model comparison method, evaluated its effectiveness, and built a reference for future works. In the future, with a more flexible and sophisticated SED modeling procedure, we will apply the Bayesian method developed in this work to a larger sample of galaxies covering a much larger redshift range.

We thank an anonymous referee for his/her very specific and constructive comments that greatly helped us to improve the paper. The authors gratefully acknowledge the computing time granted by the Yunnan Observatories and provided on the facilities at the Yunnan Observatories Supercomputing Platform. We thank Professor X. Kong for helpful discussions about the classification of galaxies and the TP-AGB star issue in SPS models. This work is supported by the National Natural Science Foundation of China (NSFC, grant nos. 11773063, 11521303, 11733008), Chinese Academy of Sciences (CAS,

no. KJZD-EW-M06-01), and Yunnan Province (grant nos. 2017FB007, 2017HC018).

# Appendix The Sensitivity of Results to Assumptions of the Priors

The choice of priors is indispensable in any Bayesian data analysis. In principle, the priors should be chosen to best represent our state of knowledge before the analysis of the data. In the main body of this paper, we have assumed a uniform prior truncated at the allowed range for all free parameters of the SED model. This just reflects the fact that, except for the allowed range, the SED model itself tells us nothing about the detailed physical distribution of its free parameters. However, to make us notice the possible variation of the results with the assumed priors, we present here an analysis of the sensitivity of results to assumptions of the priors in the parameter space. As in Section 6.2, we demonstrate this with the analysis of the multiwavelength SEDs of the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938 by assuming the commonly used BC03 SSP model with a Chabrier (2003) IMF (bc03_ch), an Exp-dec SFH, and the Calzetti et al. (2000) DAL. In addition to the truncated uniform prior, we also considered three Gaussian priors with *b a* 2 *s* = - while centered at a, (*a b* + ) 2, and b, for all free parameters of the SED model. Here a and b represent the lower and upper limits of the parameters set by the used SED model. The Gaussian prior is not necessarily more physically reasonable than the uniform prior. However, it provides a way to test the sensitivity of the results to the assumptions of priors.

In Figure 36, we present the 1D posterior PDFs of free parameters assuming four different types of priors for the PEG ULTRAVISTA114558 and the SFG ULTRAVISTA99938. As shown in the figure, different types of priors can lead to somewhat different shapes of posterior PDFs. However, the results for different parameters have different sensitivities to the assumptions of priors. For both galaxies, the posterior PDFs of z and σsys are very insensitive to the assumptions of priors, while those of log(τ/yr) and log Z ( ) *Z* are very sensitive to the assumptions of priors. Meanwhile, the situation for other free parameters seems object dependent for the two galaxies. Although the shape of posterior PDFs assuming different priors could be obviously different, the parameter estimation with the median of the posterior PDFs is actually more similar, as shown in Tables 5 and 6. Generally, the sensitivity of a parameter to the assumptions of priors is consistent with the estimated uncertainty of the parameter. Some parameters, such as log(τ/yr) and log Z ( ) *Z* , cannot be well constrained by the data and are therefore more sensitive to the assumed priors. The derived parameters, such as stellar mass and luminosity, can be well constrained by the data and are therefore very insensitive to the assumptions of priors.

On the other hand, in Tables 5 and 6, we also presented the Bayesian evidences obtained with different assumptions of priors in the parameter space. As shown in the two tables, the Bayesian evidence of the model could be very sensitive to the assumptions of priors in the parameter space. In this paper (see Section 6), to make the results about model comparison with Bayesian evidence more robust, we have considered different assumptions about the SSP, SFH, and DAL that represent different priors in the model space, and our conclusions are obtained with the comparison of the different cases. Furthermore, we believe that the sensitivity of the Bayesian evidence of a model to the assumptions of priors in the parameter space is actually a benefit of the method. Since the physically more reasonable and informative priors of the parameters can be provided by a model for the distribution and evolution of the physical parameters of galaxies, it should be considered as a part of the model. In this way, the Bayesian model comparison/selection method developed in this paper has the potential to be used as a method for the comparison/selection of the combined model of the SED and the formation and evolution of galaxies. The results of Bayesian model comparison with the uniform priors for the physical parameters in this work could be used as a reference for this kind of investigation in the future.

# ORCID iDs

Yunkun Han https://orcid.org/0000-0002-2547-0434 Zhanwen Han https://orcid.org/0000-0001-9204-7778

# References

- Abbott, B. P., Abbott, R., Abbott, T. D., et al. 2016, PhRvL, 116, 061102
- Abbott, B. P., Abbott, R., Abbott, T. D., et al. 2017, PhRvL, 119, 161101
- Abrahamse, A., Knox, L., Schmidt, S., et al. 2011, ApJ, 734, 36
- Abramson, L. E., Gladders, M. D., Dressler, A., et al. 2016, ApJ, 832, 7
- Acquaviva, V., Gawiser, E., & Guaita, L. 2011, ApJ, 737, 47
- Adriani, O., Barbarino, G. C., Bazilevskaya, G. A., et al. 2009, Natur, 458, 607
- Ahmad, Q. R., Allen, R. C., Andersen, T. C., et al. 2002, PhRvL, 89, 011301
- Arnouts, S., Cristiani, S., Moscardini, L., et al. 1999, MNRAS, 310, 540
- Bartos, I., & Kowalski, M. 2017, Multimessenger Astronomy (Bristol: Institute of Physics Publishing)
- Becker, J. K. 2008, PhR, 458, 173
- Benítez, N. 2000, ApJ, 536, 571
- Bica, E. 1988, A&A, 195, 76
- Bica, E., & Alloin, D. 1986, A&A, 162, 21
- Bolzonella, M., Miralles, J.-M., & Pelló, R. 2000, A&A, 363, 476
- Bonoli, S., Marulli, F., Springel, V., et al. 2009, MNRAS, 396, 423
- Brammer, G. B., van Dokkum, P. G., & Coppi, P. 2008, ApJ, 686, 1503
- Brammer, G. B., van Dokkum, P. G., Franx, M., et al. 2012, ApJS, 200, 13
- Brammer, G. B., Whitaker, K. E., van Dokkum, P. G., et al. 2011, ApJ, 739, 24
- Bruzual, A. G. 2007, in IAU Symp. 241, Stellar Populations as Building Blocks of Galaxies, ed. A. Vazdekis & R. Peletier (Cambridge: Cambridge Univ. Press), 125
- Bruzual, A. G. 2011, RMxAC, 40, 36
- Bruzual, A. G., & Charlot, S. 1993, ApJ, 405, 538
- Bruzual, G., & Charlot, S. 2003, MNRAS, 344, 1000
- Buzzoni, A. 1989, ApJS, 71, 817
- Calistro Rivera, G., Lusso, E., Hennawi, J. F., & Hogg, D. W. 2016, ApJ, 833, 98
- Calzetti, D., Armus, L., Bohlin, R. C., et al. 2000, ApJ, 533, 682
- Capozzi, D., Maraston, C., Daddi, E., et al. 2016, MNRAS, 456, 790
- Carnall, A. C., McLure, R. J., Dunlop, J. S., & Davé, R. 2018, MNRAS, 480, 4379
- Cassisi, S., Castellani, M., & Castellani, V. 1997a, A&A, 317, 108
- Cassisi, S., Castellani, V., Ciarcelluti, P., Piotto, G., & Zoccali, M. 2000, MNRAS, 315, 679
- Cassisi, S., degl'Innocenti, S., & Salaris, M. 1997b, MNRAS, 290, 515
- Cerviño, M. 2013, NewAR, 57, 123
- Chabrier, G. 2003, PASP, 115, 763
- Charlot, S., & Bruzual, A. G. 1991, ApJ, 367, 126
- Charlot, S., & Fall, S. M. 2000, ApJ, 539, 718
- Charlot, S., & Longhetti, M. 2001, MNRAS, 323, 887
- Chevallard, J., Charlot, S., Wandelt, B., & Wild, V. 2013, MNRAS, 432, 2061 Cid Fernandes, R., Mateus, A., Sodré, L., Stasińska, G., & Gomes, J. M. 2005, MNRAS, 358, 363
- Cid Fernandes, R., Sodré, L., Schmitt, H. R., & Leão, J. R. S. 2001, MNRAS, 325, 60
- Ciesla, L., Elbaz, D., & Fensch, J. 2017, A&A, 608, A41
- Conroy, C. 2013, ARA&A, 51, 393
- Conroy, C., & Gunn, J. E. 2010, ApJ, 712, 833
- Conroy, C., Gunn, J. E., & White, M. 2009, ApJ, 699, 486
- Conroy, C., & van Dokkum, P. G. 2012, ApJ, 760, 71
- Conroy, C., White, M., & Gunn, J. E. 2010, ApJ, 708, 58 Cullen, F., McLure, R. J., Khochfar, S., et al. 2018, MNRAS, 476, 3218
- Cullen, F., McLure, R. J., Khochfar, S., Dunlop, J. S., & Dalla Vecchia, C. 2017, MNRAS, 470, 3006
- da Silva, R. L., Fumagalli, M., & Krumholz, M. 2012, ApJ, 745, 145
- Dahlen, T., Mobasher, B., Faber, S. M., et al. 2013, ApJ, 775, 93
- Dale, D. A., Aniano, G., Engelbracht, C. W., et al. 2012, ApJ, 745, 95
- Davé, R. 2008, MNRAS, 385, 147
- de Barros, S., Schaerer, D., & Stark, D. P. 2014, A&A, 563, A81
- De Lucia, G., Kauffmann, G., & White, S. D. M. 2004, MNRAS, 349, 1101
- De Lucia, G., Muzzin, A., & Weinmann, S. 2014, NewAR, 62, 1
- Díaz-García, L. A., Cenarro, A. J., López-Sanjuan, C., et al. 2015, A&A, 582, A14
- Diemer, B., Sparre, M., Abramson, L. E., & Torrey, P. 2017, ApJ, 839, 26 Dolphin, A. E. 2012, ApJ, 751, 60
- Draine, B. 2010, Physics of the Interstellar and Intergalactic Medium (Princeton, NJ: Princeton Univ. Press)
- Draine, B. T. 2003, ARA&A, 41, 241
- Dries, M., Trager, S. C., & Koopmans, L. V. E. 2016, MNRAS, 463, 886
- Dries, M., Trager, S. C., Koopmans, L. V. E., Popping, G., & Somerville, R. S. 2018, MNRAS, 474, 3500
- Duchêne, G., & Kraus, A. 2013, ARA&A, 51, 269
- Dye, S. 2008, MNRAS, 389, 1293
- Eldridge, J. J., Izzard, R. G., & Tout, C. A. 2008, MNRAS, 384, 1109
- Eldridge, J. J., & Stanway, E. R. 2009, MNRAS, 400, 1019
- Eldridge, J. J., Stanway, E. R., Xiao, L., et al. 2017, PASA, 34, e058 Fabian, A. C. 2012, ARA&A, 50, 455
- Ferland, G., Korista, K., Verner, D., et al. 1998, PASP, 110, 761
- Ferland, G. J., Chatzikos, M., Guzmán, F., et al. 2017, RMxAA, 53, 385
- Ferland, G. J., Porter, R. L., van Hoof, P. A. M., et al. 2013, RMxAA, 49, 137 Feroz, F., & Hobson, M. P. 2008, MNRAS, 384, 449
- Feroz, F., Hobson, M. P., & Bridges, M. 2009, MNRAS, 398, 1601
- Feroz, F., Hobson, M. P., Cameron, E., & Pettitt, A. N. 2013, arXiv:1306.2144
- Ferrarese, L., & Merritt, D. 2000, ApJL, 539, L9
- Finlator, K., Davé, R., & Oppenheimer, B. D. 2007, MNRAS, 376, 1861
- Fioc, M., & Rocca-Volmerange, B. 1997, A&A, 326, 950
- Gladders, M. D., Oemler, A., Dressler, A., et al. 2013, ApJ, 770, 64
- Gomes, J. M., & Papaderos, P. 2017, A&A, 603, A63
- Gregory, P. 2005, Bayesian Logical Data Analysis for the Physical Sciences (New York: Cambridge Univ. Press)
- Grogin, N. A., Kocevski, D. D., Faber, S. M., et al. 2011, ApJS, 197, 35
- Groves, B. A., Dopita, M. A., & Sutherland, R. S. 2004, ApJS, 153, 9
- Hamann, F., & Ferland, G. 1993, ApJ, 418, 11
- Han, Y., & Han, Z. 2012, ApJ, 749, 123
- Han, Y., & Han, Z. 2014, ApJS, 215, 2
- Han, Z., Podsiadlowski, P., & Lynas-Gray, A. E. 2007, MNRAS, 380, 1098 Heavens, A. F., Jimenez, R., & Lahav, O. 2000, MNRAS, 317, 965
- Heber, U. 2009, ARA&A, 47, 211
- Heckman, T. M., & Best, P. N. 2014, ARA&A, 52, 589
- Henriques, B., Maraston, C., Monaco, P., et al. 2011, MNRAS, 415, 3571
- Hopkins, P. F., Cox, T. J., Kereš, D., & Hernquist, L. 2008a, ApJS, 175, 390
- Hopkins, P. F., Hernquist, L., Cox, T. J., & Kereš, D. 2008b, ApJS, 175, 356
- Hurley, J. R., Pols, O. R., & Tout, C. A. 2000, MNRAS, 315, 543
- Hurley, J. R., Tout, C. A., & Pols, O. R. 2002, MNRAS, 329, 897
- Ilbert, O., Arnouts, S., McCracken, H. J., et al. 2006, A&A, 457, 841
- Ilbert, O., Capak, P., Salvato, M., et al. 2009, ApJ, 690, 1236
- Iyer, K., & Gawiser, E. 2017, ApJ, 838, 127
- Jeffreys, H. 1961, The Theory of Probability (3rd ed.; Oxford: Oxford Univ. Press)
- Jenkins, C. 2014, in AIP Conf. Ser. 1636, Bayesian Inference and Maximum Entropy Methods in Science and Engineering (Melville, NY: AIP), 106
- Johnson, S. P., Wilson, G. W., Tang, Y., & Scott, K. S. 2013, MNRAS, 436, 2535
- Kauffmann, G., Heckman, T. M., White, S. D. M., et al. 2003, MNRAS, 341, 33
- Kennicutt, R. C., & Evans, N. J. 2012, ARA&A, 50, 531
- Kennicutt, R. C., Jr. 1998, ARA&A, 36, 189
- Kobayashi, M. A. R., Inoue, Y., & Inoue, A. K. 2013, ApJ, 763, 3
- Koekemoer, A. M., Faber, S. M., Ferguson, H. C., et al. 2011, ApJS, 197, 36
- Koleva, M., Prugniel, P., Bouchard, A., & Wu, Y. 2009, A&A, 501, 1269
- Kong, X., Charlot, S., Weiss, A., & Cheng, F. Z. 2003, A&A, 403, 877
- Kormendy, J., & Ho, L. C. 2013, ARA&A, 51, 511
- Kotulla, R., Fritze, U., Weilbacher, P., & Anders, P. 2009, MNRAS, 396, 462
- Kriek, M., & Conroy, C. 2013, ApJL, 775, L16
- Kriek, M., Labbé, I., Conroy, C., et al. 2010, ApJL, 722, L64
- Kriek, M., van Dokkum, P. G., Labbé, I., et al. 2009, ApJ, 700, 221
- Kroupa, P. 2001, MNRAS, 322, 231
- Kroupa, P., Tout, C. A., & Gilmore, G. 1993, MNRAS, 262, 545
- Lee, S.-K., Ferguson, H. C., Somerville, R. S., et al. 2014, ApJ, 783, 81 Lee, S.-K., Ferguson, H. C., Somerville, R. S., Wiklind, T., & Giavalisco, M.
- 2010, ApJ, 725, 1644
- Lee, S.-K., Idzi, R., Ferguson, H. C., et al. 2009, ApJS, 184, 100
- Leitherer, C., Schaerer, D., Goldader, J. D., et al. 1999, ApJS, 123, 3
- Leja, J., Johnson, B. D., Conroy, C., van Dokkum, P. G., & Byler, N. 2017, ApJ, 837, 170
- Li, A., & Greenberg, J. M. 1997, A&A, 323, 566
- Longhetti, M., & Saracco, P. 2009, MNRAS, 394, 774
- MacKay, D. 2003, Information Theory, Inference and Learning Algorithms (Cambridge: Cambridge Univ. Press)
- MacKay, D. J. C. 1992, Neural Computation, 4, 415
- Magris, C. G., Bruzual, A. G., & Mateu, J. 2011, RMxAA, 27, 66
- Magris, C. G., Mateu, P. J., Mateu, C., et al. 2015, PASP, 127, 16
- Maraston, C. 1998, MNRAS, 300, 872
- Maraston, C. 2005, MNRAS, 362, 799
- Maraston, C., Daddi, E., Renzini, A., et al. 2006, ApJ, 652, 85
- Maraston, C., Pforr, J., Renzini, A., et al. 2010, MNRAS, 407, 830
- Maraston, C., & Strömbäck, G. 2011, MNRAS, 418, 2785
- Marigo, P., Bressan, A., Nanni, A., Girardi, L., & Pumo, M. L. 2013, MNRAS, 434, 488
- Marigo, P., & Girardi, L. 2007, A&A, 469, 239
- Marulli, F., Bonoli, S., Branchini, E., Moscardini, L., & Springel, V. 2008, MNRAS, 385, 1846
- Massarotti, M., Iovino, A., & Buzzoni, A. 2001, A&A, 368, 74
- McCracken, H. J., Milvang-Jensen, B., Dunlop, J., et al. 2012, A&A, 544, A156
- McKee, C. F., & Ostriker, E. C. 2007, ARA&A, 45, 565
- McKee, C. F., & Ostriker, J. P. 1977, ApJ, 218, 148
- Melia, F., & Falcke, H. 2001, ARA&A, 39, 309
- Merloni, A. 2004, MNRAS, 353, 1035
- Michałowski, M. J., Hayward, C. C., Dunlop, J. S., et al. 2014, A&A, 571, A75
- Miller, G. E., & Scalo, J. M. 1979, ApJS, 41, 513
- Mo, H., van den Bosch, F., & White, S. 2010, Galaxy Formation and Evolution (The Edinburgh Building (Cambridge: Cambridge Univ. Press)
- Murase, K., Ioka, K., Nagataki, S., & Nakamura, T. 2008, PhRvD, 78, 023005
- Muzzin, A., Marchesini, D., Stefanon, M., et al. 2013a, ApJ, 777, 18
- Muzzin, A., Marchesini, D., Stefanon, M., et al. 2013b, ApJS, 206, 8
- Naab, T., & Ostriker, J. P. 2017, ARA&A, 55, 59
- Noll, S., Burgarella, D., Giovannoli, E., et al. 2009, A&A, 507, 1793
- Nomoto, K., Kobayashi, C., & Tominaga, N. 2013, ARA&A, 51, 457
- Ocvirk, P., Pichon, C., Lançon, A., & Thiébaut, E. 2006, MNRAS, 365, 46
- Pacifici, C., Charlot, S., Blaizot, J., & Brinchmann, J. 2012, MNRAS, 421, 2002
- Pacifici, C., da Cunha, E., Charlot, S., et al. 2015, MNRAS, 447, 786
- Pforr, J., Maraston, C., & Tonini, C. 2012, MNRAS, 422, 3285
- Pirzkal, N., Rothberg, B., Nilsson, K. K., et al. 2012, ApJ, 748, 122
- Pols, O. R., Schröder, K.-P., Hurley, J. R., Tout, C. A., & Eggleton, P. P. 1998, MNRAS, 298, 525
- Reddy, N. A., Kriek, M., Shapley, A. E., et al. 2015, ApJ, 806, 259
- Reddy, N. A., Pettini, M., Steidel, C. C., et al. 2012, ApJ, 754, 25
- Renzini, A., & Buzzoni, A. 1986, in Spectral Evolution of Galaxies, ed. C. Chiosi & A. Renzini (Dordrecht: Reidel), 195
- Rosenfield, P., Marigo, P., Girardi, L., et al. 2014, ApJ, 790, 22
- Rosenfield, P., Marigo, P., Girardi, L., et al. 2016, ApJ, 822, 73
- Salim, S., Boquien, M., & Lee, J. C. 2018, ApJ, 859, 11
- Salim, S., Rich, R. M., Charlot, S., et al. 2007, ApJS, 173, 267
- Salmon, B., Papovich, C., Long, J., et al. 2016, ApJ, 827, 20
- Salpeter, E. E. 1955, ApJ, 121, 161
- Sana, H., de Mink, S. E., de Koter, A., et al. 2012, Sci, 337, 444
- Sawicki, M. 2012, PASP, 124, 1208
- Scannapieco, C., Tissera, P. B., White, S. D. M., & Springel, V. 2005, MNRAS, 364, 552
- Schaerer, D., & de Barros, S. 2009, A&A, 502, 423
- Schenker, M. A., Ellis, R. S., Konidaris, N. P., & Stark, D. P. 2013, ApJ, 777, 67
- Scoville, N., Aussel, H., Brusa, M., et al. 2007, ApJS, 172, 1
- Serra, P., Amblard, A., Temi, P., et al. 2011, ApJ, 740, 22
- Simha, V., Weinberg, D. H., Conroy, C., et al. 2014, arXiv:1404.0402
- Skelton, R. E., Whitaker, K. E., Momcheva, I. G., et al. 2014, ApJS, 214, 24
- Skrutskie, M. F., Cutri, R. M., Stiening, R., et al. 2006, AJ, 131, 1163
- Somerville, R. S., & Davé, R. 2015, ARA&A, 53, 51
- Spitzer, L. 1978, Physical Processes in the Interstellar Medium (Weinheim: Wiley)
- Stanway, E. R., & Eldridge, J. J. 2018, MNRAS, 479, 75
- Stark, D. P., Schenker, M. A., Ellis, R., et al. 2013, ApJ, 763, 129
- Sutherland, R. S., & Dopita, M. A. 1993, ApJS, 88, 253
- Suyu, S. H., Marshall, P. J., Hobson, M. P., & Blandford, R. D. 2006, MNRAS, 371, 983
- Timmes, F. X., Woosley, S. E., & Weaver, T. A. 1995, ApJS, 98, 617
- Tojeiro, R., Heavens, A. F., Jimenez, R., & Panter, B. 2007, MNRAS, 381, 1252
- Trotta, R. 2008, ConPh, 49, 71

428, 1479

36

- van Dokkum, P. G. 2008, ApJ, 674, 29
- van Dokkum, P. G., & Conroy, C. 2012, ApJ, 760, 70
- Vazdekis, A., Sánchez-Blázquez, P., Falcón-Barroso, J., et al. 2010, MNRAS, 404, 1639
- Walcher, J., Groves, B., Budavári, T., & Dale, D. 2011, Ap&SS, 331, 1
- Whitaker, K. E., Labbé, I., van Dokkum, P. G., et al. 2011, ApJ, 735, 86
- Williams, R. J., Quadri, R. F., Franx, M., van Dokkum, P., & Labbé, I. 2009, ApJ, 691, 1879

Zhang, F., Li, L., Han, Z., Zhuang, Y., & Kang, X. 2013, MNRAS, 428, 3390

Zibetti, S., Gallazzi, A., Charlot, S., Pierini, D., & Pasquali, A. 2013, MNRAS,

- Witt, A. N., & Gordon, K. D. 2000, ApJ, 528, 799
- Wuyts, S., Förster Schreiber, N. M., Lutz, D., et al. 2011, ApJ, 738, 106 York, D. G., Adelman, J., Anderson, J. E., Jr., et al. 2000, AJ, 120, 1579

Zackrisson, E., Bergvall, N., & Leitet, E. 2008, ApJL, 676, L9 Zhang, F., Han, Z., Li, L., & Hurley, J. R. 2004, A&A, 415, 117 Zhang, F., Han, Z., Li, L., & Hurley, J. R. 2005a, MNRAS, 357, 1088

Zhang, H.-X., Puzia, T. H., & Weisz, D. R. 2017, ApJS, 233, 13

Zoccali, M., Cassisi, S., Frogel, J. A., et al. 2000, ApJ, 530, 418

Zhang, F., Li, L., & Han, Z. 2005b, MNRAS, 364, 503

