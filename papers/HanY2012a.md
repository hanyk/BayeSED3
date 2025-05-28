# **Evolution of the luminosity function and obscuration of active galactic nuclei: comparison between X-ray and infrared**

Yunkun Han,1,2,3- Benzhong Dai,4 Bo Wang,1,3 Fenghui Zhang1,3 and Zhanwen Han1,3

1*National Astronomical Observatories/Yunnan Observatory, the Chinese Academy of Sciences, Kunming 650011, China*

2*Graduate University of Chinese Academy of Sciences, Beijing 100049, China*

3*Key Laboratory for the Structure and Evolution of Celestial Objects, Chinese Academy of Sciences, Kunming 650011, China*

4*Department of Physics, Yunnan University, Kunming 650091, China*

Accepted 2012 March 8. Received 2012 March 6; in original form 2010 September 6

#### **ABSTRACT**

We present a detailed comparison between the 2–10 keV hard X-ray and infrared (IR) luminosity functions (LFs) of active galactic nuclei (AGNs). The composite X-ray to IR spectral energy distributions (SEDs) of AGNs, which are used to compare the hard X-ray LF (HXLF) and the IRLF, are modelled with a simple, but well-tested torus model, based on the radiative transfer and photoionization code, CLOUDY. Four observational determinations of the evolution of the 2–10 keV HXLF and six evolution models of the obscured type 2 AGN fraction (*f* 2) are considered. The 8.0- and 15-µm LFs for unobscured type 1, obscured type 2 and all AGNs are predicted from the HXLFs, and then compared with the measurements currently available. We find that the IRLFs predicted from the HXLFs tend to underestimate the number of the most IR-luminous AGNs. This is independent of the choices of HXLF and *f* 2, and this is even more obvious for the HXLFs recently measured. We show that the discrepancy between the HXLFs and IRLFs can be largely resolved when the anticorrelation between the ultraviolet (UV) to X-ray slope αox and UV luminosity *L*UV is appropriately considered. We also discuss other possible explanations for the discrepancy, such as the missing population of Compton-thick AGNs and the possible contribution of star formation in the host to the mid-IR. Meanwhile, we find that the HXLFs and IRLFs of AGNs might be more consistent with each other if the obscuration mechanisms of quasars and Seyferts are assumed to be different, corresponding to their different triggering and fuelling mechanisms. In order to clarify these interesting issues, it would be very helpful to obtain more accurate measurements of the IRLFs of AGNs, especially those determined at smaller redshift bins, and to more accurately separate the measurements for type 1 and type 2 AGNs.

**Key words:** galaxies: active – galaxies: evolution – galaxies: formation – galaxies: luminosity function, mass function – infrared: galaxies – X-rays: galaxies.

## **1 INTRODUCTION**

Active galactic nuclei (AGNs), which are the compact regions at the centre of active galaxies, release a great deal of energy in the form of radiation over the electromagnetic spectrum from radio, infrared (IR), optical, ultraviolet (UV) and X-ray to γ -ray. They are now believed to be powered by the accretion of mass into supermassive black holes (SMBHs). In the local Universe, SMBHs are found to exist at the centre of the most massive galaxies. There is a good correlation between the mass of SMBHs and the properties of host galaxies (Hopkins et al. 2007b; Gultekin et al. 2009; ¨ Kormendy & Bender 2009; Merloni et al. 2010), such as velocity dispersion (Ferrarese & Merritt 2000; Gebhardt et al. 2000; Tremaine et al. 2002), mass (Magorrian et al. 1998; Graham 2012; Haring & Rix 2004) and luminosity (Kormendy & Richstone 1995; ¨ Marconi & Hunt 2003) of the host bulge. However, the AGN activity and star formation are found to peak at a similar redshift and to decline towards low redshift simultaneously (Hopkins 2004; Silverman et al. 2008; Aird et al. 2010). Meanwhile, the mass density of local SMBHs in the galaxy centre is found to be consistent with that accreted by AGNs throughout the history of the Universe (Yu & Tremaine 2002; Marconi et al. 2004; Merloni 2004). These correlations strongly support the idea that the growth of <sup>-</sup>E-mail: hanyk@ynao.ac.cn

SMBHs should be coupled with the formation and evolution of galaxies (Di Matteo, Springel & Hernquist 2005; Hopkins et al. 2005, 2006, 2008; Bower et al. 2006; Croton et al. 2006; Di Matteo et al. 2008), although some authors (e.g. Peng 2007; Jahnke & Maccio 2011) have argued that they could have a non-causal origin. `

While the important role that SMBHs, and thus AGNs, play in the formation and evolution of galaxies has been well established, the detailed mechanisms about this process are still largely unknown. The luminosity functions (LFs) of AGNs, which describe the spacial density of AGNs as a function of luminosity and redshift, is an important observable quantity for understanding the distribution and evolution of AGNs. It constrains the accretion history of SMBHs, and it reveals the triggering and fuelling mechanism of AGNs and their coevolution with host galaxies. An observational determination of the bolometric LFs of AGNs requires multiwavelength observations spanning the whole wavelength range of electromagnetic spectrum and a sampling of the large comoving volume and luminosity range. So, in practice, the LFs of AGNs are measured independently from different wavelength bands, such as radio (e.g. Nagar, Falcke & Wilson 2005), IR (e.g. Babbedge et al. 2006; Brown et al. 2006; Matute et al. 2006), optical (e.g. Fan et al. 2001; Wolf et al. 2003; Croom et al. 2004; Richards et al. 2006; Bongiorno et al. 2007; Fontanot et al. 2007; Shankar & Mathur 2007), soft X-ray (e.g. Miyaji, Hasinger & Schmidt 2000, 2001; Hasinger, Miyaji & Schmidt 2005; Silverman et al. 2005a), hard X-ray (e.g. Ueda et al. 2003; La Franca et al. 2005; Silverman et al. 2005b, 2008; Ebrero et al. 2009; Yencho et al. 2009; Aird et al. 2010) and emission lines (e.g. Hao et al. 2005). However, because of the different selection effect suffered by different bands, the LFs of AGNs measured from different bands are not necessarily consistent with each other.

Among the various bands, the X-ray, especially the hard X-ray, band is the most efficient for selecting AGNs. Recently, the evolution of the hard X-ray LF (HXLF) of AGNs from *z* ∼ 0 to 5 has been found to be best described by a luminosity-dependent density evolution (LDDE) model. According to this model, the spatial density of AGNs with lower luminosity peaks at a lower redshift than those with high luminosity, and the faint-end slope of the LFs is flattened as the redshift increases (Ueda et al. 2003; Barger et al. 2005; Hasinger et al. 2005). This type of so-called 'cosmic downsizing' evolution trend of the AGN population has been further confirmed in the radio and optical bands (Cirasuolo, Magliocchetti & Celotti 2005; Bongiorno et al. 2007). These results have revealed a dramatically different evolutionary model for Seyfert galaxies and quasars, and they imply very different triggering, fuelling and accretion mechanisms for the two classes of AGNs.

Meanwhile, AGNs are classified into two major classes, according to their optical spectra. Type 1 AGNs exhibit both broad permitted lines and narrow forbidden lines in their spectra, while type 2 AGNs present only narrow lines (Khachikian & Weedman 1974). Rowan-Robinson (1977) was the first to put forward the idea that AGNs are surrounded by a dusty medium that absorbs their visible and UV light and then re-emits this in the mid-IR. The extinction resulting from this obscuring medium is responsible for the distinction between type 1 and type 2 AGNs. Latterly, this idea was developed into the so-called unified model of AGNs (Pier & Krolik 1992; Antonucci 1993; Maiolino & Rieke 1995; Krolik 1999; Zhang & Wang 2006; Wang & Zhang 2007). In the model, the differences between different types of AGNs can be explained by the anisotropically distributed obscuring medium (often visualized as a geometrically and optically thick torus comprised of dust and molecular gas) surrounding a basic black-hole–accretion-disc system, while different lines of sight into and through these obscuring mediums result in the diverse observational properties of the AGN population.

However, the obscuration of AGNs by an anisotropically distributed gas/dust medium implies great systematic selection bias for understanding the properties and evolution of AGNs. Moreover, obscuring mediums around AGNs have recently been found to be distributed in a much more complex manner than a simple compact torus (Risaliti, Elvis & Nicastro 2002; Kuraszkiewicz et al. 2003; Risaliti et al. 2005; Goulding & Alexander 2009), and they might evolve with luminosity (Steffen et al. 2003; Ueda et al. 2003; Hasinger 2004; Barger et al. 2005; Simpson 2005) and redshift (La Franca et al. 2005; Ballantyne, Everett & Murray 2006a; Treister & Urry 2006; Hasinger 2008; Ebrero et al. 2009). These results imply that the observational properties of AGNs might vary significantly from object to object. This complicates the understanding of the intrinsic characteristics of AGNs and their correlations with the host galaxy. However, current state-of-the-art synthesis models of the cosmic X-ray background (CXRB) (e.g. Gilli, Comastri & Hasinger 2007) show that a large population of heavily obscured Compton-thick AGNs (with *N*H ≤ 1024 cm−2) are required to fit the CXRB spectrum. This population of Compton-thick AGNs can be missed by even deep hard X-ray surveys, because they are deeply buried by the obscuring medium.

Furthermore, the recent results of Hasinger (2008) and Treister et al. (2010) show that the fraction of absorbed AGNs increases significantly with redshift to *z* ∼ 2–3, accompanied with the cosmic coevolution of star formation and AGN activity. These results support the idea that the obscuration of AGNs cannot simply come from an unevolving torus, which is employed by the traditional unified model of AGNs. The obscuration mechanism of AGNs with different triggering, fuelling and accretion mechanisms might be different and it could be associated with their coevolution with galaxies (Davies et al. 2006; Ballantyne et al. 2006a; Ballantyne 2008).

So, detailed studies of the obscuring medium around AGNs, such as its geometry, distribution, composition, origin and evolution, are very important (Zhang 2004; Wang, Zhang & Luo 2005; Liu & Zhang 2011). The obscured or absorbed optical, UV and X-ray radiation will be re-emitted in the IR. The IR bands represent an important complement for understanding the properties of the obscuring medium around AGNs and their coevolution with host galaxies. With existing IR space telescopes, such as *Spitzer*, *Herschel* and the forthcoming *James Webb Space Telescope* (*JWST*), our observation and understanding of AGNs from the IR band will be greatly improved. Given the limitations suffered by X-ray observations, it is important to study the LFs and obscuration of AGNs together and from combined views of X-ray and IR.

By using the spectral energy distributions (SEDs) modelled with a simple torus model, which is based on the radiative transfer and photoionization code CLOUDY, Ballantyne et al. (2006b) have been able to relate the X-ray and IR properties of AGNs and to explore the effects of the parameters of the obscuring medium. They have presented the mid-IR number counts and LFs for three evolution models of *f* 2 (equal to the covering factor under the unified model of AGNs), which are constrained by the synthesis model of the CXRB. The mid-IR number counts and LFs predicted from the HXLF are in good agreement with direct IR observations, especially when assuming an inner radius *R*in of 10 pc for the obscuring medium, as expected if the obscuring material is connected to the galactic-scale phenomenon. The mid-IR LFs of AGNs are found to be a much better tool for determining the evolution of *f* 2 with *z*.

Ballantyne et al. (2006b) have presented the mid-IR LFs for all AGNs at different redshifts, but the observational mid-IR LFs of AGNs (i.e. Brown et al. 2006), used for comparison, are for type 1 AGNs only. Following the work of Ballantyne et al. (2006b), some important improvements to the measurement of the HXLF of AGNs have been presented (e.g. Silverman et al. 2008; Ebrero et al. 2009; Aird et al. 2010). Furthermore, the actual evolution model of AGN obscuration is not necessarily within the three models proposed by Ballantyne et al. (2006a); other possibilities need to be tested for more reasonable conclusions.

In this paper, we present a more detailed comparison between the HXLFs and mid-IR LFs of AGNs, which are connected by the composite X-ray to IR SEDs that are modelled by a modified version of the simple, but well-tested torus model of Ballantyne et al. (2006b). More observational determinations of the 2–10 keV HXLF and the evolution models of *f* 2 have been considered. The 8.0- and 15-µm LFs for unobscured type 1, obscured type 2 and all AGNs are predicted from different combinations of HXLF and *f* 2, and these are then compared with current IR observational results. Besides the measurement of Brown et al. (2006), the 15-µm LF given by Matute et al. (2006) and the recent results of Fu et al. (2010) have also been added for comparison.

We begin in Section 2 by reviewing the current understanding of AGN evolution from the X-ray band. We include the current observational determination of the evolution of the HXLF of AGNs in Section 2.1, and the evolution of AGN obscuration in Section 2.2. In Section 3, we present the detailed procedures for modelling the composite X-ray to IR SED of AGNs, and our modifications to the original torus model of Ballantyne et al. (2006b). In Section 4, we present the method used to compute the IRLFs of type 1, type 2 and all AGNs from different combinations of HXLF and *f* 2. In Section 5, we present our results and we compare these with measurements from direct mid-IR observations in order to come to conclusions about the evolution of LFs and the obscuration of AGNs from combined views of hard X-ray and mid-IR. We find that the mid-IR LFs predicted from HXLFs tend to underestimate the number of the most IR-luminous AGNs, which is independent of the choices of HXLF and *f* 2, and this is even more obvious for the HXLFs recently determined. In Section 6, we discuss explanations for this. Finally, we present a summary of this paper in Section 7.

Throughout this paper, we adopt a *H*0 = 70 km s−1 Mpc−1, = 0.7 andm =0.3 (Spergel et al. 2003) cosmology. Minor differences in the cosmology have negligible effects on our conclusions.

## **2 EVOLUTION OF AGNS REVEALED FROM X -RAY BANDS**

#### **2.1 Evolution of the HXLF of AGNs**

Strong X-ray emission is a unique indication of AGN activity at the centre of galaxies. Deep X-ray surveys by *Chandra* and *XMM– Newton*, which have already resolved most of the 2–10 keV CXRB into individual sources, have found that most sources of the CXRB are AGNs. The X-ray, especially the hard X-ray, band with energy ≥2 keV is highly efficient for selecting AGNs. Both moderately obscured (*N*H ≤ 1023 cm−2) and low-luminosity sources, which are commonly missed by optical observations, can be selected from the hard X-ray band. So, the evolution trends of AGNs revealed from hard X-ray observations might be much more reliable.

As mentioned above, the HXLF of AGNs is found to be best described by a LDDE model. However, the exact form of the evolution is still under debate, especially at high redshifts. In this paper, we adopt the LDDE model given by Ueda et al. (2003), where the present-day HXLF is described as a smoothly connected double power-law form, as follows:

$$\frac{\mathrm{d}\Phi(L_{X},z=0)}{\mathrm{d}\log L_{X}}=A[(L_{X}/L_{*})^{\gamma^{1}}+(L_{X}/L_{*})^{\gamma^{2}}]^{-1}.\tag{1}$$

Here, γ 1 is the faint-end slope, γ 2 is the bright-end slope, *L*∗ is the characteristic break luminosity and *A* is a normalization factor. The evolution of LFs is given by

$$\frac{\mathrm{d}\Phi(L_{\mathrm{X}},z)}{\mathrm{d}\log L_{\mathrm{X}}}=\frac{\mathrm{d}\Phi(L_{\mathrm{X}},0)}{\mathrm{d}\log L_{\mathrm{X}}}e(z,L_{\mathrm{X}}),\tag{2}$$

where the evolution term is given by

$$e(z,L_{\rm X})=\left\{\begin{array}{ll}(1+z)^{p1}&[z<z_{\rm c}(L_{\rm X})]\\ e(z_{\rm c})\left[\frac{1+z}{1+z_{\rm c}(L_{\rm X})}\right]^{p2}&[z\geq z_{\rm c}(L_{\rm X})].\end{array}\right.\tag{3}$$

The cut-off redshift*z*c, with a dependence on the luminosity starting from a characteristic luminosity *L*a, is given by a power law of *L*X:

$$z_{\rm c}(L_{\rm X})=\left\{\begin{array}{ll}z_{\rm c}^{*}&(L_{\rm X}\geq L_{\rm a})\\ z_{\rm c}^{*}(L_{\rm X}/L_{\rm a})^{\alpha}&(L_{\rm X}<L_{\rm a}),\end{array}\right.\tag{4}$$

where α measures the strength of the dependence of *z*c with luminosity.

Recently, Ebrero et al. (2009) re-measured the HXLF of AGNs using the *XMM–Newton* Medium Survey (XMS; Barcons et al. 2007) and other highly complete deeper and shallower surveys in order to assemble an overall sample of ∼450 identified AGNs in the 2–10 keV band. This is one of the largest and most complete samples to date. Aird et al. (2010) have presented a new observational determination of the evolution of the 2–10 keV HXLF of AGNs using data from many surveys, including the 2 Ms *Chandra* Deep Field survey and the AEGIS-X 200 ks survey. Combined with a sophisticated Bayesian methodology, these surveys allow a more accurate measurement of the evolution of the faint end of the HXLF. These authors found that the evolution of the HXLF is best described by a so-called luminosity and density evolution (LADE) model, rather than by the LDDE model. The LADE model is a modified pure luminosity evolution (PLE) model. According to the PLE model (Ueda et al. 2003), the evolution of the HXLF with redshift is described by allowing the characteristic break luminosity *L*∗ in the present-day HXLF (as given by equation 1) to evolve as

$$\log L_{*}(z)=\log L_{0}-\log\left[\left(\frac{1+z_{\rm c}}{1+z}\right)^{p_{1}}+\left(\frac{1+z_{\rm c}}{1+z}\right)^{p_{2}}\right].\tag{5}$$

Here, the parameter *z*c controls the transition from the strong low-*z* evolution to the high-*z* form. The LADE model is constructed by additionally allowing for overall decreasing density evolution with redshift, that is, by allowing the normalization constant *A* in the present-day HXLF (as given by equation 1) to evolve as

$\log A(z)=\log A_{0}+d(1+z)$.  
  

Here, *d* is an additional parameter describing the overall density decrease.

Fig. 1 shows the HXLF of AGNs at *z* = 0.5, 1.5, 2.0 and 2.5, as given by the LDDE model of Ueda et al. (2003), Ebrero et al. (2009) and Aird et al. (2010) and by the LADE model of Aird et al. (2010). The LADE modelling of the HXLF retains the same shape at all redshifts, but it undergoes strong luminosity evolution out to *z* ∼ 1. Meanwhile, the HXLF undergoes overall negative density evolution with increasing redshift. Fig. 1 clearly shows that the HXLF of Aird et al. (2010) is different from the others at high luminosity.

Downloaded from

**Figure 1.** HXLF of AGNs at*z* = 0.5, 1.5, 2.0 and 2.5, as given by the LDDE models of Ueda et al. (2003) (red solid line, called U03LDDE), Ebrero et al. (2009) (green dashed line, called E09LDDE) and Aird et al. (2010) (blue short-dashed line, called A10LDDE) and the LADE model of Aird et al. (2010) (purple dotted line, called A10LADE), respectively.

Different HXLFs have very different implications for the AGN population, such as different lifetimes, duty cycles, fuelling, triggering and evolution. Further complementary views from other wavelength bands, such as the IR, are important for a full understanding of the evolution of the AGN population. In this paper, the four models of the HLXF mentioned above have been used to predict the evolution of the mid-IR LFs of type 1 and/or type 2 AGNs.

#### **2.2 Evolution of the obscuration of AGNs**

Because there is a good correspondence between AGNs with *N*H ≥ 1022 cm−2 and those optically identified as being of type 2 (Tozzi et al. 2006), type 2 AGNs are commonly defined as those with absorbing column densities *N*H ≥ 1022 cm−2 in the X-ray band. According to the unified model of AGNs, *f* 2 is approximately equal to the covering factor of the gas with *N*H ≥ 1022 cm−2 around the AGN.

Using a population synthesis model of the CXRB, Ballantyne et al. (2006a) constrained the evolution of *f* 2 as a function of both *z* and *L*X. They presented three parametrizations for the evolution of *f* 2(log *L*X, *z*) that could fit the observed shape of the CXRB and X-ray number counts of AGNs. The first (shown in Fig. 2 and called 'f2_1'), with a moderate redshift evolution, is given as

$f_{2}=K_{1}(1+z)^{0.3}(\log L_{\rm X})^{-4.8}$.  
  

Here, *K*1 is a constant defined by *f* 2(log *L*X = 41.5, *z* = 0) = 0.8, which is based on observations in the local Universe. The second (shown in Fig. 2 and called 'f2_2'), with a more rapid redshift evolution, is given as

$f_{2}=K_{2}(1+z)^{0.9}(\log L_{\rm X})^{-1.3}$, (8)

where *K*2 is based on the Sloan Digital Sky Survey (SDSS) measurement of *f* 2(log *L*X = 41.5, *z* = 0) = 0.5 by Hao et al. (2005). In the above two cases, the *z* evolution is halted at *z* = 1, because there is no constraint on *f* 2 at higher redshifts. The last parametrization (shown in Fig. 2 and called 'f2_3'), which is considered as a nullhypothesis, assumes that *f* 2 does not evolve with redshift. This is given as

$$f_{2}=K_{3}\cos^{2}\left(\frac{\log L_{\rm X}-41.5}{9.7}\right),\tag{9}$$

where *K*3 is determined by *f* 2(log *L*X = 41.5, *z* = 0) = 0.8.

The fraction of type 2 AGNs can also be measured directly from observations in different bands. For example, in the optical, AGNs can be selected using their high-ionization lines to construct the standard diagnostic diagrams (Baldwin, Phillips & Terlevich 1981; Kewley, Dopita & Smith 2001; Kauffmann et al. 2003; Kewley et al. 2006). Furthermore, the ratio of narrow-line and broad-line AGNs can be measured as a function of *z* and the luminosity of emission lines (such as the [O III] 5007 Å line), which can be used as AGN power indicators. However, the significant limitations of the optical selection and classification of AGNs have been noticed (e.g. Moran, Filippenko & Chornock 2002; Netzer et al. 2006; Rigby et al. 2006). The nuclear emission can be obscured by the torus and/or outshone by the host-galaxy light. Alternatively, AGNs can be efficiently selected in the X-ray, and X-ray luminous AGNs can be classified as absorbed or unabsorbed, according to their absorbing column densities log *N*H < 22 or >22. The X-ray selection of AGNs suffers from the limited sensitivity of telescopes, which are only sensitive at ≤10 keV. So, a significant fraction of absorbed objects, especially the large number of Compton-thick AGNs with log *N*H > 24 predicted by the population synthesis model of the CXRB, might be missed by the current hard X-ray selection of AGNs.

The combination of X-ray and optical criteria is a much more robust method for the selection and classification of AGNs. Recently, Hasinger (2008) have presented a new determination of the fraction of absorbed sources as a function of X-ray luminosity and redshift from a sample of 1290 AGNs. These are selected in the 2–10 keV band from different flux-limited surveys with very high

**Figure 2.** The six evolution models of *f* 2(log *L*X, *z*) used in this paper are shown at log *L*X = 41.5 (red solid line), 43.0 (green dashed line), 44.5 (blue short-dashed line) and 46.0 (purple dotted line). The first three, which are constrained by the CXRB spectrum and the X-ray number count as given by Ballantyne et al. (2006a), are called 'f2_1', 'f2_2' and 'f2_3'. The last three, which are constructed according to the recent measurement of Hasinger (2008), are called 'f2_4', 'f2_5' and 'f2_6'. (See the text for more detailed explanations of these evolution models.)

-C 2012 The Authors, MNRAS **423,** 464–477 Monthly Notices of the Royal Astronomical Society -C 2012 RAS

optical identification completeness, and they are grouped into type 1 and type 2 according to their optical spectroscopic classification and X-ray absorption properties. So, the evolution of AGN absorption with luminosity and redshift is determined with higher statistical accuracy and smaller systematic errors than for previous results. The absorbed fraction is found to decrease strongly with X-ray luminosity, and it can be represented by an almost linear decrease with a slope of 0.281 ± 0.016. Meanwhile, it increases significantly with redshift as ∼(1 + *z*) 0.62 ±0.11 from *z* = 0 to *z* ∼ 2. However, the evolution of the absorbed AGN fraction over the whole redshift from *z* = 0 to *z* ∼ 5 can also be described as ∼(1 + *z*) 0.48 ±0.08, or ∼(1 + *z*) 0.38 ±0.09 when data with crude redshifts are excluded.

These findings could have important consequences for the broader context of AGN and galaxy coevolution. According to the results of Hasinger (2008), we have constructed three new evolution models of AGN obscuration, which are expressed as

$$f_{2}=-0.281(\log L_{\rm X}-43.75)+0.279(1+z)^{0.62},\tag{10}$$

$$f_{2}=-0.281(\log L_{\rm X}-43.75)+0.308(1+z)^{0.48}\tag{11}$$

and

$$f_{2}=-0.281(\log L_{\rm X}-43.75)+0.309(1+z)^{0.38}.\tag{12}$$

These are called 'f2_4', 'f2_5' and 'f2_6', respectively (as shown in Fig. 2). Because of the simple linear dependence on luminosity, the type 2 AGN fraction will quickly become zero as luminosity increases. According to the recent results of Brusa et al. (2010), the fraction of the obscured AGN population at the highest (*L*X > 1044 erg s−1) X-ray luminosity is ∼15–30 per cent. So, we have set a lower limit of 0.15 for the evolution of *f* 2(log *L*X, *z*) to denote a flattening of the decline at the highest luminosities, as expected. Naturally, an upper limit of 1 is forced in all cases.

Except for *f* 2(log *L*X,*z*), additional assumptions are needed to determine a specific distribution of *N*H. The exact distribution of *N*H is unknown except for local bright Seyfert 2 galaxies (e.g. Risaliti, Maiolino & Salvati 1999), but the covering factor is a useful parameter for its theoretical description. Here, we use a simple assumption about the distribution of *N*H, following Ballantyne et al. (2006b). In the 'simple *N*H distribution', 10 values of *N*H are considered: log (*N*H/cm−2) = 20, 20.5, ... , 24.0, 24.5, and a type 1 AGN is assumed to have an equal probability *p* of being absorbed by columns with log *N*H < 22. Likewise, a type 2 AGN has an equal chance of being absorbed by columns with log *N*H ≥ 22:

$$\log(N_{\mathrm{H}}/\mathrm{cm}^{-2})=\left\{\begin{array}{l}{{20.0,\ldots,21.5,\ p=\frac{1-f_{2}(\log L_{\mathrm{X}},z)}{4.0}}}\\ {{22.0,\ldots,24.5,\ p=\frac{f_{2}(\log L_{\mathrm{X}},z)}{6.0}}}\end{array}\right..$$

(13)

Because *f* 2(log *L*X, *z*) depends on log *L*X and *z*, the distribution of the obscuring medium around AGNs evolves with both log *L*X and *z*. It is worth noting that this simple assumption about the distribution of the obscuring medium with different values of *N*H is only used to construct *N*H-averaged SEDs (as described in Section 3). We do not expect it to give a correct fraction of Compton-thick AGNs. In fact, there are AGNs with estimated log *N*H ≥ 25 (e.g. NGC 1068; Matt et al. 1997). The inclusion of a very Compton-thick obscuring medium dramatically increases the computation time of SEDs. However, these have important effects, mainly in the far-IR, but they have only ignorable effects in the mid-IR, which is what we are mostly interested in currently.

# **3 MODELLING THE SPECTRAL ENERGY DISTRIBUTION OF AGNS**

In order to predict the IR properties of AGNs, we must know the relation between the IR and X-ray luminosity of AGNs. This can be given by the SEDs of AGNs with different X-ray luminosities. The SEDs of AGNs can be obtained from observations or theoretical calculations. The observational SEDs have the advantage of being based on observations of real AGNs. However, the number of observed objects is limited, and they only cover a narrow range of luminosity, redshift and wavelength. Alternatively, we can use theoretical dusty torus emission models, which include a detailed radiative transfer calculation, to compute the expected IR SED for a given X-ray luminosity. However, most radiative transfer calculations for the IR dust emission of AGNs do not include detailed considerations of gas and its interaction with dust (e.g. Treister et al. 2004, 2006; Nenkova et al. 2008a,b). The gas and dust are expected to be interacting with each other, and gas is responsible for the absorption of X-rays. So, for a reasonable comparison of the X-ray and IR properties of AGNs, gas and its interaction with dust must be considered.

Here, the calculation of the SEDs of AGNs is performed by using the photoionization code CLOUDY v. 07.02.01 (Ferland et al. 1998), following a procedure similar to that of Ballantyne et al. (2006b), but with some simplifications. In CLOUDY, atomic gas physics and detailed dust radiation physics, such as polycyclic aromatic hydrocarbon (PAH) emission and emission from very small grains, have been self-consistently considered. In addition, many important physical properties of the obscuring medium around AGNs, such as its distance from the central engine, gas density, distribution and gas/dust ratio, can be varied freely, and so explored extensively. CLOUDY is a one-dimensional radiative transfer code, and the methods we have employed to model the SEDs of AGNs are less sophisticated than those used by Treister et al. (2006) and Nenkova et al. (2008a). However, Ballantyne et al. (2006b) have shown that the SEDs, when averaged over a *N*H distribution, have very similar properties to the ensemble of AGNs found in the deep surveys of *Chandra*, *XMM–Newton* and *Spitzer*.

## **3.1 Construction of the CLOUDY model**

To construct a CLOUDY model, three ingredients must be specified. First, the shape and intensity of the radiation source, which define the incident continuum, must be set. The intrinsic spectrum of AGNs is described by a multicomponent continuum typical of AGNs, which extends from 100 keV to >1000 µm. Specifically, the 'big bump' component, peaking at ∼1 Ryd, is a rising power law with a high-energy exponential cut-off and it is parametrized by the temperature of the bump. The big blue bump temperature is set to a typical value of 105 K. The X-ray to UV ratio αox, which is defined by

$$\alpha_{\rm ox}=\frac{\log(L_{2\,{\rm keV}}/L_{2500\,{\rm\AA}})}{\log(v_{2\,{\rm keV}}/v_{2500\,{\rm\AA}})}=0.3838\,\log\left(\frac{L_{2\,{\rm keV}}}{L_{2500\,{\rm\AA}}}\right),\tag{14}$$

has an important effect on the resulting X-ray to IR ratio. Especially, there is evidence (e.g. Steffen et al. 2006; Hopkins, Richards & Hernquist 2007a; Vagnetti et al. 2010) that this parameter can be anticorrelated with the UV luminosity of AGNs. To explore the effects of this important parameter, we set αox to constant values of −1.5, −1.4 and −1.3, respectively. We have also tested the αox– *L*UV relation presented in Hopkins et al. (2007a), which is given by

$$\alpha_{\rm ox}=-0.107\,\log\left(\frac{L_{2500\,\AA}}{\rm erg\,s^{-1}\,Hz^{-1}}\right)+1.739,\tag{15}$$

and is determined specifically for unobscured (type 1) quasars. This results in a luminosity-dependent shape of the input SED from the AGN centre. The low-energy slope of the big bump continuum αuv is set to be the default value of −0.5. The X-ray photon index is assumed to be = 1.9, and so the energy index αx = 1 − = −0.9. The full continuum is the sum of two components, as given by

$$f_{v}=v^{\alpha_{\rm w}}\exp(-hv/KT_{\rm bb})\exp(-KT_{\rm IR}/hv)+av^{\alpha_{\rm x}},\tag{16}$$

where *T*bb is the temperature of the big bump and the coefficient *a* is adjusted to produce the correct αox for the case where the big bump does not contribute to the emission at 2 keV. The big bump component is assumed to have an IR exponential cut-off at *KT*IR = 0.01 Ryd (1 Ryd ∼ 13.6 eV).1 Finally, this spectrum is scaled to have a luminosity of L2500 A˚ (erg s−1 Hz−1).

The second ingredient of a CLOUDY model is the chemical composition of the obscuring medium. A gaseous element abundance similar to that of the Orion nebula is assumed. The size distributions and abundances of graphitic, silicate and PAHs grains are also set to be similar to those of the Orion nebula. The obscuring medium is assumed to distribute uniformly and to have a constant hydrogen density *n*H of 104 cm−3.

The last ingredient of a CLOUDY model is the geometry of the obscuring medium. Here, the obscuring medium is assumed to be *R*in pc away from the centre, with a column density of *N*H. However, to be consistent with the unified model, Ballantyne et al. (2006b) have set the covering factor of the obscuring medium to *f* 2 when *N*H ≥ 1022 cm−2 or 1 − *f* 2 otherwise, in the CLOUDY model. Because *f* 2 depends on both luminosity and redshift, the CLOUDY simulation needs to be carried out for each luminosity and redshift. This would result in a great number of CLOUDY models. However, in CLOUDY models, the covering factor only has second-order effects on the spectrum through changes in the transport of the diffuse emission. So, we just use the default geometric covering factor of unity (the shell fully covers the continuum source) but a radiative covering factor of zero (i.e. an open geometry) is assumed, and the reflected radiation can also be obtained. The effects of the covering factor on the diffuse and reflected emissions are considered after the CLOUDY simulation, as described in the Section 3.2.

## **3.2 CLOUDY model grids and construction of the SEDs of AGNs**

The CLOUDY models are built for L2500 A˚ (erg s−1 Hz−1) 2 from 27 to 34 (in steps of 0.25), and log (*N*H/cm−2) from 20.0 to 24 (in steps of 0.5). Following Ballantyne et al. (2006b), first we set *R*in to be 10 pc. However, we have found that the temperatures of grains will be much higher than their sublimation temperatures at the highluminosity end if a constant *R*in of 10 pc is assumed. In practice, to fix this problem, we have found a luminosity-dependent *R*in that is given by

$$R_{\rm in}=10*\left[\frac{vL_{v}(2500\,\hat{\rm A})}{10^{46}}\right]^{1/2}\quad\mbox{pc},\tag{17}$$

2 To explore the effects of *L*UV-dependent *R*in and αox on the resulting SEDs, we use L2500 A˚ instead of *L*X to define the luminosity of the input SED.

The CLOUDY models are also built by assuming luminositydependent *R*in for comparison. Finally, as mentioned above, the CLOUDY models are built for four choices of αox.

For each CLOUDY model, three types of SED are predicted: the attenuated incident continuum, diffuse continuum and reflected continuum. The SEDs with different *N*H correspond to observations from different directions. However, according to the unified model of AGNs, obscuring mediums with all values of column density *N*H simultaneously exist around AGNs. However, there is evidence that obscured and unobscured AGNs present more similar *L*MIR/*L*X ratios (e.g. Alonso-Herrero et al. 2001; Krabbe, Boker & Maiolino 2001; Lutz et al. 2004; Horst et al. 2006, 2008) than those predicted by traditional torus models, which assume a smooth distribution of for the dusty obscuring medium. Recent works (Nenkova et al. 2008a,b; Honig et al. 2010; H ¨ onig & Kishimoto 2010) have shown ¨ that the distribution of the dusty obscuring medium is clumpy rather than contiguous. So, when we observe an AGN from one direction, both diffuse and reflected emission from all *N*H can be observed, in addition to the attenuated incident emission through an obscuring medium with a column density *N*H for this direction. For this reason, we have made a modification to the original torus emission model of Ballantyne et al. (2006b). The SED of an AGN with column density *N*H is constructed by adding the diffuse and reflected emission averaged over all 10 models with different values of *N*H to the attenuated incident emission through a particular value of *N*H. The weights are given by the probability distribution of the column densities, which is a function of *f* 2(log *L*X, *z*) (or covering factor), as discussed in Section 2.2. The SEDs constructed in this way are called unified SEDs.

Finally, the unified SEDs undergo an average over *N*H again, to produce the *N*H-averaged SEDs that will be used later to predict the IRLFs of AGNs. Here, three types of *N*H-averaged SEDs are constructed. The type 1 SED is an average of the unified SED with 1020.0 cm−2 ≤ *N*H < 1022 cm−2. The type 2 SED is an average of the unified SED with 1022 cm−2 ≤ *N*H ≤ 1024.5 cm−2. The average SED is an average of the unified SED with 1020.0 cm−2 ≤ *N*H ≤ 1024.5 cm−2. As an example, Fig. 3 shows the rest-frame SEDs taken from the f2_1 evolutionary grid (equation 7) for a Seyfert-like AGN (*L*X = 1043.54 erg s−1, *z* = 0.7, *f* 2 = 0.7484) and a quasar-like AGN (*L*X = 1046.54 erg s−1, *z* = 1.4, *f* 2 = 0.5705).

## **3.3 Testing the model SEDs of AGNs**

The method we have used to model the SEDs of AGNs is similar to that of Ballantyne et al. (2006b). They have extensively tested this method against large samples of AGNs. Here, we present two additional tests that are more directly related to the goal of this paper (i.e. the prediction of IRLFs from the HXLF). For this goal, the most important thing is to have a correct X-ray to IR relation.

Recently, Gandhi et al. (2009) have found a strong mid-IR : X-ray [log λ*L*λ (12.3 µm)–log *L*2−10 keV] luminosity correlation for a sample of local Seyfert galaxies, the cores of which have been resolved in the mid-IR. The relation is given by

$$\log\lambda L_{\lambda}(12.3\,\mu{\rm m})=-4.37+1.106\log L_{2-10\,{\rm keV}}.\tag{18}$$

It is found to be valid in a wide range of luminosity and it might extend into the quasar regime. Mullaney et al. (2011) have converted this correlation to that between log *L*IR and log *L*2−10 keV, which is <sup>1</sup> See the CLOUDY document for more detailed explanations of the construction of this AGN spectrum.

**Figure 3.** Top: rest-frame SEDs for a Seyfert-like AGN (*L*X = 1043.5 erg s−1) at *z* = 0.7, obscured by a dusty medium with an inner radius of 10 pc, hydrogen density *n*H = 104 cm−3 and covering factor *f* 2 = 0.7484. The type 1 SED is shown in red, the type 2 SED is shown in green, while the average SED is shown in blue. Bottom: as for the top panel, but for a quasar-like AGN (*L*X = 1046.5 erg s−1) at *z* = 1.4, obscured by a dusty medium with a covering factor *f* 2 = 0.5705. These spectra are taken from the 'f2_1' evolutionary grid (equation 7).

given by

$$\log\frac{L_{\rm IR}}{10^{43}\,{\rm erg\,s^{-1}}}=0.53+1.11\,\log\frac{L_{2-10\,{\rm keV}}}{10^{43}\,{\rm erg\,s^{-1}}}.\tag{19}$$

In Fig. 4, the log *L*IR–log *L*2−10 keV relation, computed from our model SEDs of AGNs using different choices for αox and *R*in, are tested against the observational relation given by Mullaney et al. (2011). As shown in the figure, the result obtained for αox = −1.4 and *R*in = 10 pc (as in Ballantyne et al. 2006b) significantly deviates from the nearly linear relation given by Mullaney et al. (2011). This problem has also been noted by Draper & Ballantyne (2011). We have found that a more linear log *L*IR–log *L*2−10 keV relation can be obtained if *R*in increases with *L*UV (as given by equation 17). With the typical value of αox = −1.4, this leads to a result similar to that of Mullaney et al. (2011), especially for the high-luminosity range. For the low-luminosity range, a larger αox seems to be necessary. However, when the αox–*L*UV relation given by Hopkins et al. (2007a) (as given by equation 15) is assumed, a too steep relation is obtained. So, these results support the anticorrelation between αox and *L*UV found by other independent observations (e.g. Steffen et al. 2006; Vagnetti et al. 2010), but they imply a more flat relation.

**Figure 4.** Test of the log *L*IR–log *L*2−10 keV relation computed from our model SEDs of AGNs against the observational relation (black solid line) given by Mullaney et al. (2011). The results obtained by using different choices of αox and *R*in are presented. The *L*UV-dependent *R*in is given by equation (17), while the *L*UV-dependent αox is given by equation (15).

#### **4 CONNECT ING X -RAY AND IR**

The IR is less affected by selection effects resulting from obscuration , which are suffered by optical and X-ray, while the X-ray band is currently the most efficient for selecting AGNs up to high redshifts. Connecting X-ray and IR can provide a clearer view of the evolution of the AGN population. If the evolution of AGNs shown in the HXLF and *f* 2, which are revealed mainly from X-ray observations, is intrinsic, it should be shown in the directly observed IRLFs of AGNs.

#### **4.1 Mid-IR LFs of type 1 and/or type 2 AGNs**

Because the HXLF tells us how the number density of AGNs per increment of log *L*X changes with *z* and *L*X, the following expression can be used to relate the HXLF to IRLFs d/d(log ν*L*ν ):

$$\frac{\mathrm{d}\Phi}{\mathrm{d}(\log vL_{v})}=\frac{\mathrm{d}\Phi}{\mathrm{d}(\log L_{X})}\frac{\mathrm{d}(\log L_{X})}{\mathrm{d}(\log vL_{v})}.\tag{20}$$

Here, d/d(log *L*X) is the HXLF of AGNs, as given in Section 2.1, and *L*ν is the luminosity at a given wavelength.

In Section 3, we obtained the SEDs spanning from X-ray all the way to IR for AGNs with different luminosities and redshifts. So, the dependence of the IR luminosity on the hard X-ray luminosity, as described by d(log *L*X)/d(log ν*L*ν ), can easily be obtained from the SEDs. To predict the IRLFs for all AGNs (type 1 + type 2), we use the average SED presented in Section 3. The IRLF of AGNs is not an integrated quantity, and so it is much more sensitive to the evolution trends in the HXLF and *f* 2 than the cumulative number count distribution and background spectra intensity of AGNs. Using the IRLFs for all AGNs has the advantage that it is independent of the methods used to perform a further classification of AGNs, such as detailed optical emission-line spectra, or an accurate measurement of X-ray absorbing column densities *N*H. For these reasons, Ballantyne et al. (2006b) suggested the use of IRLFs for

**Figure 5.** Rest-frame 8.0-µm LF for all AGNs at *z* = 0.7, as predicted from the LDDE modelling of the HXLF of Ueda et al. (2003), Ebrero et al. (2009) and Aird et al. (2010) and from the LADE modelling of the HXLF of Aird et al. (2010). In each panel, the results for the six evolution models of *f* 2 are presented. The blue lines show the results for the three evolution models of *f* 2 given by Ballantyne et al. (2006a), who constrained the evolution of *f* 2 by fitting to the shape of the CXRB spectrum and X-ray number counts. The short dashed-dotted, long dashed-dotted and dotted lines show the results for the 'f2_1', 'f2_2' and 'f2_3' evolution models, respectively. The red lines show the results for the three evolution models of *f* 2 given by Hasinger (2008) from direct X-ray observations of the evolution of type 2 AGN fraction. Here, the short-dashed, long-dashed and solid lines show the results for the 'f2_4', 'f2_5' and 'f2_6' evolution models, respectively. The data points are the 8.0-µm IRS-decomposed AGN LFs of Fu et al. (2010) at *z* ∼ 0.7. The green dot-dashed lines show the obscuration-corrected AGN bolometric LF of Hopkins et al. (2007a) (taken from Fu et al. 2010). The purple solid lines show the LF for all AGN from Matute et al. (2006), combined and converted to 8.0 µm at *z* ∼ 0.7 by Fu et al. (2010).

**Figure 6.** Similar to Fig. 5, but for the 15-µm LF. The AGN obscuration evolution models of Ballantyne et al. (2006b) give better agreement with the measurements of Fu et al. (2010) and Matute et al. (2006), and with the results of Hopkins et al. (2007a).

all AGNs in order to distinguish the different evolution models of AGN obscuration. However, the results of Ballantyne et al. (2006b) show that the IRLF is not very sensitive to the evolution model of AGN obscuration except at much longer IR wavelengths, where the contaminant from star formation is important.

The separated IRLFs for type 1 and type 2 AGNs are expected to be much more sensitive to the overall evolution of AGN spatial density and obscuration, although a detailed classification is necessary. The classification of AGNs into type 1 and type 2 involves the problem of consistency between different classification methods. However, this could be a possible key to the problem of the evolution of AGN obscuration, because inconsistent classification methods will directly result in very different conclusions for the evolution of AGN obscuration. By separating the IRLFs into those for type 1 and type 2 AGNs, the intrinsic evolution of AGNs and the variations resulting from the evolution of AGN obscuration can be investigated in more detail, and possibly clarified. Furthermore, by considering the IRLFs for type 1 and type 2 AGNs separately, the modelling of the SEDs of AGNs can be further constrained. So, it would be more fruitful to separate the IRLFs of AGNs into those for type 1 and type 2. This could provide a more useful tool to explore the properties of the obscuring medium around different types of AGNs.

The separated IRLFs of type 1 and type 2 AGNs are given as

$$\frac{\mathrm{d}\Phi_{1}}{\mathrm{d}(\log\nu L_{v})}=\frac{\mathrm{d}\Phi}{\mathrm{d}(\log L_{X})}[1-f_{2}(\log L_{X},z)]\frac{\mathrm{d}(\log L_{X})}{\mathrm{d}(\log\nu L_{v})}\tag{21}$$
  
  
and

$$\frac{\mathrm{d}\Phi_{2}}{\mathrm{d}(\log vL_{v})}=\frac{\mathrm{d}\Phi}{\mathrm{d}(\log L_{\mathrm{X}})}f_{2}(\log L_{\mathrm{X}},z)\frac{\mathrm{d}(\log L_{\mathrm{X}})}{\mathrm{d}(\log vL_{v})},\tag{22}$$

-C 2012 The Authors, MNRAS **423,** 464–477 Monthly Notices of the Royal Astronomical Society -C 2012 RAS where *f* 2(log *L*X, *z*) is the fraction of type 2 AGNs. We have used the type 1 SED presented in Section 3 to predict the IRLFs of type 1 AGNs, while using the type 2 SED to predict the IRLFs of type 2 AGNs.

## **5 RESULTS**

In this section, we present the predicted mid-IR LFs for type 1, type 2 and all AGNs. The SEDs of AGNs, used to obtain the X-ray to IR luminosity relation, are computed by assuming a constant αox = −1.4, and *L*UV-dependent *R*in, as described by equation (17). We leave the discussion of *L*UV-dependent αox to Section 6. We present the results for all combinations of HXLF, as given in Section 2.1, and the different evolution models of AGN obscuration, as given in Section 2.2. Then, we compare these with the measurements of mid-IR LFs of AGNs currently available. Because we are mainly interested in discovering the more obvious trends, a simple qualitative comparison by eye is made here, rather than a more detailed fitting.3

## **5.1 Mid-IR LFs of all AGNs**

Figs 5 and 6 show the predicted rest-frame 8.0- and 15-µm LFs, respectively, for all AGNs. In each panel, the results are predicted from an evolution model of the HXLF and six evolution models of

3 Also, much more careful considerations of the covariance between points or systemic errors are not included.

AGN obscuration, as discussed in Section 2.2. The observational results used for comparison are from Fu et al. (2010). They used high-quality *Spitzer* 7–38 µm spectra to cleanly separate star formation and AGNs in individual galaxies for a 24-µm flux-limited sample of galaxies at *z* ∼ 0.7, and they decomposed the mid-IR LFs between star formation and AGNs.

As can clearly be seen in Figs 5 and 6,our results agree reasonably well with those of Fu et al. (2010), Matute et al. (2006) and Hopkins et al. (2007a). This general agreement shows that the methods we have used to model the SEDs of AGNs and to predict the corresponding mid-IR LFs from the HXLF are basically reasonable. Specifically, the different evolution models of HXLF give very similar results at 8.0 and 15 µm. However, the different evolution models of AGN obscuration are distinguishable at 15 µm, while they are not at 8.0 µm. As shown in Fig. 6, the results at 15 µm are divided into two groups, corresponding to the use of models from Ballantyne et al. (2006a) and the use of models constructed according to the recent results of Hasinger (2008). The results predicted using the evolution models of AGN obscuration from Ballantyne et al. (2006a) are in better agreement with the measurements of Fu et al. (2010) and the results from other authors, especially at the relative higher luminosities. It seems that the evolution of AGN obscuration is better described by the models of Ballantyne et al. (2006a) at the redshift and luminosity ranges covered by the measurements of Fu et al. (2010), that is, z -1 and νLν (8.0 µm, 15 µm) < 1012 L.

#### **5.2 Mid-IR LFs of type 1 and type 2 AGNs**

As mentioned above, it is better to separate the mid-IR LFs into those for type 1 and type 2 AGNs. Here, we present the mid-IR LFs for type 1 and type 2 AGNs and we then compare them with the mid-IR observational results of Brown et al. (2006) and Matute et al. (2006), respectively.

# *5.2.1 The 8.0-*μ*m LF of type 1 AGNs*

From a sample consisting of 292 24-µm sources brighter than 1 mJy, selected from the Multiband Imaging Photometer for *Spitzer* (MIPS) survey, Brown et al. (2006) have determined the rest-frame 8.0-µm LF for type 1 quasars with 1 < *z* < 5 and 1.5 < *z* < 2.5. Ballantyne et al. (2006b) used these results (in their fig. 13), but they compared them with the predicted mid-IR LFs for all AGNs. Despite this, they found that the predicted and measured LFs show a surprising agreement. As suggested by Brown et al. (2006), if the fraction of obscured quasars decreases rapidly with increasing luminosity, the type 1 quasar LF given by them would be approximate to the LF of all quasars at the highest luminosities. However, if there are very few type 2 AGNs at very high luminosities, and the type 1 AGN LF provides a good approximation of the total LF at high luminosities, then this would be an important constraint for the evolution of AGN obscuration at high luminosities.

**Figure 7.** Rest-frame 8.0-µm LF for type 1 AGNs at *z* = 1.5, 2.0 and 2.5, as predicted from the LDDE modelling of the HXLF of Ueda et al. (2003), Ebrero et al. (2009) and Aird et al. (2010), and from the LADE modelling of the HXLF of Aird et al. (2010). In each panel, the results for the six evolution models of *f* 2 are presented with line styles and colours that are the same as in Fig. 5. The data points show the measured 8.0-µm LF of type 1 quasars as determined by Brown et al. (2006) from a sample consisting of 292 24-µm sources brighter than 1 mJy and selected from the MIPS survey. The black solid points denote the results for objects over the redshift range 1 < *z* < 5, and the green triangles for those with 1.5 < *z* < 2.5. All the results predicted from the HXLF, especially those recently presented, tend to underestimate the number of IR-luminous AGNs, independent of the choices of the evolution of HXLF and obscuration.

This important information was not fully utilized in the method of Ballantyne et al. (2006b). So, it would be more reasonable and worthwhile to predict the mid-IR LFs for only type 1 AGNs from different evolution models of the HXLF and AGN obscuration, and then to compare them with the measurements of Brown et al. (2006).

Fig. 7 presents the predicted rest-frame 8.0-µm LF of type 1 AGNs at *z* = 1.5, 2.0 and 2.5. We compare these with the measurements of Brown et al. (2006). As expected, the results predicted from different evolution models of AGN obscuration are more distinguishable when we use the rest-frame 8.0-µm LF for only type 1 AGNs, instead of all AGNs. The rest-frame 8.0-µm LF of type 1 AGNs predicted from different choices of the HXLF tends to underestimate the number of AGNs as measured by Brown et al. (2006), no matter which evolution model of AGN obscuration is used. Surprisingly, when using the HXLF of Ueda et al. (2003), there is better agreement with the measurements of Brown et al. (2006) than when using all the more recent HXLF measurements. The results shown here might, to some extent, confirm the credibility of the widely used results of Ueda et al. (2003). However, this could just be a coincidence, because the more recent observational determinations of the HXLF are generally expected to be more accurate. More reasonable conclusions can only be obtained from comparisons with other independent measurements.

# *5.2.2 The 15-*μ*m LF of type 1/type 2 AGNs*

From a sample of AGNs selected at 15 µm, from the *Infrared Space Observatory* (*ISO*), and 12 µm, from the *IRAS*, Matute et al. (2006) measured the rest-frame 15-µm LF of type 1 and type 2 AGNs, which are classified separately based on their optical spectra. Figs 8 and 9 show the predicted rest-frame 15-µm LF for type 1 and type 2 AGNs. We compare these with the results of Matute et al. (2006).

Fig. 8 presents the rest-frame 15-µm LF of type 1 AGNs at *z* = 0.1 and 1.2. As can clearly be seen, the predicted results show reasonable agreement with the measurements of Matute et al. (2006). However, it is also clear, especially at *z* = 1.2, that the predicted IRLFs tend to underestimate the number of the most IR-luminous type 1 AGNs, which is independent of the choices of the evolution of the HXLF and obscuration. Interestingly, the measurements at *z* = 0.1 can basically be explained by the results predicted from most HXLFs. The only exception is the result predicted from the HXLF of Aird et al. (2010) modelled with LADE, which significantly underestimates the number of the most IR-luminous AGNs, even at *z* = 0.1. Similar to Fig. 7, these results show that the mid-IR LFs predicted from the HXLF tend to underestimate the number of the most IR-luminous AGNs, and to become significant at z 1.

Fig. 9 presents the rest-frame 15-µm LF of type 2 AGNs at *z* = 0.05 and 0.35. As mentioned by Matute et al. (2006), the observational determination of the rest-frame 15-µm LF of type 2 AGNs is much poorer than that of type 1 AGNs. So, their measurement of the density of type 2 AGNs can only be considered as a lower limit. However, as can be seen in Fig. 9, the results predicted from the AGN obscuration evolution models, which are constructed according to the results of Hasinger (2008), tend to underestimate even the number of type 2 AGNs currently measured. Because of the much larger uncertainties in the measurements of the 15-µm LF of type 2 AGNs, more definitive conclusions cannot be drawn.

## **6 DISCUSSION**

By separating the mid-IR LFs of AGNs into those for type 1 and type 2 AGNs, the modelling of the SEDs of AGNs, the evolution of LFs and obscuration of AGNs can be further constrained. The results presented in Section 5 show that the mid-IR LFs predicted from the HXLF tend to underestimate the number of the most IRluminous AGNs, despite the general agreement between predictions and measurements. This is independent of the choices of the evolution models of the HXLF and obscuration of AGNs, and it is even more obvious for the HXLFs recently proposed. Meanwhile, this

**Figure 8.** Rest-frame 15-µm LF for type 1 AGNs at *z* = 0.1 and 1.2, as predicted from the LDDE modelling of the HXLF of Ueda et al. (2003), Ebrero et al. (2009) and Aird et al. (2010), and the LADE modelling of the HXLF of Aird et al. (2010). The data points are the measured 15-µm LF of type 1 AGNs determined by Matute et al. (2006) from a sample of type 1 AGNs with redshift in *z* = [0, 0.2] (top) and *z* = [0.2, 2.2] (bottom) selected at 15 µm (*ISO*) and 12 µm (*IRAS*), and classified by their optical spectra. In each panel, the results for the six evolution models of *f* 2 are presented with line styles and colours as in Fig. 5. The results predicted from all HXLFs tend to underestimate the number of the most IR-luminous AGNs at *z* = 1.2. However, this is not the case at *z* = 0.1, unless the LADE modelling of the HXLF of Aird et al. (2010) is used.

**Figure 9.** Similar to Fig. 8, but for type 2 AGNs at *z* = 0.05 and 0.35. The data points are the measured 15-µm LF of type 2 AGNs determined by Matute et al. (2006) from a sample of type 2 AGNs with redshift in *z* = [0, 0.1] (top) and *z* = [0.1, 0.6] (bottom) selected at 15 µm (*ISO*) and 12 µm (*IRAS*), and classified by their optical spectra. The measurement of the mid-IR LF of type 2 AGNs is much poorer than that of type 1 AGNs, and therefore it is much harder to explain. However, the results predicted from the AGN obscuration evolution model, constructed according to the results of Hasinger (2008), are below even the current measurements.

trend does not seem to be significant for AGNs with z - 1 and/or AGNs that are less luminous at IR.

Here, we discuss some possible explanations for this contradiction between HXLFs and mid-IR LFs. First, this might be caused by the missing fraction of AGNs, especially those heavily obscured Compton-thick AGNs that cannot be detected by current X-ray observations. Recently, Fu et al. (2010) have compared their mid-IR spectroscopic selection with other AGN identification methods and they have concluded that only half of the mid-IR spectroscopically selected AGNs were detected in X-ray. However, after considering

**Figure 10.** Similar to Fig. 7, but now αox is assumed to have a smaller value of −1.5 instead of the typical value of −1.4, as used in Section 5. As can be seen, the results have been greatly improved after this small change. Now, the measurements of Brown et al. (2006) can easily be explained by the recently proposed HXLF, especially that of Ebrero et al. (2009), when combined with the obscuration evolution models constructed according to the results of Hasinger (2008). An even smaller αox seems to be required at νLν (8.0 µm) -5 × 1012 L.

**Figure 11.** Similar to Fig. 8, but now αox is assumed to have a smaller value of −1.5 at *z* = 1.2 instead of the typical value of −1.4, as used in Section 5. At *z* = 1.2, it is clear that the results at the highest luminosities have been greatly improved. Now, the AGN obscuration evolution models proposed by Ballantyne et al. (2006a) seem more favourable. However, the change of αox is not required at *z* = 0.1, unless the LADE modelling of the HXLF of Aird et al. (2010) is used.

this, we find that it only results in a slight improvement to the prediction of IRLFs from the HXLF. Furthermore, this explanation needs a larger fraction of missing AGNs at the high-luminosity end, which is in contradiction with the general expectation that AGNs dominate in the most IR-luminous sources.

Secondly, the contribution of star formation in the AGN host to mid-IR, which has not been considered yet, might be important. If this is important, the X-ray to mid-IR relation, which is used to predict mid-IR LFs from HXLFs in Section 5, needs to be corrected significantly. This is particularly important for sources that are not spatially resolved, or those with intensive star formation near the nuclear region (Lutz et al. 2004; Horst et al. 2008). We find that if the contribution of star formation in the host to 8.0- and 15-µm emission is comparable to the reprocessed nuclear emission, the 8.0 µm and 15-µm LFs predicted from the HXLF could be consistent with the corresponding mid-IR measurements. Currently, it is still difficult to separate the contribution of star formation and AGNs to the IR emission of galaxies, especially in systems where the two are comparable and their additive effects are non-linear (Hopkins et al. 2010). In particular, the relative fractions of their contributions to mid-IR are likely to be different in different types of galaxies, and they can change with both luminosity and redshift of the source. However, even for powerfully star-forming quasars, the contribution of star formation to mid-IR is small (Netzer et al. 2007). So, the possible contribution of star formation to mid-IR is not likely to be the main reason for the contradiction.

Thirdly, the contradiction found in Section 5 could represent limitations in the torus model used so far. Although the simple torus model inherited from Ballantyne et al. (2006b) has been well tested, the distribution and composition of the obscuring mediums around AGNs are still very uncertain. Meanwhile, this CLOUDYbased torus model essentially assumes a smooth distribution of dusty obscuring mediums. Recently, a clumpy distribution of dusty obscuring mediums has been suggested (Nenkova, Ivezic & Elitzur ´ 2002; Honig et al. 2006). These authors have recently proposed ¨ sophisticated clumpy torus models (Nenkova et al. 2008a,b; Honig ¨ et al. 2010; Honig & Kishimoto 2010) that are in better agree- ¨ ment with current IR observations of AGNs. Unfortunately, these clumpy torus models mainly give the IR emission properties of AGNs, while the self-consistent hard X-ray property is not presented. To give the X-ray to mid-IR luminosity ratios that are more comparable to the observational results of Mullaney et al. (2011), we have made some improvements to the original torus model of Ballantyne et al. (2006b). However, the improved torus model still has some limitations. It is worth additional effort to improve it further, but this is beyond the scope of this paper.

Finally, as shown in Section 3.3, the anticorrelation between αox and *L*UV, which has been found by many observations, is important for obtaining X-ray to mid-IR luminosity ratios that are more comparable to the results of Mullaney et al. (2011). Here, we present the results obtained by assuming αox = −1.5 instead of the typical value of −1.4, as used in Section 5. As can be seen in Figs 10 and 11, the mid-IR LFs measurements of Brown et al. (2006) and Matute et al. (2006) can now be explained much better by the results predicted from the HXLFs recently proposed, especially those of Ebrero et al. (2009). However, it is interesting to notice the dramatic difference between the results in Figs 10 and 11. The 8.0-µm LF measurement of Brown et al. (2006) is for luminous quasars with *z* > 1, while the 15-µm LF of Matute et al. (2006) is mainly for Seyfert galaxies at much lower luminosities and redshifts. While the results at 8.0 µm favour the obscuration evolution models constructed according to the results of Hasinger (2008), the results at 15 µm nevertheless give more support to the models proposed by Ballantyne et al. (2006a).

These results imply that the obscuration of quasars are different from those of Seyfert galaxies. Luminous quasars are often associated with galaxy major mergers (Canalizo & Stockton 2001) or interactions (Hutchings 1987; Disney et al. 1995; Bahcall et al. 1997; Kirhakos et al. 1999), while there is little observational evidence for less-luminous Seyfert galaxies being associated with mergers (Laurikainen & Salo 1995; Schmitt 2001; Grogin et al. 2005). If the evolution and fuelling mechanisms of quasars are very different from those of lower-luminosity Seyfert galaxies, it is natural to expect that the distribution and evolution of the obscuring medium around them are very different. As pointed out by Ballantyne et al. (2006b), the dusty mediums obscuring luminous quasars are likely to be distributed on a larger scale and to be linked to the starburst region, while less-luminous quasars and Seyfert galaxies are obscured by the commonly suggested compact torus located at a much smaller scale.

## **7 SUMMARY**

We have presented a detailed comparison between the 2–10 keV HXLFs and mid-IR LFs of AGNs. The combination of hard X-ray and mid-IR provides complementary views for understanding the evolution of LFs and the obscuration of AGNs and their coevolution with galaxies. Four measurements of the HXLFs of AGNs have been collected from the literature, for comparison. A simple but well-tested torus model, which is based on the photoionization and radiative transfer code CLOUDY, is then employed to model the composite X-ray to IR SEDs for AGNs with different luminosities and redshifts. In the modelling of SEDs, we have assumed six evolution models of AGN obscuration, which are constrained by the CXRB (Ballantyne et al. 2006a), or constructed according to recent direct measurements (Hasinger 2008). The model SEDs of AGNs have been tested against the observational relations between the Xray and mid-IR luminosities of AGNs recently given by Mullaney et al. (2011). The mid-IR LFs predicted from different combinations of the evolution models of the HXLF and obscuration of AGNs are compared with the measurements of AGN mid-IR LFs given by Brown et al. (2006), Matute et al. (2006) and Fu et al. (2010). By predicting mid-IR LFs for type 1, type 2 and all AGNs from the HXLF, and by comparing them with the corresponding observational results, the evolution of LFs and obscuration of AGN can be understood further.

We find that the mid-IR LFs predicted from HXLFs tend to underestimate the number of the most IR-luminous AGNs, which is independent of the evolution model of AGN obscuration. We have discussed the possible explanations for this contradiction. This could partly be because of the missing fraction of Compton-thick AGNs that have been predicted by the synthesis model of the CXRB, but systematically missed by current X-ray observations. However, we find that the underestimation of the number of the most IR-luminous AGNs cannot be eliminated, even if an extreme assumption is used, in which it is claimed that only half of the mid-IR spectroscopically selected AGNs are detected in current X-ray observations.

We conclude that the contradiction mainly results from limitations in the modelling of the composite X-ray to IR SEDs of AGNs. A possible reason is the contribution of star formation in the AGN host to mid-IR, which has not yet been considered. We find that the contribution of star formation to the 8.0- and 15-µm emission needs to be comparable with that of reprocessed nuclear emission, and even more in the most IR-luminous sources, in order to eliminate the contradiction. However, the contribution of star formation in the AGN host to mid-IR is not likely to be so large, but it actually decreases with increasing *L*IR. However, the contradiction could represent limitations in the torus model employed. It is clear that the torus model is further constrained to give the specific prediction of mid-IR LFs for type 1 and type 2 AGNs. We have made some improvements to the original torus model of Ballantyne et al. (2006b), such as a different handling of the diffuse emission and *L*UV-dependent *R*in, in order to give X-ray to mid-IR luminosity ratios that are more comparable to the observational results of Mullaney et al. (2011). Meanwhile, with some tests, we find that the anticorrelation between αox and *L*UV is important for bringing the X-ray to mid-IR luminosity ratios closer to the results of Mullaney et al. (2011). Interestingly, a smaller αox improves the prediction of the high-*L*IR end of the IRLFs significantly at the same time.

Finally, with all the improvements mentioned above, we find that the HXLFs and IRLFs of AGNs can be more consistent with each other if the obscuration mechanisms of quasars and Seyfert galaxies are assumed to be different. This is consistent with the idea that the obscuration mechanism of luminous quasars dominating at high redshifts is very different from that of less-luminous Seyfert galaxies dominating at lower redshifts, corresponding to their different triggering and fuelling mechanisms. However, current measurements of the IRLFs of AGNs are not accurate enough to allow a more complete understanding of this when using the method presented here. Because of this limitation, the conclusions drawn here need to be tested further when better measurements of IRLFs are available.

More accurate measurements of the IRLFs of AGNs, especially those determined at smaller redshift bins and more accurately separated into those for type 1 and type 2, would be very helpful for a more complete understanding of the evolution of LFs and obscuration of AGNs. Based on the observations of newly launched IR space telescopes, such as *Spitzer*, *Herschel* and the forthcoming *JWST*, better measurements of the IRLFs of AGNs are expected. These measurements will greatly improve our understanding of the evolution of LFs and obscuration of AGNs and their coevolution with galaxies.

## **ACKNOWLEDGMENTS**

We are grateful to the anonymous referee for reviewing our paper very carefully and for the relevant comments on our work. We thank the previous anonymous referee for comments and suggestions that greatly improved this paper. This work is supported by the National Natural Science Foundation of China (Grant Nos. 10778702, 11033008, 11063003 and 11103072), the National Basic Research Programme of China (Grant No. 2009CB824800) and the Chinese Academy of Sciences (Grant No. KJCX2-YW-T24).

## **REFERENCES**

- Aird J. et al., 2010, MNRAS, 401, 2531
- Alonso-Herrero A., Quillen A. C., Simpson C., Efstathiou A., Ward M. J., 2001, AJ, 121, 1369
- Antonucci R., 1993, ARA&A, 31, 473
- Babbedge T. S. R. et al., 2006, MNRAS, 370, 1159
- Bahcall J. N., Kirhakos S., Saxe D. H., Schneider D. P., 1997, ApJ, 479, 642
- Baldwin J. A., Phillips M. M., Terlevich R., 1981, PASP, 93, 5
- Ballantyne D. R., 2008, ApJ, 685, 787
- Ballantyne D. R., Everett J. E., Murray N., 2006a, ApJ, 639, 740
- Ballantyne D. R., Shi Y., Rieke G. H., Donley J. L., Papovich C., Rigby J. R., 2006b, ApJ, 653, 1070
- Barcons X. et al., 2007, A&A, 476, 1191
- Barger A. J., Cowie L. L., Mushotzky R. F., Yang Y., Wang W-H., Steffen A. T., Capak P., 2005, AJ, 129, 578
- Bongiorno A., Zamorani G., Gavignaud I., Marano, 2007, A&A, 472, 443
- Bower R. G., Benson A. J., Malbon R., Helly J. C., Frenk C. S., Baugh C. M., Cole S., Lacey C. G., 2006, MNRAS, 370, 645
- Brown M. J. I. et al., 2006, ApJ, 638, 88

Brusa M. et al., 2010, ApJ, 716, 348

- Canalizo G., Stockton A., 2001, ApJ, 555, 719
- Cirasuolo M., Magliocchetti M., Celotti A., 2005, MNRAS, 357, 1267
- Croom S. M., Smith R. J., Boyle B. J., Shanks T., Miller L., Outram P. J., Loaring N. S., 2004, MNRAS, 349, 1397
- Croton D. J. et al., 2006, MNRAS, 365, 11
- Davies R. I. et al., 2006, ApJ, 646, 754
- Di Matteo T., Springel V., Hernquist L., 2005, Nat, 433, 604

- Di Matteo T., Colberg J., Springel V., Hernquist L., Sijacki D., 2008, ApJ, 676, 33 Disney M. J. et al., 1995, Nat, 376, 150 Draper A. R., Ballantyne D. R., 2011, ApJ, 729, 109 Ebrero J. et al., 2009, A&A, 493, 55 Fan X. et al., 2001, AJ, 121, 54 Ferland G., Korista K., Verner D., Ferguson J., Kingdon J., Verner E., 1998, PASP, 110, 761 Ferrarese L., Merritt D., 2000, ApJ, 539, L9 Fontanot F., Cristiani S., Monaco P., Nonino M., Vanzella E., Brandt W. N., Grazian A., Mao J., 2007, A&A, 461, 39 Fu H. et al., 2010, ApJ, 722, 653 Gandhi P., Horst H., Smette A., Honig S., Comastri A., Gilli R., Vignali C., ¨ Duschl W., 2009, A&A, 502, 457 Gebhardt K. et al., 2000, ApJ, 539, L13 Gilli R., Comastri A., Hasinger G., 2007, A&A, 463, 79 Goulding A. D., Alexander D. M., 2009, MNRAS, 398, 1165 Graham A. W., 2012, ApJ, 746, 113 Grogin N. A. et al., 2005, ApJ, 627, L97 Gultekin K. et al., 2009, ApJ, 698, 198 ¨ Hao L. et al., 2005, AJ, 129, 1795 Haring N., Rix H.-W., 2004, ApJ, 604, L89 ¨ Hasinger G., 2004, NuPhS, 132, 86 Hasinger G., 2008, A&A, 490, 905 Hasinger G., Miyaji T., Schmidt M., 2005, A&A, 441, 417 Honig S. F., Kishimoto M., 2010, A&A, 523, A27 ¨ Honig S. F., Beckert T., Ohnaka K., Weigelt G., 2006, A&A, 452, 459 ¨
- Honig S. F., Kishimoto M., Gandhi P., Smette A., Asmus D., Duschl W., ¨ Polletta M., Weigelt G., 2010, A&A, 515, A23
- Hopkins A. M., 2004, ApJ, 615, 209
- Hopkins P. F., Hernquist L., Cox T. J., Di Matteo T., Martini P., Robertson B., Springel V., 2005, ApJ, 630, 705
- Hopkins P. F., Somerville R. S., Hernquist L., Cox T. J., Robertson B., Li Y., 2006, ApJ, 652, 864
- Hopkins P. F., Richards G. T., Hernquist L., 2007a, ApJ, 654, 731
- Hopkins P. F., Hernquist L., Cox T. J., Robertson B., Krause E., 2007b, ApJ, 669, 67
- Hopkins P. F., Hernquist L., Cox T. J., Keres D., 2008, ApJ, 175, 356
- Hopkins P. F., Younger J. D., Hayward C. C., Narayanan D., Hernquist L., 2010, MNRAS, 402, 1693
- Horst H., Smette A., Gandhi P., Duschl W. J., 2006, A&A, 457, L17
- Horst H., Gandhi P., Smette A., Duschl W. J., 2008, A&A, 479, 389
- Hutchings J. B., 1987, ApJ, 320, 122
- Jahnke K., Maccio A. V., 2011, ApJ, 734, 92 `
- Kauffmann G. et al., 2003, MNRAS, 346, 1055
- Kewley L. J., Dopita M. A., Smith H. A., 2001, Bull. Amer. Astron. Soc., 33, 1365
- Kewley L. J., Groves B., Kauffmann G., Heckman T., 2006, MNRAS, 372, 961
- Khachikian E. Y., Weedman D. W., 1974, ApJ, 192, 581
- Kirhakos S., Bahcall J. N., Schneider D. P., Kristian J., 1999, ApJ, 520, 67
- Kormendy J., Bender R., 2009, ApJ, 691, L142
- Kormendy J., Richstone D., 1995, ARA&A, 33, 581
- Krabbe A., Boker T., Maiolino R., 2001, ApJ, 557, 626
- Krolik J. H., 1999, Active Galactic Nuclei: From the Central Black Hole to the Galactic Environment. Princeton Univ. Press, Princeton, NJ
- Kuraszkiewicz J. K. et al., 2003, ApJ, 590, 128
- La Franca F. et al., 2005, ApJ, 635, 864
- Laurikainen E., Salo H., 1995, A&A, 293, 683
- Liu Y., Zhang S. N., 2011, ApJ, 728, L44
- Lutz D., Maiolino R., Spoon H. W. W., Moorwood A. F. M., 2004, A&A, 418, 465
- Magorrian J. et al., 1998, AJ, 115, 2285
- Maiolino R., Rieke G. H., 1995, ApJ, 454, 95
- Marconi A., Hunt L. K., 2003, ApJ, 589, L21
- Marconi A., Risaliti G., Gilli R., Hunt L. K., Maiolino R., Salvati M., 2004, MNRAS, 351, 169
- Matt G. et al., 1997, A&A, 325, L13
- Matute I., La Franca F., Pozzi F., Gruppioni C., Lari C., Zamorani G., 2006, A&A, 451, 443
- Merloni A., 2004, MNRAS, 353, 1035
- Merloni A. et al., 2010, ApJ, 708, 137
- Miyaji T., Hasinger G., Schmidt M., 2000, A&A, 353, 25
- Miyaji T., Hasinger G., Schmidt M., 2001, A&A, 369, 49
- Moran E. C., Filippenko A. V., Chornock R., 2002, ApJ, 579, L71
- Mullaney J. R., Alexander D. M., Goulding A. D., Hickox R. C., 2011, MNRAS, 414, 1082
- Nagar N. M., Falcke H., Wilson A. S., 2005, A&A, 435, 521
- Nenkova M., Ivezic´ Z., Elitzur M., 2002, ApJ, 570, L9 ˇ
- Nenkova M., Sirocky M. M., Ivezic v. Z., Elitzur M., 2008a, ApJ, 685, 147 ´
- Nenkova M., Sirocky M. M., Nikutta R., Ivezic v. Z., Elitzur M., 2008b, ´
- ApJ, 685, 160 Netzer H., Mainieri V., Rosati P., Trakhtenbrot B., 2006, A&A, 453, 525
- Netzer H. et al., 2007, ApJ, 666, 806
- Peng C. Y., 2007, ApJ, 671, 1098
- Pier E. A., Krolik J. H., 1992, ApJ, 401, 99
- Richards G. T. et al., 2006, AJ, 131, 2766
- Rigby J. R., Rieke G. H., Donley J. L., Alonso-Herrero A., Perez-Gonzalez P. G., 2006, ApJ, 645, 115
- Risaliti G., Maiolino R., Salvati M., 1999, ApJ, 522, 157
- Risaliti G., Elvis M., Nicastro F., 2002, ApJ, 571, 234
- Risaliti G., Elvis M., Fabbiano G., Baldi A., Zezas A., 2005, ApJ, 623, L93
- Rowan-Robinson M., 1977, ApJ, 213, 635
- Schmitt H. R., 2001, AJ, 122, 2243
- Shankar F., Mathur S., 2007, ApJ, 660, 1051
- Silverman J. D. et al., 2005a, ApJ, 624, 630
- Silverman J. D. et al., 2005b, ApJ, 618, 123
- Silverman J. D. et al., 2008, ApJ, 679, 118
- Simpson C., 2005, MNRAS, 360, 565
- Spergel D. N. et al., 2003, ApJS, 148, 175
- Steffen A. T., Barger A. J., Cowie L. L., Mushotzky R. F., Yang Y., 2003, ApJ, 596, L23
- Steffen A. T., Strateva I., Brandt W. N., Alexander D. M., Koekemoer A. M., Lehmer B. D., Schneider D. P., Vignali C., 2006, AJ, 131, 2826
- Tozzi P. et al., 2006, A&A, 451, 457
- Treister E., Urry C. M., 2006, ApJ, 652, L79
- Treister E. et al., 2004, ApJ, 616, 123
- Treister E. et al., 2006, ApJ, 640, 603
- Treister E., Natarajan P., Sanders D. B., Urry C. M., Schawinski K., Kartaltepe J., 2010, Sci, 328, 600
- Tremaine S. et al., 2002, ApJ, 574, 740
- Ueda Y., Akiyama M., Ohta K., Miyaji T., 2003, ApJ, 598, 886
- Vagnetti F., Turriziani S., Trevese D., Antonucci M., 2010, A&A, 519, A17
- Wang J. M., Zhang E. P., 2007, ApJ, 660, 1072
- Wang J-M., Zhang E-P., Luo B., 2005, ApJ, 627, L5
- Wolf C., Wisotzki L., Borch A., Dye S., Kleinheinrich M., Meisenheimer K., 2003, A&A, 408, 499
- Yencho B., Barger A. J., Trouille L., Winter L. M., 2009, ApJ, 698, 380
- Yu Q., Tremaine S., 2002, MNRAS, 335, 965
- Zhang S. N., 2004, ApJ, 618, L79
- Zhang E. P., Wang J. M., 2006, ApJ, 653, 137

This paper has been typeset from a TEX/LATEX file prepared by the author.

