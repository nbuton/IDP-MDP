# IDP-MDP (IDP Molecular Dynamics Properties)
Tool to create the DynaFold Benchmark

**Objective:** Evaluate model that predict conformational set properties directly from sequence


## Pipeline
1- Download data from IDRome (https://github.com/KULL-Centre/_2023_Tesei_IDRome?tab=readme-ov-file) and put this inside the data folder  
2- Run pipeline/compute_all_backmapping.py to do the backmapping and have the full atom version for the 28058 IDPs of IDRome  
3- Run pipeline/compute_all_properties.py to compute all properties from conformation set of each IDP  
4- Run python3 pipeline/split_and_create_dataset.py to create train/validation/test splits to have the final benchmark  

## Computed properties
L: length of the proteins  
N: number of frames/conformations  

### End-to-End Distance ($R_{ee}$)
The Euclidean distance in 3D between the $C_{\alpha}$ atom of the first residue and the last residue of the sequence.

**Formula:**
$$R_{ee} = \sqrt{(\mathbf{r}_N - \mathbf{r}_1)^2} = |\mathbf{r}_N - \mathbf{r}_1|$$

**Shape:** (N,)

---

### Radius of Gyration ($R_g$)
Measures the distribution of the protein's mass relative to its center of mass, indicating structural compactness.
$$R_g = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\mathbf{r}_i - \mathbf{r}_{cm})^2}$$


$$\mathbf{r}_{cm} = \frac{\sum_{i=1}^{N} m_i \mathbf{r}_i}{\sum_{i=1}^{N} m_i}$$

Where:
* $m_i$ is the mass of atom $i$.
* $\mathbf{r}_i$ is the position vector of atom $i$.
* $N$ is the total number of atoms.

**Shape:** (N,)

---

### Gyration Tensor ($\mathbf{S}$)
The gyration tensor is a $3 \times 3$ matrix that characterizes the shape and orientation of the protein.

**Formula:**
The elements $S_{mn}$ (where $m, n \in \{x, y, z\}$) of the tensor are calculated as:  

$$S_{mn} = \frac{1}{N} \sum_{i=1}^{N} (r_{i,m} - r_{cm,m})(r_{i,n} - r_{cm,n})$$

For the mass-weighted version:

$$S_{mn} = \frac{\sum_{i=1}^{N} m_i (r_{i,m} - r_{cm,m})(r_{i,n} - r_{cm,n})}{\sum_{i=1}^{N} m_i}$$

The matrix is symmetric:

$$\mathbf{S} = \begin{pmatrix} S_{xx} & S_{xy} & S_{xz} \\ S_{yx} & S_{yy} & S_{yz} \\ S_{zx} & S_{zy} & S_{zz} \end{pmatrix}$$

**Eigenvalues:**
Diagonalizing this matrix yields three principal moments (eigenvalues) $\lambda_1, \lambda_2, \lambda_3$ (usually ordered such that $\lambda_1 \ge \lambda_2 \ge \lambda_3$). 
The Radius of Gyration is related to these eigenvalues by:
$$R_g^2 = \lambda_1 + \lambda_2 + \lambda_3$$

### Asphericity ($b$)
Asphericity measures the non-spherical nature of the particle distribution. It is calculated using the eigenvalues ($\lambda_i$) of the gyration tensor. For a perfectly spherical object, $b = 0$.

**Formula:**
$$\Delta=\frac{\left(\lambda_1-\lambda_2\right)^2+\left(\lambda_1-\lambda_3\right)^2+\left(\lambda_2-\lambda_3\right)^2}{2\left(\lambda_1+\lambda_2+\lambda_3\right)^2}$$

Where:
* $\lambda_1, \lambda_2, \lambda_3$ are the principal moments (eigenvalues) of the gyration tensor, typically ordered such that $\lambda_1 \ge \lambda_2 \ge \lambda_3$.

**Shape:** (N,)

### Prolateness ($S$)
Determines if the shape is elongated like a cylinder (prolate, $S > 0$) or flattened like a disk (oblate, $S < 0$).

$$S = \frac{(2\lambda_1 - \lambda_2 - \lambda_3)(2\lambda_2 - \lambda_1 - \lambda_3)(2\lambda_3 - \lambda_1 - \lambda_2)}{2(\lambda_1^2 + \lambda_2^2 + \lambda_3^2 - \lambda_1\lambda_2 - \lambda_1\lambda_3 - \lambda_2\lambda_3)^{3/2}}$$

**Shape:** (N,)

### Normalized acylindricity ($c$)

Acylindricity, denoted as $c$, measures the deviation from cylindrical symmetry.1 It is defined as:

$$ \large c_{norm} = \frac{(\lambda_2 - \lambda_3)}{R_g^2} $$

**Shape:** (N,)

### Relative Shape Anisotropy ($\kappa$)
The relative shape anisotropy ($\kappa^2$) is perhaps the most robust single measure of asymmetry, as it integrates asphericity and acylindricity into a single dimensionless value.1 It is defined as:

$$ \kappa^2=1-3 \frac{\lambda_1 \lambda_2+\lambda_2 \lambda_3+\lambda_3 \lambda_1}{\left(\lambda_1+\lambda_2+\lambda_3\right)^2} $$

**Shape:** (N,)

---

### Scaling Exponent ($\nu$)
The scaling exponent characterizes the solvent quality and the conformational state of the polymer (protein) chain. It is derived from the power-law relationship between the number of residues ($L$) and the Radius of Gyration ($R_g$).

**Power Law Formula:**
$$R_g \approx R_0 L^\nu$$

**Calculation via Linear Regression:**
To compute $\nu$ from experimental or simulation data, a linear fit is performed on a log-log scale:

$$\ln(R_g) = \nu \ln(L) + \text{constant}$$

**Physical Interpretations:**
* **$\nu \approx 0.33$ (Poor solvent):** The protein is in a collapsed, globular/folded state.
* **$\nu \approx 0.50$ (Theta solvent):** The protein behaves like an ideal Gaussian chain (polymer-polymer and polymer-solvent interactions balance out).
* **$\nu \approx 0.588$ (Good solvent):** The protein is in an expanded, "self-avoiding walk" state (typical for Intrinsically Disordered Proteins or denatured states).

**Shape:** (N,)

---

### Secondary Structure Propensity
The statistical likelihood of a specific residue sequence to form $\alpha$-helices, $\beta$-sheets, or random coils, often calculated via DSSP or STRIDE algorithms.

**Shape:** (L, 3)

---

### Dihedral Entropy per Residue ($S_{dih, i}$)
Dihedral entropy quantifies the level of disorder or conformational flexibility for a specific residue $i$ by analyzing the probability distribution of its backbone torsion angles ($\phi, \psi$).

**Formula (Shannon Entropy):**
For a specific residue $i$, the entropy is calculated by integrating over the 2D dihedral probability density $P_i(\phi, \psi)$:

$$S_{dih, i} = -k_B \int_{0}^{2\pi} \int_{0}^{2\pi} P_i(\phi, \psi) \ln P_i(\phi, \psi) \, d\phi \, d\psi$$

Where:
* $k_B$ is the Boltzmann constant (often omitted or set to $R$ for molar units).
* $P_i(\phi, \psi)$ is the normalized probability of finding residue $i$ at a specific $(\phi, \psi)$ coordinate.

**Numerical Approximation (Circular Binning):**
In practice, the $(\phi, \psi)$ space is divided into discrete bins (e.g., $10^\circ \times 10^\circ$). The formula becomes:

$$S_{dih, i} \approx -k_B \sum_{j} p_j \ln p_j$$

Where:
* $p_j$ is the frequency of the residue appearing in bin $j$.
* The sum is taken over all bins where $p_j > 0$.

**Interpretation:**
* **Low Entropy:** The residue is restricted to a small region of the Ramachandran plot (e.g., a stable $\alpha$-helix or a rigid binding motif).
* **High Entropy:** The residue samples a wide variety of conformations, typical of highly flexible linkers in IDPs.

**Shape:** (L, 2)

---

## Distance Fluctuation

**Definition:**  
This metric measures the standard deviation of the distance between pairs of residues over time. In biophysics, this physical fluctuation is interpreted as "Communication Propensity": pairs with low fluctuations are considered to have high communication efficiency, as they are mechanically coupled.

**Formula:**  
The communication propensity between residues $i$ and $j$ is the standard deviation of their inter-residue distance:

$$\mathrm{CP}_{ij} = \sqrt{\langle (d_{ij} - \langle d_{ij} \rangle)^2 \rangle} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (d_{ij}(t) - \bar{d}_{ij})^2}$$

Where:  
- $d_{ij}(t)$ is the Euclidean distance between $C_\alpha$ atoms of residues $i$ and $j$ at time $t$.  
- $\bar{d}_{ij}$ is the time-averaged distance between those two residues.  
- $T$ is the total number of frames in the trajectory.

**Interpretation:**  
- **High $\mathrm{CP}_{ij}$ (High Fluctuation):** Indicates a "noisy" or flexible link. Structural signals are likely lost due to independent thermal motions.  
- **Low $\mathrm{CP}_{ij}$ (Low Fluctuation):** Indicates a "rigid" or coordinated link. These residues move in sync, allowing for efficient allosteric communication or signal transduction across the protein.

**Applications:**  
This is a primary tool for identifying allosteric pathways in proteins. In Intrinsically Disordered Proteins (IDPs), it helps distinguish between purely random motion and coordinated "functional" disorder.

**Shape:** (L, L)

---
### Motion Correlation (DCCM) - TO REMOVE because comparaison with artificial mean not pertinent for IDP?
The Dynamic Cross-Correlation Map (DCCM) represents the correlated displacements of $C_{\alpha}$ atoms.  

$$C_{ij} = \frac{\langle \Delta \mathbf{r}_i \cdot \Delta \mathbf{r}_j \rangle}{\sqrt{\langle \Delta \mathbf{r}_i^2 \rangle \langle \Delta \mathbf{r}_j^2 \rangle}}$$


**Implementation Tip:**
A **low value** of $CP_{ij}$ indicates high communication efficiency. Because this metric is essentially a "noise" measurement, residues with very low fluctuations are considered to be "mechanically coupled," allowing signals to propagate without being lost to random thermal motion.

**Why compute this for IDPs?**
In disordered proteins, most pairs will have very high $CP$ (high noise). However, "Functional IDPs" often have hidden pathways where specific residues maintain low $CP$ despite being far apart in the sequence, suggesting a conserved signaling mechanism.


### Contact Map Frequency
The probability $P_{ij}$ that two residues $i$ and $j$ are within a specific spatial cutoff (usually $8$ Å) throughout a simulation.

**Shape:** (L, L)

---

### Local Chirality ($\chi_i$)
Local chirality measures the geometric handedness of the polypeptide backbone. For each residue $i$, it is computed using a set of four consecutive $C_\alpha$ atoms to capture the local twist of the chain.

**Mean Local Chirality ($\langle \chi_i \rangle$):**
Captures the primary structural bias of the residue.  

$$\langle \chi_i \rangle = \frac{1}{T} \sum_{t=1}^{T} \left[ \mathbf{v}_{i-1,i} \cdot (\mathbf{v}_{i,i+1} \times \mathbf{v}_{i+1,i+2}) \right]$$

* **Interpretation:** Positive values indicate a right-handed bias (alpha-helical tendency), while negative values indicate a left-handed bias.

**Chirality Variance ($\sigma^2_{\chi, i}$):**
Quantifies the stability of the local handedness.

$$\sigma^2_{\chi, i} = \langle \chi_i^2 \rangle - \langle \chi_i \rangle^2$$

* **Interpretation:** High variance indicates a residue that frequently interconverts between different chiral states, a common feature in highly disordered IDP regions.

**Shape:** (L, 2)

---

### Solvent Accessible Surface Area ($SASA_i$)
SASA measures the area of a residue's surface that is accessible to a solvent probe (radius $\approx 1.4$ Å). In IDPs, it is a key indicator of the "compactness" of the ensemble and the exposure of hydrophobic patches.

**Mean SASA ($\langle \text{SASA}_i \rangle$):**

Indicates the average degree of exposure of residue $i$ across the conformational ensemble.  

$$ \langle \text{SASA}_i \rangle = \frac{1}{T} \sum_{t=1}^{T} \text{SASA}_i(t) $$


* **Interpretation:** High values suggest the residue is highly disordered and solvated; low values suggest the residue is frequently buried in transient hydrophobic clusters or "globule-like" states.

**SASA Variance ($\sigma^2_{\text{SASA}, i}$):**
Measures the frequency of "burial-exposure" transitions.

$$\sigma^2_{\text{SASA}, i} = \langle \text{SASA}_i^2 \rangle - \langle \text{SASA}_i \rangle^2$$

* **Interpretation:** High variance identifies residues that act as "gatekeepers," moving between the core of a transiently collapsed IDP and the solvent-exposed surface.

**Shape:** (L, 2)

---

## Size of properties to extract

| Property Name | Shape | Number of floats to store |
| :--- | :--- | :--- |
| End-to-End Distance ($R_{ee}$) | $(N,)$ | $N$ |
| Radius of Gyration ($R_g$) | $(N,)$ | $N$ |
| Asphericity ($b$) | $(N,)$ | $N$ |
| Prolateness ($S$) | $(N,)$ | $N$ |
| Scaling Exponent ($\nu$) | $(N,)$ | $N$ |
| Secondary Structure Propensity | $(L, 3)$ | $3L$ |
| Dihedral Entropy per Residue | $(L, 2)$ | $2L$ |
| Distance Fluctuation ($CP_{ij}$) | $(L, L)$ | $L^2$ |
| Motion Correlation (DCCM) - TO REMOVE | $(L, L)$ | $L^2$ |
| Contact Map Frequency | $(L, L)$ | $L^2$ |
| Local Chirality | $(L, 2)$ | $2L$ |
| Solvent Accessible Surface Area (SASA) | $(L, 2)$ | $2L$ |

Finally only mean and std were saved for properties define for each frame
