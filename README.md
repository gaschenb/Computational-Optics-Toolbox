# Computational Optics Toolbox

I started to build this toolbox a year or two ago to include computational optics techniques I see used most often, and used it to learn techniques as I went. It covers digital holographic microscopy, phase unwrapping and phase retrieval, tomography, computational microscopy, 3-D shape recovery, computational cameras, and photon-counting and quantum imaging.

Every technique has a physically based simulator behind it, so everything runs out of the box and gets scored against a Ground Truth. Most techniques also take your own measurements. All the methods are classical: physical forward models, closed-form inversions and iterative or variational solvers. There are no learned models.

Each technique is written once as an *experiment* (a simulator, one or more reconstruction methods, the equations behind them, and the metrics against the Ground Truth). The GUI, the command line and the Python API are all built from that one registry, so they always agree with each other.

![GUI](docs/figures/gui.png)

## Install and run

```bash
git clone https://github.com/gaschenb/Computational-Optics-Toolbox.git
cd Computational-Optics-Toolbox
pip install -e ".[gui]"
opticstoolbox-gui
```

You need Python 3.10 or newer. A few notes on what gets installed:

1. `pip install -e .` installs two commands: `opticstoolbox-gui` for the GUI and `opticstoolbox` for the command line.
2. The core only needs numpy, scipy, matplotlib and pillow. PyQt5 and pyqtgraph are pulled in by the `gui` extra.
3. `pip install -e ".[gui3d]"` adds PyOpenGL, which the volume viewer uses for its 3-D rendering tab.
4. `pip install -r requirements.txt` installs every optional package, including tifffile and h5py (16-bit and multi-page TIFF, HDF5, CXI and MATLAB v7.3 files) and PyOpenGL.

## Features

1. **Nine main areas, 40 experiments.**

   | Area | Experiments |
   |---|---|
   | Wave propagation | diffraction propagation, aberrated imaging (Zernike), Talbot self-imaging |
   | Digital holography | off-axis DHM, phase-shifting DHM, slightly off-axis DHM, Kramers–Kronig holography, in-line (Gabor) and multi-height holography, numerical autofocus, compressive holography |
   | Phase unwrapping | Itoh, Goldstein branch cuts, quality-guided, least squares (DCT), weighted least squares (PCG), PUMA (graph cuts) |
   | Phase retrieval | coherent diffraction imaging, phase retrieval from \|Ax\|², multi-plane / Gerchberg–Saxton, ptychography, Fourier ptychographic microscopy, transport of intensity, differential phase contrast |
   | Tomography | 2-D and 3-D projection tomography, optical diffraction tomography (3-D refractive index) |
   | Microscopy & sensing | deconvolution, structured illumination microscopy, Shack–Hartmann wavefront sensing, imaging through scattering, two-photon excitation microscopy |
   | 3-D shape | structure from focus / defocus, structure from polarisation, light-field refocusing and depth, fringe projection profilometry |
   | Computational cameras | computational ghost imaging, single-pixel imaging, lensless imaging (DiffuserCam), snapshot hyperspectral imaging (CASSI), coded exposure (flutter shutter), non-line-of-sight imaging |
   | Photon counting & quantum | photon-limited (Poisson) imaging, first-photon imaging, single-photon lidar, quantum (biphoton) ghost imaging |

2. **Ground Truth for everything simulated.** Each result shows the estimate next to its Ground Truth, an error map, and RMSE, PSNR and SSIM [5]. Anything that is only defined up to a global phase, a scale or a phase ramp is aligned before it is scored.
3. **One library of sample objects.** Every 2-D experiment can image the same objects: a USAF-1951 target with groups and elements in the standard layout [78], a Siemens star, text, a single letter, cells, blobs, fluorescent beads, the Shepp–Logan phantom [43] and a natural texture. The 3-D experiments add a 3-D Shepp–Logan, a cell with beads, or any 2-D object extruded into a sphere. CASSI, defocus and propagation use the same objects as material layouts, textures and apertures.
4. **Your own data.** Most experiments accept measurements from files.
   1. Images can be TIFF (including 16-bit and multi-page), PNG, JPEG or BMP. Arrays can be `.npy`, `.npz`, MATLAB `.mat` (v7.3 too), HDF5 or CXI. Tables can be CSV or TXT. Complex fields, such as ODT fields or a ptychography probe, come from `.npy`, `.npz` or `.mat`.
   2. RGB images are converted to grey. A stack can be a folder of images (in natural sort order), a multi-frame file or a list of files.
   3. Physical parameters go in a JSON file. A key can carry a unit suffix (`_nm`, `_um`, `_mm`, `_m`) and gets converted to micrometres, for example `{"wl_nm": 632.8, "dx_um": 3.45, "z_mm": 12.5, "na": 0.25}`.
   4. `opticstoolbox info <experiment>` lists what each experiment needs. Off-axis DHM wants a hologram, an optional blank, the wavelength, pixel size and defocus. Ptychography wants the diffraction patterns, the scan positions and optionally a probe. Fourier ptychography wants one image per LED with the LED geometry, NA and magnification. TIE wants a defocus stack and its spacing. ODT wants the complex sample and background fields. SPAD lidar wants a histogram cube with its bin and pulse widths.

## Methods

The numbers in brackets point to [Citations](#citations).

### Wave propagation

1. The propagators are the angular spectrum method, with and without the band limit of Matsushima and Shimobaba [2], the Fresnel transfer function and impulse response, Rayleigh–Sommerfeld convolution, single-FFT Fresnel, and Fraunhofer with a zoom DFT [1]. A sampling report gives the critical distance, the Fresnel number and the largest NA the grid samples.
2. Aberrated coherent and incoherent imaging uses Noll-ordered, RMS-normalised Zernike pupils [3], and reports the Strehl ratio and the MTF.
3. The Talbot carpet [4] is computed over one Talbot length.

### Digital holography

1. Off-axis DHM [10] finds the +1 order automatically with sub-pixel refinement, filters it with a Hann, Tukey or circular window, refocuses numerically, and compensates the phase aberrations with a polynomial fit, PCA or a blank reference hologram.
2. Phase-shifting DHM [11] uses N-step least squares, the advanced iterative algorithm for unknown shifts [12], or PCA demodulation [13].
3. Two-frame slightly off-axis holography [14] handles carriers too small for Fourier filtering.
4. Kramers–Kronig holography [15] puts the reference at the pupil edge and recovers the field from the intensity alone.
5. Gabor in-line holography [16] is done by back-propagation, iterative twin-image removal, or multi-height phase retrieval [17].
6. Autofocus compares four focus metrics: the Tamura coefficient [18], amplitude variance, Tenengrad and spectral ℓ1.
7. Compressive holography [19] recovers a multi-plane scene from one hologram with TV or ℓ1 regularisation, solved with FISTA [6].

### Phase unwrapping

1. The six algorithms are Itoh [20], Goldstein branch cuts [21], quality-guided, least squares by DCT and weighted least squares by PCG [22], and PUMA graph cuts [23].
2. Each run shows the residues and the quality map, checks congruence, and scores against the Ground Truth phase.

### Phase retrieval

1. Coherent diffraction imaging uses ER and HIO [25], the Difference Map [26], RAAR [27], ASR and OSS [29], with a fixed or shrink-wrap support [28]. Before scoring, the result is aligned to remove the twin and translation ambiguity.
2. Generic |Ax|² retrieval, with Gaussian or coded-diffraction measurements, uses Wirtinger flow [30], truncated [31] and reshaped Wirtinger flow, alternating minimisation, PhaseCut [33] and PhaseLift [32]. The last two are semidefinite programs, so keep those problems small.
3. Multi-plane Gerchberg–Saxton works from a defocus stack, next to the classical two-plane version [24].
4. Ptychography uses ePIE [34], rPIE and momentum-accelerated mPIE [35], with probe recovery and sub-pixel position correction.
5. Fourier ptychography [36] uses Gauss–Newton with embedded pupil recovery [37] or alternating projections, with optional LED position correction.
6. Transport of intensity [38, 39] is solved with FFT or DCT Poisson solvers for uniform intensity, and with Teague's method for non-uniform intensity.
7. Differential phase contrast [40] computes the weak-object transfer functions from the actual source and pupil, then inverts with Tikhonov (phase only, or phase and absorption) or with TV through ADMM [7].

### Tomography

1. 2-D and 3-D parallel-beam CT use an exactly matched sparse projector and back-projector pair. The reconstructions are FBP with Ram-Lak, Shepp-Logan [43], cosine, Hann and Hamming filters, SIRT [41], and ordered-subsets SART [42].
2. Optical diffraction tomography is simulated with multislice beam propagation, under illumination scanning or sample rotation. It is inverted with the Born or Rytov approximation [45] through the Fourier diffraction theorem [44], followed by Gerchberg–Papoulis positivity iterations that fill the missing cone [46].

### Microscopy and sensing

1. Deconvolution uses Wiener, Richardson–Lucy [47] (optionally with TV), TV-ADMM [7] and blind Richardson–Lucy [48].
2. 2-D structured illumination [49] estimates the pattern vectors, phases and modulation, then recombines with a generalised Wiener filter.
3. Shack–Hartmann wavefront sensing simulates wave-optical spot images, offers three centroiding methods, and reconstructs zonally (Southwell [50]) or modally (Zernike [3]).
4. Imaging through scattering uses speckle correlations within the memory effect [51, 52]. The forward model is physical and in µm: an exact spherical wave from each object point, an iris and a Gaussian-correlated phase screen, then band-limited, 2×-padded angular-spectrum propagation to the sensor [2]. The field is computed on a sub-pixel grid fine enough that |U|² is Nyquist sampled (dx ≤ λv/2D) and the phase screen is resolved, and the grid is big enough that the band limit passes every ray that reaches the sensor. The camera integrates over each pixel's active area and adds shot noise, read noise and ADC quantisation. Each run reports the sampling checks and the measured speckle grain and contrast. The object comes back from one phase-retrieval run (HIO with shrink-wrap and iterate averaging [25, 28], or OSS [29]) on the autocorrelation averaged over a few independent diffuser positions.
5. Two-photon microscopy [53] compares the 1P, confocal and 2P effective PSFs, the optical-sectioning curves and depth-dependent scattering, and deconvolves in 3-D with Richardson–Lucy [47].

### 3-D shape

1. Depth from focus [54] works from a focal stack with three focus measures, Gaussian peak interpolation and an all-in-focus composite.
2. Depth from defocus [55] uses the relative blur between two images.
3. Shape from polarisation [56] gets the Stokes parameters from three or more polariser angles, inverts the Fresnel degree of polarisation for diffuse or specular reflection, resolves the azimuth with a convexity prior or with shading, and integrates the normals with Frankot–Chellappa [57] or least squares.
4. Light fields [58] are refocused by shift-and-add, with depth from angular variance or refocus contrast.
5. Fringe projection uses Fourier-transform profilometry [59], N-step phase shifting with spatial unwrapping, or multi-frequency temporal unwrapping [60].

### Computational cameras

1. Ghost imaging [61] uses pseudo-thermal speckle or random DMD patterns, and reconstructs with classical, differential [62] and normalised correlation, a pseudo-inverse, or compressive TV.
2. Single-pixel imaging [63] uses differential Hadamard patterns in natural, sequency or cake-cutting order, or four-step Fourier patterns [64], with linear and TV reconstructions.
3. DiffuserCam [65] is solved with ADMM [7] using an exact crop model, TV and non-negativity, with FISTA [6] and a naive Wiener filter for comparison.
4. CASSI [66] uses a single-disperser forward model with its exact adjoint, GAP-TV [67] and ADMM-TV, and renders the cube to RGB through the CIE colour-matching functions.
5. The flutter shutter [68] compares optimised, random and box shutter codes through their MTFs.
6. Confocal non-line-of-sight imaging has a transient simulator and reconstructs with the light-cone transform [70], f–k migration [71] and filtered back-projection [69].

### Photon counting and quantum

1. Poisson imaging is denoised with Gaussian smoothing, Anscombe [72] plus TV [8], or Poisson-likelihood TV solved by Chambolle–Pock [9], optionally with deblurring.
2. First-photon imaging [73] starts from a per-pixel maximum-likelihood estimate, then uses log-domain TV for reflectivity and censoring plus TV for depth.
3. SPAD lidar simulates TCSPC histograms including first-photon pile-up, applies the Coates correction [75], and estimates depth from the histogram peak, a log-matched filter, or censored ML plus TV [74].
4. Quantum ghost imaging [76] runs a Monte Carlo of SPDC photon pairs with a bucket detector and a photon-counting camera, forms heralded coincidences with accidental subtraction and the coincidence-to-accidental ratio, and compares the result with the unheralded camera image [77].

## Results

Every figure below comes from `python -m opticstoolbox.examples.gallery`, which runs each experiment with its default scene and method and writes the result to `docs/figures`.

| | | |
|---|---|---|
| ![](docs/figures/offaxis.png) Off-axis DHM | ![](docs/figures/inline.png) Multi-height in-line holography | ![](docs/figures/phase_shifting.png) Phase-shifting DHM |
| ![](docs/figures/kk.png) Kramers–Kronig holography | ![](docs/figures/compressive_holo.png) Compressive holography | ![](docs/figures/unwrap.png) Phase unwrapping |
| ![](docs/figures/cdi.png) Coherent diffraction imaging | ![](docs/figures/ptychography.png) Ptychography | ![](docs/figures/fpm.png) Fourier ptychography |
| ![](docs/figures/tie.png) Transport of intensity | ![](docs/figures/dpc.png) Differential phase contrast | ![](docs/figures/ct.png) 2-D tomography |
| ![](docs/figures/ct3d.png) 3-D tomography | ![](docs/figures/odt.png) Optical diffraction tomography | ![](docs/figures/sim.png) Structured illumination |
| ![](docs/figures/deconvolution.png) Deconvolution | ![](docs/figures/shack_hartmann.png) Shack–Hartmann | ![](docs/figures/speckle.png) Imaging through scattering |
| ![](docs/figures/two_photon.png) Two-photon microscopy | ![](docs/figures/defocus.png) Structure from defocus | ![](docs/figures/polarization.png) Structure from polarisation |
| ![](docs/figures/lightfield.png) Light fields | ![](docs/figures/fringe.png) Fringe projection | ![](docs/figures/ghost.png) Ghost imaging |
| ![](docs/figures/single_pixel.png) Single-pixel imaging | ![](docs/figures/diffusercam.png) DiffuserCam | ![](docs/figures/cassi.png) CASSI |
| ![](docs/figures/coded_exposure.png) Flutter shutter | ![](docs/figures/nlos.png) Non-line-of-sight imaging | ![](docs/figures/photon_limited.png) Photon-limited imaging |
| ![](docs/figures/first_photon.png) First-photon imaging | ![](docs/figures/spad_lidar.png) SPAD lidar | ![](docs/figures/biphoton.png) Quantum ghost imaging |
| ![](docs/figures/propagation.png) Diffraction propagation | ![](docs/figures/psf.png) Aberrated imaging | ![](docs/figures/talbot.png) Talbot carpet |
| ![](docs/figures/autofocus.png) Autofocus | ![](docs/figures/multiplane.png) Multi-plane phase retrieval | ![](docs/figures/generic_pr.png) Wirtinger flow |
| ![](docs/figures/slightly_offaxis.png) Slightly off-axis DHM | | |

## GUI

Start it with `opticstoolbox-gui`, or `python -m opticstoolbox.gui` with `--dark` or `--light`.

1. **Picking an experiment.** Choose an area in the sidebar and an experiment from the list at the top of the left column. Its description and governing equations show up underneath. The equations are drawn from vector outlines inside the program, so you don't need LaTeX.
2. **Data.** *Simulated* lets you edit the scene and the physical parameters. *My data* gives you one file picker per input (with folder or multi-file selection for stacks) and a form for the physical parameters, which you can fill from a JSON file.
3. **Reconstruction.** Pick the method and its parameters. Method-specific equations appear where they apply.
4. **Images.** Every image is fitted to its panel with physical axes (µm, NA, time, wavelength, view index and so on) and a colour bar. Each plot is shaped to its image, so images of different sizes and aspect ratios all fill their panels. You can zoom in as far as you like but not out past the full image. Zoom and pan are linked across panels with the same field of view, a double-click or *Fit all* refits, and a read-out follows the cursor. When an image is drawn smaller than its pixel count, pixels are averaged rather than skipped, so fine fringes don't alias into moiré.
5. **Volumes.** Every volume and depth stack gets a slice slider, and *Link slice sliders* moves the reconstruction, Ground Truth and error together. *3-D view* opens the volume viewer, with orthogonal x–y, x–z and y–z slices through a shared cursor (click to move it), maximum-intensity or mean projections, one row per volume so the estimate and Ground Truth stay aligned, and a 3-D rendering tab when PyOpenGL is installed.
6. **3-D surfaces.** Every reconstructed amplitude, phase and height map in the holography experiments has a *3-D surface* button. It opens a rotatable surface next to the matching Ground Truth (or the companion amplitude or phase). The two views rotate together, and you can share the z range, change the vertical scale, and unwrap wrapped phase for display.
7. **Curves, log and metrics.** Convergence, profiles and spectra are under *Curves*, and every run fills the metrics table.
8. **Auto update.** Every edit re-runs the experiment on a background thread, so the interface never freezes. Runs are debounced, and a running job is cancelled and restarted. The slower experiments start with auto update off. *Run* (F5) and *Stop* (Esc) always work.
9. **Export.** *Export result* writes every panel as `.npy`, a JSON summary of metrics and curves, and a PNG overview. *Save image* saves the image grid as it looks on screen.
10. **Theme.** Light and dark themes (Ctrl+T). The choice is remembered.

## CLI

```bash
opticstoolbox list                                   # all experiments by area
opticstoolbox info ptychography                      # description, equations, parameters, inputs
opticstoolbox run sim --method "Wiener SIM (estimated parameters)" --out results --png
opticstoolbox run ct --method OS-SART --set angles=45 range=120 --solver iters=40
opticstoolbox run offaxis --input hologram=holo.tif blank=blank.tif --meta setup.json --out results
```

`python -m opticstoolbox` works in place of `opticstoolbox`. The options of `run` are:

1. `--set key=value` sets scene or physical parameters.
2. `--solver key=value` sets solver parameters.
3. `--input key=path` loads your data. A stack can be a folder, a multi-frame file, or several files joined with `;`.
4. `--meta file.json` gives the physical parameters for your data.
5. `--out` writes the panels and a summary to a folder, and `--png` adds a PNG overview.
6. `--json` prints the metrics as JSON.

## Library use

```python
import opticstoolbox as ot

e = ot.get("offaxis")
data = e.make_data({"defocus": 5.0, "object": "usaf"})            # simulated example
res = e.run(data, {"compensation": "reference hologram"})
print(res.metrics)                                                # RMSE, PSNR, SSIM, ...
phase = res.panels["unwrapped phase"].data                        # numpy array

# your own data, same experiment, same solvers
from opticstoolbox.core import io
arrays = {"hologram": io.read_input("holo.tif"),                  # off-axis hologram
          "blank": io.read_input("blank.tif")}                    # optional object-free hologram
data = e.make_user_data(arrays, {"dx": 0.2, "wl": 0.532, "defocus": 0.0})
res = e.run(data)
```

The numerical pieces can also be used on their own. A few I reach for often:

1. `opticstoolbox.core.propagation.angular_spectrum`
2. `opticstoolbox.core.holography.offaxis_filter`
3. `opticstoolbox.core.phaseretrieval.cdi`
4. `opticstoolbox.core.ptychography.ptycho`
5. `opticstoolbox.core.unwrap.unwrap`
6. `opticstoolbox.core.tomography.odt_reconstruct`
7. `opticstoolbox.core.nlos.lct`

## Layout

```
opticstoolbox/
  core/          numerics: propagation, waveoptics, holography, unwrap, phaseretrieval, ptychography,
                 intensity (TIE/DPC), tomography, deconv, sim, wavefront, depth, scattering, ghost,
                 cassi, nlos, photon, regularizers, fields (phantoms, metrics), io, registry
  experiments/   one module per area; each registers its experiments
  gui/           PyQt5 / pyqtgraph application
  examples/      gallery script
  cli.py, render.py
docs/figures/    gallery images
```

## Citations

**General and wave propagation**

1. J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed. (Roberts & Company, 2005).
2. K. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Opt. Express* **17**, 19662 (2009).
3. R. J. Noll, "Zernike polynomials and atmospheric turbulence," *J. Opt. Soc. Am.* **66**, 207 (1976).
4. H. F. Talbot, "Facts relating to optical science. No. IV," *Phil. Mag.* **9**, 401 (1836).
5. Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Trans. Image Process.* **13**, 600 (2004).
6. A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear inverse problems," *SIAM J. Imaging Sci.* **2**, 183 (2009).
7. S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein, "Distributed optimization and statistical learning via the alternating direction method of multipliers," *Found. Trends Mach. Learn.* **3**, 1 (2011).
8. A. Chambolle, "An algorithm for total variation minimization and applications," *J. Math. Imaging Vis.* **20**, 89 (2004).
9. A. Chambolle and T. Pock, "A first-order primal-dual algorithm for convex problems with applications to imaging," *J. Math. Imaging Vis.* **40**, 120 (2011).

**Digital holography**

10. E. Cuche, P. Marquet and C. Depeursinge, "Spatial filtering for zero-order and twin-image elimination in digital off-axis holography," *Appl. Opt.* **39**, 4070 (2000).
11. I. Yamaguchi and T. Zhang, "Phase-shifting digital holography," *Opt. Lett.* **22**, 1268 (1997).
12. Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Opt. Lett.* **29**, 1671 (2004).
13. J. Vargas, J. A. Quiroga and T. Belenguer, "Phase-shifting interferometry based on principal component analysis," *Opt. Lett.* **36**, 1326 (2011).
14. N. T. Shaked, Y. Zhu, M. T. Rinehart and A. Wax, "Two-step-only phase-shifting interferometry with optimized detector bandwidth for microscopy of live cells," *Opt. Express* **17**, 15585 (2009).
15. Y. Baek, K. Lee, S. Shin and Y. Park, "Kramers–Kronig holographic imaging for high-space-bandwidth product," *Optica* **6**, 45 (2019).
16. D. Gabor, "A new microscopic principle," *Nature* **161**, 777 (1948).
17. A. Greenbaum and A. Ozcan, "Maskless imaging of dense samples using pixel super-resolution based multi-height lensfree on-chip microscopy," *Opt. Express* **20**, 3129 (2012).
18. P. Memmolo, C. Distante, M. Paturzo, A. Finizio, P. Ferraro and B. Javidi, "Automatic focusing in digital holography and its application to stretched holograms," *Opt. Lett.* **36**, 1945 (2011).
19. D. J. Brady, K. Choi, D. L. Marks, R. Horisaki and S. Lim, "Compressive holography," *Opt. Express* **17**, 13040 (2009).

**Phase unwrapping**

20. K. Itoh, "Analysis of the phase unwrapping algorithm," *Appl. Opt.* **21**, 2470 (1982).
21. R. M. Goldstein, H. A. Zebker and C. L. Werner, "Satellite radar interferometry: two-dimensional phase unwrapping," *Radio Sci.* **23**, 713 (1988).
22. D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software* (Wiley, 1998).
23. J. M. Bioucas-Dias and G. Valadão, "Phase unwrapping via graph cuts," *IEEE Trans. Image Process.* **16**, 698 (2007).

**Phase retrieval**

24. R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the determination of phase from image and diffraction plane pictures," *Optik* **35**, 237 (1972).
25. J. R. Fienup, "Phase retrieval algorithms: a comparison," *Appl. Opt.* **21**, 2758 (1982).
26. V. Elser, "Phase retrieval by iterated projections," *J. Opt. Soc. Am. A* **20**, 40 (2003).
27. D. R. Luke, "Relaxed averaged alternating reflections for diffraction imaging," *Inverse Probl.* **21**, 37 (2005).
28. S. Marchesini, H. He, H. N. Chapman, S. P. Hau-Riege, A. Noy, M. R. Howells, U. Weierstall and J. C. H. Spence, "X-ray image reconstruction from a diffraction pattern alone," *Phys. Rev. B* **68**, 140101 (2003).
29. J. A. Rodriguez, R. Xu, C.-C. Chen, Y. Zou and J. Miao, "Oversampling smoothness: an effective algorithm for phase retrieval of noisy diffraction intensities," *J. Appl. Cryst.* **46**, 312 (2013).
30. E. J. Candès, X. Li and M. Soltanolkotabi, "Phase retrieval via Wirtinger flow: theory and algorithms," *IEEE Trans. Inf. Theory* **61**, 1985 (2015).
31. Y. Chen and E. J. Candès, "Solving random quadratic systems of equations is nearly as easy as solving linear systems," *Commun. Pure Appl. Math.* **70**, 822 (2017).
32. E. J. Candès, T. Strohmer and V. Voroninski, "PhaseLift: exact and stable signal recovery from magnitude measurements via convex programming," *Commun. Pure Appl. Math.* **66**, 1241 (2013).
33. I. Waldspurger, A. d'Aspremont and S. Mallat, "Phase recovery, MaxCut and complex semidefinite programming," *Math. Program.* **149**, 47 (2015).
34. A. M. Maiden and J. M. Rodenburg, "An improved ptychographical phase retrieval algorithm for diffractive imaging," *Ultramicroscopy* **109**, 1256 (2009).
35. A. Maiden, D. Johnson and P. Li, "Further improvements to the ptychographical iterative engine," *Optica* **4**, 736 (2017).
36. G. Zheng, R. Horstmeyer and C. Yang, "Wide-field, high-resolution Fourier ptychographic microscopy," *Nat. Photonics* **7**, 739 (2013).
37. X. Ou, G. Zheng and C. Yang, "Embedded pupil function recovery for Fourier ptychographic microscopy," *Opt. Express* **22**, 4960 (2014).
38. M. R. Teague, "Deterministic phase retrieval: a Green's function solution," *J. Opt. Soc. Am.* **73**, 1434 (1983).
39. C. Zuo, J. Li, J. Sun, Y. Fan, J. Zhang, L. Lu, R. Zhang, B. Wang, L. Huang and Q. Chen, "Transport of intensity equation: a tutorial," *Opt. Lasers Eng.* **135**, 106187 (2020).
40. L. Tian and L. Waller, "Quantitative differential phase contrast imaging in an LED array microscope," *Opt. Express* **23**, 11394 (2015).

**Tomography**

41. A. C. Kak and M. Slaney, *Principles of Computerized Tomographic Imaging* (IEEE Press, 1988).
42. A. H. Andersen and A. C. Kak, "Simultaneous algebraic reconstruction technique (SART): a superior implementation of the ART algorithm," *Ultrason. Imaging* **6**, 81 (1984).
43. L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head section," *IEEE Trans. Nucl. Sci.* **21**, 21 (1974).
44. E. Wolf, "Three-dimensional structure determination of semi-transparent objects from holographic data," *Opt. Commun.* **1**, 153 (1969).
45. A. J. Devaney, "Inverse-scattering theory within the Rytov approximation," *Opt. Lett.* **6**, 374 (1981).
46. Y. Sung, W. Choi, C. Fang-Yen, K. Badizadegan, R. R. Dasari and M. S. Feld, "Optical diffraction tomography for high resolution live cell imaging," *Opt. Express* **17**, 266 (2009).

**Microscopy and sensing**

47. W. H. Richardson, "Bayesian-based iterative method of image restoration," *J. Opt. Soc. Am.* **62**, 55 (1972); L. B. Lucy, "An iterative technique for the rectification of observed distributions," *Astron. J.* **79**, 745 (1974).
48. D. A. Fish, A. M. Brinicombe, E. R. Pike and J. G. Walker, "Blind deconvolution by means of the Richardson–Lucy algorithm," *J. Opt. Soc. Am. A* **12**, 58 (1995).
49. M. G. L. Gustafsson, "Surpassing the lateral resolution limit by a factor of two using structured illumination microscopy," *J. Microsc.* **198**, 82 (2000).
50. W. H. Southwell, "Wave-front estimation from wave-front slope measurements," *J. Opt. Soc. Am.* **70**, 998 (1980).
51. J. Bertolotti, E. G. van Putten, C. Blum, A. Lagendijk, W. L. Vos and A. P. Mosk, "Non-invasive imaging through opaque scattering layers," *Nature* **491**, 232 (2012).
52. O. Katz, P. Heidmann, M. Fink and S. Gigan, "Non-invasive single-shot imaging through scattering layers and around corners via speckle correlations," *Nat. Photonics* **8**, 784 (2014).
53. W. Denk, J. H. Strickler and W. W. Webb, "Two-photon laser scanning fluorescence microscopy," *Science* **248**, 73 (1990).

**3-D shape**

54. S. K. Nayar and Y. Nakagawa, "Shape from focus," *IEEE Trans. Pattern Anal. Mach. Intell.* **16**, 824 (1994).
55. M. Subbarao and G. Surya, "Depth from defocus: a spatial domain approach," *Int. J. Comput. Vis.* **13**, 271 (1994).
56. G. A. Atkinson and E. R. Hancock, "Recovery of surface orientation from diffuse polarization," *IEEE Trans. Image Process.* **15**, 1653 (2006).
57. R. T. Frankot and R. Chellappa, "A method for enforcing integrability in shape from shading algorithms," *IEEE Trans. Pattern Anal. Mach. Intell.* **10**, 439 (1988).
58. R. Ng, M. Levoy, M. Brédif, G. Duval, M. Horowitz and P. Hanrahan, "Light field photography with a hand-held plenoptic camera," Stanford Tech. Rep. CTSR 2005-02 (2005).
59. M. Takeda, H. Ina and S. Kobayashi, "Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry," *J. Opt. Soc. Am.* **72**, 156 (1982).
60. C. Zuo, L. Huang, M. Zhang, Q. Chen and A. Asundi, "Temporal phase unwrapping algorithms for fringe projection profilometry: a comparative review," *Opt. Lasers Eng.* **85**, 84 (2016).

**Computational cameras**

61. J. H. Shapiro, "Computational ghost imaging," *Phys. Rev. A* **78**, 061802 (2008).
62. F. Ferri, D. Magatti, L. A. Lugiato and A. Gatti, "Differential ghost imaging," *Phys. Rev. Lett.* **104**, 253603 (2010).
63. M. F. Duarte, M. A. Davenport, D. Takhar, J. N. Laska, T. Sun, K. F. Kelly and R. G. Baraniuk, "Single-pixel imaging via compressive sampling," *IEEE Signal Process. Mag.* **25**, 83 (2008).
64. Z. Zhang, X. Ma and J. Zhong, "Single-pixel imaging by means of Fourier spectrum acquisition," *Nat. Commun.* **6**, 6225 (2015).
65. N. Antipa, G. Kuo, R. Heckel, B. Mildenhall, E. Bostan, R. Ng and L. Waller, "DiffuserCam: lensless single-exposure 3D imaging," *Optica* **5**, 1 (2018).
66. A. Wagadarikar, R. John, R. Willett and D. Brady, "Single disperser design for coded aperture snapshot spectral imaging," *Appl. Opt.* **47**, B44 (2008).
67. X. Yuan, "Generalized alternating projection based total variation minimization for compressive sensing," *Proc. IEEE ICIP*, 2539 (2016).
68. R. Raskar, A. Agrawal and J. Tumblin, "Coded exposure photography: motion deblurring using fluttered shutter," *ACM Trans. Graph.* **25**, 795 (2006).
69. A. Velten, T. Willwacher, O. Gupta, A. Veeraraghavan, M. G. Bawendi and R. Raskar, "Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging," *Nat. Commun.* **3**, 745 (2012).
70. M. O'Toole, D. B. Lindell and G. Wetzstein, "Confocal non-line-of-sight imaging based on the light-cone transform," *Nature* **555**, 338 (2018).
71. D. B. Lindell, G. Wetzstein and M. O'Toole, "Wave-based non-line-of-sight imaging using fast f–k migration," *ACM Trans. Graph.* **38**, 116 (2019).

**Photon counting and quantum imaging**

72. F. J. Anscombe, "The transformation of Poisson, binomial and negative-binomial data," *Biometrika* **35**, 246 (1948).
73. A. Kirmani, D. Venkatraman, D. Shin, A. Colaço, F. N. C. Wong, J. H. Shapiro and V. K. Goyal, "First-photon imaging," *Science* **343**, 58 (2014).
74. D. Shin, A. Kirmani, V. K. Goyal and J. H. Shapiro, "Photon-efficient computational 3-D and reflectivity imaging with single-photon detectors," *IEEE Trans. Comput. Imaging* **1**, 112 (2015).
75. P. B. Coates, "The correction for photon 'pile-up' in the measurement of radiative lifetimes," *J. Phys. E* **1**, 878 (1968).
76. T. B. Pittman, Y. H. Shih, D. V. Strekalov and A. V. Sergienko, "Optical imaging by means of two-photon quantum entanglement," *Phys. Rev. A* **52**, R3429 (1995).
77. P. A. Morris, R. S. Aspden, J. E. C. Bell, R. W. Boyd and M. J. Padgett, "Imaging with a small number of photons," *Nat. Commun.* **6**, 5913 (2015).

**Test targets**

78. MIL-STD-150A, *Photographic Lenses* (U.S. Department of Defense, 1959).
