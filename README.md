# linboot
CUDA-C (Monte-Carlo) non-parametric bootstrap confidence intervals on the slope of a linear model

## Introduction/Basics
This software reads a comma-delimited input file of (x,y) pairs and computes percentile and BCa confidence intervals on the best-fit (BLUE) slope. It is fairly straight-forward to instead (or in addition) return the CIs for the intercept. Since CUDA does not allow dynamic memory allocation, the maximum length of the data are constrained to 100 points via a "#define" for easy adjustment.

## Motivation
This code was written in support of academic research by the publisher, who is a doctoral student in the Computational Science Research Center (CSRC), as part of a multidisciplinary project with the Viral Information Instute (VII). The CSRC and VII are both located at San Diego State University.

## Build Status
Builds! Runs! Only tested on GNU/Linux:
- Ubuntu 18.04 LTS: `uname -a` (kernel 4.15.0-39-generic)
- NVCC/toolkit: `nvcc --version` Cuda compilation tools, release 9.1, V9.1.85
- nVidia driver: `nvidia-smi` 396.54 
- GNU Scientific Library: `dpkg -l libgsl*`
  - libgsl-dev, version: 2.4+dfsg-6
  - libgsl23, version: 2.4+dfsg-6
  - libgslcblas0, version: 2.4+dfsg-6
  
## Compiling
A makefile is included.
```bash
make
```

## Running
```bash
./linboot <filename> <alpha-value>
```

## Output
The output includes a text file `<filename>-slope-CI.dat` which lists the results in a comma-separated list of the following:
```
lower_percentile,upper_percentile,lower_BCa,upper_BCa,SE_lower,SE_upper,median,mean,#pts,#bs_iterations
```
as well as a text file `<filename>-slope-histogram.dat` containing comma-separated histogram data, as
```
bin_RHS,bin_count
```

## Post-processing
A sample application `plot_hist.R` to plot the histogram is included. It is written in the `R` language, and depends on `ggplot2`.

## Test data
- Test data can be generated using `generate_test_data.R`.
- A test data file is included, called `testdata.csv`.

## Legal
License -- MIT. Use/Share away!

Disclaimers: 
1. Any/all libraries used have their own licenses/rights; be sure your use is compatible with these !
2. This content reflects the opinions of the content creator and is not meant to represent the CSRC, VII or SDSU.
