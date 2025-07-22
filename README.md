<h2 align="center">About Tempus</h2>

Tempus is a project aimed at utilizing the cutting edge methods to apply non-linear regression problems. In it's core Tempus uses an advanced SVM implementation that fixes several inconsistencies of the original theory by prof. Vapnik and Chervonenkis to achieve forecast precision better than any other algorithm currently available in the field of statistical learning. This is accomplished by calculating the ideal kernel matrix for the given observations (labels) and then produce a kernel function fitted as tight as possible to it in order to extract all relevant information contained in the dataset. Having the ideal kernel matrix available for any given data allows this SVM implementation to nest several support vector machines into each other as kernel functions, or having a different statistical model produce the kernel function (eg. gradient boosted trees using <a href="https://github.com/microsoft/LightGBM">LightGBM</a> or a temporal fusion transformer using <a href="https://pytorch.org/">Torch</a>). Also, several conventional kernel methods are also implemented as part of Tempus, such are a fast variant of the <a href="https://www.scitepress.org/PublishedPapers/2013/42673/">Path Kernel (Baisero et al.)</a>, the Radial-basis function, the <a href="https://marcocuturi.net/GA.html">Global Alignment Kernel (Cuturi et al.)</a>. For this implementation there is no need to tune the regularization cost parameter, epsilon thresholding or do tuning of hyperparameters using a train-predict validation cycle. The model produced is optimal for the available data and hardware resources configured. Beside nesting of kernel functions, three other methods of scaling (or ensembling) are provided; in the spectral domain using a modified online empirical model decomposition, in the time domain or time slicing and sequential residuals or gradient boosting (the last one is work in progress). Each SVM model can have multiple weight layers according to the complexity of the trained data. Weights are produced by a standard non-linear Ax=b matrix solver - Pruned <a href="https://github.com/avaneev/biteopt">BiteOpt</a> which is based on the CMA-ES algorithm, but interfaces to <a href="https://petsc.org/release">PETSc</a>, <a href="https://github.com/icl-utk-edu/magma">MAGMA</a>, <a href="https://developer.nvidia.com/cusolver">CUSolver</a> and pruned <a href="https://github.com/libprima">PRIMA</a> are also available. Feature alignment and outlier filtering of the input data is implemented using the <a href="https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-summary-statistics-notes/2021-1/using-the-bacon-algorithm-for-outlier-detection.html">BACON</a> algorithm, <a href="https://github.com/ojmakhura/hdbscan">HDBScan</a> and simple Euclidean distance works good enough. This allows Tempus to efficiently model extremely noisy data.  
An online learning mode is in the works.

Tempus is a project that unites researchers and engineers from several countries and academic institutions in order to exchange ideas, learn and achieve the best performance in time series analysis using the latest methods in statistical analysis and signal processing. It makes use of HPC technologies such is OpenCL, Cilk Plus, CUDA and MPI to deliver maximum performance over highly scaled systems. Tempus can scale on many GPUs and and CPUs increasing precision and almost linear increase in performance in data processing.

It consists of multiple modules (libraries):

1. SVRMain (Main app for configuring and running)
2. SVRBusiness (the Service layer for managing domain model objects)
3. SVRPersist (the persistence layer for storing and reading persisted objects)
4. SVRCommon (the common (shared) functionality accross modules)
5. SVRModel (the domain model classes)

To build the app you may run make all while in SVRMain/Debug

To run the app you need to:

1. configure PostgreSQL database (connection string is configurable in SVRMain/include/main-config.hpp)
2. If you are linking dynamically, add the SVRBusiness, SVRCommon and SVRPersist libraries to LD_LIBRARY_PATH:
		- LD_LIBRARY_PATH=~/git/master/SVRWorkspace/SVRCommon/Debug:~/git/master/SVRWorkspace/SVRBusiness/Debug:~/git/master/SVRWorkspace/SVRPersist/Debug
		- export LD_LIBRARY_PATH

Note:
	- Change the value of *SQL_PROPERTIES_LOCATION* config variable according to the location on your disk (which is located in SVRWorkspace/SVRCommon/include/config/common.hpp)


<h2 align="center">History</h2>

The first implementation of Tempus started as a spin off of my University diploma work on the subject of scaling support vectors machine using the Wavelet transform in 2011. I pushlished the intial implementation as an open source project on sourceforge.net, main contributors being me and first employee Viktor. Later on I closed the project and secured financing from the Papakaya company in the order of 1.2 million euros, including my own means. This allowed me to hire many professional contributors and consult known experts on the subject, engineers and professors. The final goal being prediction of financial indexes or indicators. Tempus can, even with a modest GPU server (24 TFLOPS) produce positive forecast alpha when applied to the XAUUSD index.

Thanks to everyone involved at the Bulgarian Academy of Sciences permitting access to their super computer,
Ben Gurion University at Ber-Sheva for another computing server, students at FINKI Skopje and Taras Shevchenko University.

<h3>Authors:</h3>

- me - Lead programmer, design, testing, implementation and optimizing.
- prof Emanouil Atanasov - Design and implementation of chunking MIMO SVR, IRWLS hybrid direct solver, Online VMD, fast Path kernel approximation, hyperparameter optimization, CUDA and OpenCL parallelization
- Andrey Bezrukov - Project architecture and skeleton, optimization and parallelization
- Evgeniy Marinov - Actually accurate Online SVR
- Bojko Perfanov - SVR Epsilon and Cost path computation, (almost) boundless Wavelet decomposition à trous
- Sergej Kondratiuk - SMO and Online SVR pilot implementation, cascaded SVM
- Viktor Gjorgjievski - Initial project architecture design and implementation
- Taras Maliarcuk - Infrastructure and project architecture
- Guy Tal - Database and multithreading, optimization
- Stilyan Stojanov -  Implementation of the MIMO SVR, EVMD, kernels,
- Petar Simov - SVR and epsilon SVR path, infrastructure
- Dimitar Conov - GA kernel
- Stanislav Georgiev - Online SMO SVR, Wavelet decomposition
- Dimitar Slavchev - OpenCL kernels
- Vladimir Hizanov - Parameter tuning

Thanks to:
- Kristina Eskenazi - PR and connections
- Petar Ivanov - Web interface code
- George Kour - ML consulting
- prof Jihad El-Sana - Advice and support
- prof Dejan Gjorgjevik - Consultancy and support, referring students at FINKI Skopje
- prof William Cohen - for the ADA boost code
- Oleg Gumbar, Milen Hristov, Aleksandar Miladinov, Ali Kasmu - Sysadmins in Papakaya
- Vasil Savuliak - for referring people to the office in Kiev


<h2>BSD 3-Clause License</h2>

Copyright (c) 2025, Žarko Asen

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
