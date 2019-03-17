# High Performance R

_Instructions and benchmarks for high-performance computing in R_

## Tests

Test OpenMP:

``` {.sh}
R CMD SHLIB omp_sample.c
(echo 'dyn.load("omp_sample.so"); .Call("dumbCumsum",runif(100000),0L)') | time R --slave --no-save
taskset -p 0xffffffff `pidof R`
```

Run R benchmarks:

``` {.sh}
(echo 'library(MASS);set.seed(1)' && cat MASS-ex.R) | time R --slave --no-save
cat R-benchmark-25.R | time R --slave --no-save
cat bench.R | time R --slave --no-save
```

Matrix multiplication:

``` {.R}
a = matrix(rnorm(5000*5000), 5000, 5000)
b = matrix(rnorm(5000*5000), 5000, 5000)
c = a%*%b
```

Parallel backends:

`doMC` creates forks in the background:

``` {.R}
library(doMC)
registerDoMC()
x <- iris[which(iris[,5]!='setosa'),c(1,5)]
trials<- 10000
r <- foreach(icount(trials), .combine=cbind) %dopar% {
  ind <- sample(100,100,replace=T)
  result1 <- glm(x[ind,2]~x[ind,1],family=binomial(logit))
  coefficients(result1)
}
doParallel::stopImplicitCluster()
```

`doParallel` uses `multicore` in Linux and `snow` in Windows:

``` {.R}
library(foreach)
library(doParallel)
registerDoParallel()
set.seed(100)
foreach (i = 1:4) %do% {
  a = replicate(10000, rnorm(10000))
  d = determinant(a)
}
doParallel::stopImplicitCluster()
```


## Results (R-Benchmark-25)


### Server - 2 x Xeon E5-2640 - 12 cores (HT)

| Version   | Options             | Total time | Overall mean |
| --------- | ------------------- | ---------- | ------------ |
| R 3.2.2   | vanilla             | 39.307     | 1.263        |
| R 3.2.2   | OpenBlas            | 8.978      | 0.580        |
| R 3.2.2   | OpenBlas-OpenMP     | 8.881      | 0.433        |
| R 3.2.2   | OpenBlas-Pthreads   | 7.911      | 0.431        |
| MRO 3.2.2 | RevoMath            | 6.723      | 0.384        |


### Server - 2 x Xeon E5-2670 - 16 cores (HT) - 16 cores in LXC

| Version   | Options      | Total time | Overall mean |
| --------- | ------------ | ---------- | ------------ |
| R 3.4.3   | Vanilla      | 35.196     | 1.072        |
| R 3.4.3   | ATLAS        | 11.587     | 0.661        |
| R 3.4.3   | OpenBLAS     | 35.843     | 0.746        |
| R 3.4.3   | OpenBLAS 2c  | 9.250      | 0.537        |
| R 3.4.3   | OpenBLAS 8c  | 6.335      | 0.353        |
| R 3.4.3   | OpenBLAS 12c | 12.599     | 0.589        |
| R 3.4.3   | Intel MKL    | 10.499     | 0.560        |
| MRO 3.5.1 | Intel MKL    | 6.222      | 0.349        |


### Laptop DV6 - i5-2450M - 2 cores (HT)

| Version   | Options           | Total time | Overall mean |
| ----------| ----------------- | ---------- | ------------ |
| R 3.3.2   | OpenBlas-LAPACK   | 10.371     | 0.480        |
| MRO 3.3.1 | RevoMath          | 7.312      | 0.465        |


### Laptop T420 - i7-2640M - 2 cores (HT)

| Version   | Options                | Total time | Overall mean |
| --------- | ---------------------- | ---------- | ------------ |
| R 3.2.1   | OpenBlas-LAPACK        | 6.844      | 0.438        |
| R 3.2.2   | Intel MKL 2016U1 (icc) | 6.209      | 0.397        |
| MRO 3.2.2 | RevoMath               | 6.481      | 0.411        |
| R 3.2.3   | vanilla                | 34.812     | 1.128        |
| R 3.2.3   | OpenBlas-LAPACK        | 7.421      | 0.452        |
| R 3.2.3   | Intel MKL 2016U1 (icc) | 6.072      | 0.387        |
| R 3.2.3   | Intel MKL 2016U1 (gcc) | 6.309      | 0.405        |
| R 3.2.3   | Intel MKL 2016U2 (icc) | 5.851      | 0.379        |
| R 3.2.4   | Intel MKL 2016U2 (gcc) | 5.816      | 0.383        |
| R 3.2.4   | Intel MKL 2016U2 (icc) | 6.550      | 0.413        |
| R 3.3.0   | Intel MKL 2016U3 (gcc) | 5.893      | 0.387        |
| MRO 3.3.1 | RevoMath               | 6.529      | 0.410        |
| R 3.3.2   | Intel MKL 2017U1 (icc) | 6.594      | 0.411        |
| R 3.3.2   | Intel MKL 2017U1 (icc) | 6.332      | 0.402        |
| R 3.3.2   | Intel MKL 2017U1 (icc) | 6.348      | 0.406        |
| MRO 3.4.2 | Intel MKL              | 6.638      | 0.417        |
| MRO 3.4.2 | Intel MKL + kpti       | 6.516      | 0.413        |
| MRO 3.4.2 | Intel MKL + nopti      | 6.621      | 0.417        |


### Laptop - Elitebook 840 G3 - i7-6500U - 2 cores (HT)

| Version   | Options                         | Total time | Overall mean |
| --------- | ------------------------------- | ---------- | ------------ |
| R 3.3.2   | Intel MKL 2017U1 (icc)          | 4.628      | 0.281        |
| R 3.3.2   | Intel MKL 2017U1 (gcc)          | 4.485      | 0.274        |
| MRO 3.4.2 | Intel MKL + kpti                | 5.207      | 0.311        |
| MRO 3.4.2 | Intel MKL + nopti               | 5.160      | 0.306        |
| MRO 3.5.1 | Intel MKL                       | 4.776      | 0.280        |


### Desktop - i7-2600K - 4 cores (HT)

| Version   | Options                                     | Total time | Overall mean |
| --------- | ------------------------------------------- | ---------- | ------------ |
| R 3.3.1   | Intel MKL 2016U3 (gcc)                      | 4.822      | 0.297        |
| R 3.3.1   | Intel MKL 2017 (icc, no fp)                 | 4.706      | 0.281        |
| R 3.3.1   | Intel MKL 2017 (icc, prec-div, prec-sqrt)   | 4.829      | 0.288        |
| R 3.3.1   | Intel MKL 2017 (icc, fp precise source)     | 4.721      | 0.287        |
| MRO 3.3.1 | RevoMath                                    | 4.974      | 0.294        |
| R 3.3.2   | Intel MKL 2017 (icc, no parallel flags)     | 4.663      | 0.280        |
| R 3.3.2   | Intel MKL 2017U1 (icc)                      | 4.834      | 0.298        |
| R 3.3.2   | Intel MKL 2017U1 (gcc)                      | 4.222      | 0.265        |
| R 3.4.3   | Intel MKL 2018 (gcc) + kpti                 | 4.174      | 0.257        |
| R 3.4.3   | Intel MKL 2018 (gcc) + nopti                | 4.158      | 0.256        |
| R 3.4.4   | Intel MKL 2018 (icc + tbb + no-openmp)      | 4.963      | 0.306        |
| R 3.4.4   | Intel MKL 2018 (gcc + tbb)                  | 4.807      | 0.296        |
| R 3.5.0   | Intel MKL 2018 (gcc + tbb)                  | 4.382      | 0.278        |
| R 3.5.1   | Intel MKL 2018 (gcc + tbb)                  | 4.401      | 0.279        |
| R 3.5.1   | Intel MKL 2018 (gcc + tbb + OC)             | 3.978      | 0.251        |


### Desktop - Xeon 1270 v2 - 4 cores (HT)

| Version   | Options                                     | Total time | Overall mean |
| --------- | ------------------------------------------- | ---------- | ------------ |
| R 3.5.1   | Intel MKL 2018 (gcc + tbb)                  | 4.288      | 0.268        |
| R 3.5.1   | Intel MKL 2019 U1 (gcc)                     | 3.903      | 0.240        |
| R 3.5.2   | Intel MKL 2019 U1 (gcc)                     | 3.890      | 0.239        |
| R 3.5.3   | Intel MKL 2019 U1 (gcc)                     | 4.006      | 0.247        |


# Setting up high-performance R on Ubuntu/Debian

Install prerequisities:

```sh
sudo apt install libcurl4-gnutls-dev libgit2-dev libxml2-dev libssh2-1-dev libssl-dev git build-essential gfortran
```

Setup [CRAN R](https://cran.r-project.org/bin/linux/debian/):

```sh
echo "deb https://ftp.cc.uoc.gr/mirrors/CRAN/bin/linux/ubuntu/ bionic-cran35/" | sudo tee /etc/apt/sources.list.d/cran-r.list
echo "deb-src https://ftp.cc.uoc.gr/mirrors/CRAN/bin/linux/ubuntu/ bionic-cran35/" | sudo tee /etc/apt/sources.list.d/cran-r.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9
sudo apt update
sudo apt install r-base r-base-dev
```

Install [Intel MKL](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo):

```sh
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt update
sudo apt install intel-mkl-64bit-2019.0-045
```

[Compile R](https://software.intel.com/en-us/articles/using-intel-mkl-with-r) with the Intel MKL:

_See [1](https://www.r-bloggers.com/why-is-r-slow-some-explanations-and-mklopenblas-setup-to-try-to-fix-this/)_

```sh
sudo apt-get build-dep r-base
mkdir -p ~/src/r-with-intel-mkl && cd ~/src/r-with-intel-mkl
cd ..
wget https://cran.r-project.org/src/base/R-3/R-3.5.1.tar.gz
tar zxf R-3.5.1.tar.gz -C r-with-intel-mkl/
cd r-with-intel-mkl/R-3.5.1
source /opt/intel/mkl/bin/mklvars.sh intel64
export MKL="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
export _gcc_opt=" -O3 -m64 -fopenmp -march=native"
export LDFLAGS=" -fopenmp"
export CFLAGS="${_gcc_opt} -I${MKLROOT}/include"
export CXXFLAGS="${_gcc_opt} -I${MKLROOT}/include"
export FFLAGS="${_gcc_opt} -I${MKLROOT}/include"
export FCFLAGS="${_gcc_opt} -I${MKLROOT}/include"
./configure --prefix=/opt/R/R-3.5.1-intel-mkl --enable-R-shlib --with-blas="$MKL" --with-lapack
make && sudo make install
printf '\n#export RSTUDIO_WHICH_R=/usr/bin/R\nexport RSTUDIO_WHICH_R=/opt/R/R-3.5.1-intel-mkl/bin/R\n' | tee -a ~/.profile
```

Compile R with OpenBLAS:

```sh
mkdir -p ~/src/r-with-openblas && cd ~/src/r-with-openblas
cd ..
tar zxvf R-3.5.1.tar.gz -C r-with-openblas/
cd ~/src/r-with-openblas/R-3.5.1
./configure --prefix=/opt/R/R-3.5.1-openblas --enable-R-shlib --with-blas --with-lapack
make && sudo make install
printf '#export RSTUDIO_WHICH_R=/opt/R/R-3.5.1-openblas/bin/R\n' | tee -a ~/.profile
```

Compile vanilla R:

```sh
mkdir -p ~/src/r-vanilla && cd ~/src/r-vanilla
cd ..
tar zxvf R-3.5.1.tar.gz -C r-vanilla/
cd ~/src/r-vanilla/R-3.5.1
./configure --prefix=/opt/R/R-3.5.1-vanilla --enable-R-shlib
make && sudo make install
printf '#export RSTUDIO_WHICH_R=/opt/R/R-3.5.1-vanilla/bin/R\n' | tee -a ~/.profile
```

Choose linear algebra libs:

```sh
sudo apt install libatlas3-base
sudo apt install libopenblas-base
sudo update-alternatives --config libblas.so.3
sudo update-alternatives --config liblapack.so.3
```

Set R distribution in use by RStudio Server:

```sh
echo "rsession-which-r=/opt/R/R-3.5.1-intel-mkl/bin/R" | sudo tee -a /etc/rstudio/rserver.conf
```

Set number of threads for OpenBLAS:

```sh
OPENBLAS_NUM_THREADS=8 R --no-save
```

