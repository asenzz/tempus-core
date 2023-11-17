FROM ubuntu
MAINTAINER asenzz@gmail.com
ENV CFLAGS "-mfma -mavx -mavx2 -march=native -mtune=native"
ENV CXXFLAGS "-mfma -mavx -mavx2 -march=native -mtune=native"
ENV CUDAFLAGS "-Xcompiler=-march=native -Xcompiler=-mtune=native"
ENV CMAKE_INSTALL_PREFIX "/usr"
ENV CMAKE_BUILD_TYPE "Release"
ENV LDFLAGS "-flto"
ENV CUDADIST="ubuntu2204"
ENV CUDAARCH="x86_64"
ENV TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive

RUN apt -yq update
RUN apt -y install tzdata
RUN apt -yq upgrade
RUN apt -yq install git cmake make gpg linux-headers-generic postgresql wget libpq-dev gcc g++

RUN wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/${CUDADIST}/${CUDAARCH}/cuda-${CUDADIST}.pin
RUN wget -O /tmp/cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/${CUDADIST}/${CUDAARCH}/cuda-keyring_1.1-1_all.deb
RUN dpkg -i /tmp/cuda-keyring_1.1-1_all.deb

RUN wget -O /usr/share/keyrings/oneapi-archive-keyring https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && gpg --dearmor /usr/share/keyrings/oneapi-archive-keyring
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list

RUN apt -yq update
RUN apt -yq install cuda cuda-toolkit nvidia-gds intel-basekit

RUN cd /tmp && git clone https://gitlab.com/conradsnicta/armadillo-code.git && cd armadillo-code && mkdir build && git checkout 12.6.x && cd build && cmake .. -DCFLAGS="-mfma -mavx -mavx2 -march=native -mtune=native" -DCXXFLAGS="-mfma -mavx -mavx2 -march=native -mtune=native" -DCUDAFLAGS="-Xcompiler=-march=native -Xcompiler=-mtune=native" -DCMAKE_INSTALL_PREFIX="/usr" -DCMAKE_BUILD_TYPE="Release" && make -j all VERBOSE=1 && make -j install VERBOSE=1

RUN cd /tmp && git clone https://github.com/jtv/libpqxx.git && cd libpqxx && mkdir build && git checkout 7.8.1 && cd build && cmake .. -DCFLAGS="-mfma -fPIC -mavx -mavx2 -march=native -mtune=native" -DCXXFLAGS="-fPIC -mfma -mavx -mavx2 -march=native -mtune=native" -DCMAKE_INSTALL_PREFIX="/usr" -DCMAKE_BUILD_TYPE="Release" && make -j all VERBOSE=1 && make -j install VERBOSE=1

CMD ["echo", "Prepared Tempus Ubuntu distro"]

