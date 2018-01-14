FROM ubuntu

# Disable UI promts
ENV DEBIAN_FRONTEND noninteractive
# Update
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y build-essential

# Install deps
RUN apt-get install git -y
RUN apt-get install cmake -y
RUN apt-get install gcc g++ -y

# Install benchmark framework
RUN git clone -q https://github.com/google/benchmark.git
RUN git clone -q https://github.com/google/googletest.git benchmark/googletest
RUN cd benchmark
RUN mkdir build && cd build
WORKDIR /benchmark
RUN cmake /benchmark -DCMAKE_BUILD_TYPE=RELEASE
RUN make
RUN make install

WORKDIR ./

ADD . .

# Compile benchmark
RUN g++ ./benchmark.cpp -pthread -lbenchmark -std=c++0x -o ./benchmark

ENTRYPOINT ./benchmark