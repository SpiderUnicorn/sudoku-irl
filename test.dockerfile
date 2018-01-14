FROM gcc:7.2.0

RUN mkdir test-dir

ADD ./src/solver ./test-dir
WORKDIR ./test-dir/test

ENTRYPOINT ./test.sh