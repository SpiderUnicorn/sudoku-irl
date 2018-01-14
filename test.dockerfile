FROM gcc

RUN mkdir test-dir

ADD ./src/solver ./test-dir

WORKDIR ./test-dir/test

ENTRYPOINT ./test.sh