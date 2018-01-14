FROM madduci/docker-ubuntu-cpp

ADD . .

WORKDIR ./src/solver/test

ENTRYPOINT ./test.sh