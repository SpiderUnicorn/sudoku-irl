FROM madduci/docker-ubuntu-cpp

ADD . .

ENTRYPOINT ./src/solver/test/test.sh