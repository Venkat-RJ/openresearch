FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libxml2 g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/comparators
COPY egrasp LiSA.dtd Readme.txt.txt frozen-v2.cpp ./
RUN chmod 0555 egrasp && g++ -O2 -std=c++17 frozen-v2.cpp -o frozen-v2 && chmod 0555 frozen-v2 && dpkg-query -W > packages.txt
