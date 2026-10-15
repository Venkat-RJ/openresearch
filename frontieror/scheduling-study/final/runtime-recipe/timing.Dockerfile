FROM frontieror-comparators:20261008
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY timed_solver.py /opt/comparators/timed_solver.py
RUN dpkg-query -W > /opt/comparators/timing-packages.txt
