FROM ubuntu:20.04

ADD http://robotpkg.openrobots.org/packages/debian/robotpkg.key /

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -qqy \
    build-essential \
    cmake \
    git \
    gnupg2 \
    liburdfdom-dev \
    python-is-python3 \
 && echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub focal robotpkg" \
 >> /etc/apt/sources.list.d/robotpkg.list \
 && apt-key add /robotpkg.key \
 && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -qqy \
    robotpkg-py38-casadi \
    robotpkg-py38-hpp-fcl

ENV CMAKE_PREFIX_PATH=/opt/openrobots

RUN git clone --recursive -b pinocchio3-preview https://github.com/stack-of-tasks/pinocchio \
 && cmake -B build -S pinocchio -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_WITH_CASADI_SUPPORT=ON \
 && cmake --build build -j 4 \
 && cmake --build build -t install

 RUN python -c "import pinocchio"
