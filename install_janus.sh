sudo apt-get update

sudo apt-get install -y --no-install-recommends \
	apt-transport-https \
	bash-completion \
	ca-certificates \
	cmake \
	curl \
	file \
	git \
	htop \
	iproute2 \
	iputils-ping \
	jq \
	less \
	netcat-openbsd \
	net-tools \
	openssl \
	patch \
	procps \
	psmisc \
	rsync \
	ssh \
	strace \
	sudo \
	tcpdump \
	telnet \
	vim \
	wget \
	golang

sudo apt-get install libmicrohttpd-dev libjansson-dev libnice-dev \
		libssl-dev libsrtp-dev libsofia-sip-ua-dev libglib2.0-dev \
		libopus-dev libogg-dev libcurl4-openssl-dev liblua5.3-dev \
		pkg-config gengetopt libtool automake -y

sudo apt-get install libmicrohttpd-dev libjansson-dev libnice-dev \
	libssl-dev libsrtp-dev libsofia-sip-ua-dev libglib2.0-dev \
	libopus-dev libogg-dev libcurl4-openssl-dev pkg-config gengetopt \
	libtool automake libcurl4-openssl-dev -y
	
sudo apt-get install -y --no-install-recommends \
	automake \
	build-essential \
	cmake \
	gengetopt \
	git-core \
	libavcodec-dev \
	libavformat-dev \
	libavutil-dev \
	libcurl4-openssl-dev \
	libglib2.0-dev \
	libjansson-dev \
	libmicrohttpd-dev \
	libnice-dev \
	libogg-dev \
	libopus-dev \
	libsofia-sip-ua-dev \
	libssl-dev \
	libtool \
	pkg-config

git clone https://boringssl.googlesource.com/boringssl
cd boringssl
# Don't barf on errors
sed -i s/" -Werror"//g CMakeLists.txt
# Build
mkdir -p build
cd build
cmake -DCMAKE_CXX_FLAGS="-lrt" ..
make
cd ..
# Install
sudo mkdir -p /opt/boringssl
sudo cp -R include /opt/boringssl/
sudo mkdir -p /opt/boringssl/lib
sudo cp build/ssl/libssl.a /opt/boringssl/lib/
sudo cp build/crypto/libcrypto.a /opt/boringssl/lib/

	rm /tmp/src -r
	mkdir -p /tmp/src && \

	wget https://www.openssl.org/source/openssl-1.0.2n.tar.gz && \
	tar -xvf openssl-1.0.2n.tar.gz && \
	cd openssl-1.0.2n
	apt-get install crypto-dev && \
	sudo apt-get install build-essential gcc -y && \
	sudo ./config && \
	sudo make && \
	sudo make install

	curl -fSL https://github.com/cisco/libsrtp/archive/v2.0.0.tar.gz -o /tmp/src/libsrtp.tar.gz && \
	tar xzf /tmp/src/libsrtp.tar.gz -C /tmp/src && \
	cd /tmp/src/libsrtp-2.0.0 && \
	./configure --prefix=/usr --enable-openssl && \
	make shared_library && \
	sudo make install
	
	mkdir -p /tmp/src/ && \
	cd /tmp/src/ && \
	git clone https://github.com/sctplab/usrsctp && \
	cd /tmp/src/usrsctp && \
	git checkout 2f6478eb8d40f1766a96b5b033ed26c0c2244589 && \
	./bootstrap && \
	./configure --prefix=/usr && \
	make && \
	sudo make install
	
	mkdir -p /tmp/src/ && \
        cd /tmp/src/ && \
	git clone git://git.libwebsockets.org/libwebsockets && \
	cd libwebsockets && \
	git checkout v2.4-stable && \
	mkdir build && \
	cd build && \
	cmake -DLWS_AX_SMP=1 -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_C_FLAGS="-fpic" .. && \
	make && sudo make install

	mkdir -p /tmp/src/ && \
	git clone https://github.com/alanxz/rabbitmq-c && \
	cd rabbitmq-c && \
	git submodule init && \
	git submodule update && \
	mkdir build && cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \
	make && sudo make install

	mkdir -p /tmp/src/ && \
	cd /tmp/src/ && \
	git clone https://github.com/meetecho/janus-gateway.git && \
	cd ./janus-gateway && \
	./autogen.sh && \
	./configure \
		--prefix=/opt/janus \
		--enable-post-processing \
		--enable-rest \
		--enable-data-channels \
		--enable-websockets \
		--enable-unix-sockets \
		--enable-rabbitmq \
		--enable-boringssl \
		--enable-dtls-settimeout && \
	make && \
	sudo make install



