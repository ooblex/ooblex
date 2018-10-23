sudo apt-get install python3-pip -y
pip3 install --upgrade pip

#https://github.com/mind/wheels/releases

sudo apt install cmake
rm /tmp/src -r
mkdir -p /tmp/src && \
cd /tmp/src

git clone https://github.com/01org/mkl-dnn.git
cd mkl-dnn/scripts && ./prepare_mkl.sh && cd ..
mkdir -p build && cd build && cmake .. && make
sudo make install

sudo echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc


sudo pip3 install --upgrade https://github.com/mind/wheels/releases/download/tf1.7-cpu/tensorflow-1.7.0-cp35-cp35m-linux_x86_64.whl
