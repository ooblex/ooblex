#!/bin/bash

VERSION=1.12.4

mkdir ~/gstreamer_$VERSION
cd ~/gstreamer_$VERSION

wget https://gstreamer.freedesktop.org/src/gst-plugins-base/gst-plugins-base-$VERSION.tar.xz
wget https://gstreamer.freedesktop.org/src/gstreamer/gstreamer-$VERSION.tar.xz
wget https://gstreamer.freedesktop.org/src/gst-plugins-ugly/gst-plugins-ugly-$VERSION.tar.xz
wget https://gstreamer.freedesktop.org/src/gst-plugins-good/gst-plugins-good-$VERSION.tar.xz
wget https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-$VERSION.tar.xz
wget https://gstreamer.freedesktop.org/src/gst-libav/gst-libav-$VERSION.tar.xz
wget https://gstreamer.freedesktop.org/src/gst-python/gst-python-$VERSION.tar.xz
for a in `ls -1 *.tar.*`; do tar -xf $a; done

sudo apt-get install build-essential dpkg-dev flex bison autotools-dev automake \
liborc-dev autopoint libtool gtk-doc-tools yasm libgstreamer1.0-dev \
libxv-dev libasound2-dev libtheora-dev libogg-dev libvorbis-dev \
libbz2-dev libv4l-dev libvpx-dev libjack-jackd2-dev libsoup2.4-dev libpulse-dev \
faad libfaad-dev libfaac-dev libgl1-mesa-dev libgles2-mesa-dev \
libx264-dev libmad0-dev -y


sudo apt-get install -y build-essential autotools-dev automake autoconf \
                                    libtool autopoint libxml2-dev zlib1g-dev libglib2.0-dev \
                                    pkg-config bison flex python3 git gtk-doc-tools libasound2-dev \
                                    libgudev-1.0-dev libxt-dev libvorbis-dev libcdparanoia-dev \
                                    libpango1.0-dev libtheora-dev libvisual-0.4-dev iso-codes \
                                    libgtk-3-dev libraw1394-dev libiec61883-dev libavc1394-dev \
                                    libv4l-dev libcairo2-dev libcaca-dev libspeex-dev libpng-dev \
                                    libshout3-dev libjpeg-dev libaa1-dev libflac-dev libdv4-dev \
                                    libtag1-dev libwavpack-dev libpulse-dev libsoup2.4-dev libbz2-dev \
                                    libcdaudio-dev libdc1394-22-dev ladspa-sdk libass-dev \
                                    libcurl4-gnutls-dev libdca-dev libdvdnav-dev \
                                    libexempi-dev libexif-dev libfaad-dev libgme-dev libgsm1-dev \
                                    libiptcdata0-dev libkate-dev libmimic-dev libmms-dev \
                                    libmodplug-dev libmpcdec-dev libofa0-dev libopus-dev \
                                    librsvg2-dev librtmp-dev libschroedinger-dev libslv2-dev \
                                    libsndfile1-dev libsoundtouch-dev libspandsp-dev libx11-dev \
                                    libxvidcore-dev libzbar-dev libzvbi-dev liba52-0.7.4-dev \
                                    libcdio-dev libdvdread-dev libmad0-dev libmp3lame-dev \
                                    libmpeg2-4-dev libopencore-amrnb-dev libopencore-amrwb-dev \
                                    libsidplay1-dev libtwolame-dev libx264-dev libusb-1.0 \
                                    python-gi-dev yasm python3-dev libgirepository1.0-dev

cd ~/gstreamer_$VERSION

cd gstreamer-$VERSION
./configure && make && sudo make install && cd ..

sudo ldconfig

#Base
cd gst-plugins-base-$VERSION
./configure && make && sudo make install && cd ..
sudo ldconfig

#Good - 8 minutes
cd gst-plugins-good-$VERSION
./autogen.sh && make && sudo make install && cd ..

#Bad
cd gst-plugins-bad-$VERSION
./configure && make && sudo make install && cd ..

# Ugly
cd gst-plugins-ugly-$VERSION
./configure && make && sudo make install && cd ..

# LibAV
cd gst-libav-$VERSION
./configure && make && sudo make install && cd ..

# Python-bindings
cd gst-python-$VERSION
./configure && make && sudo make install && cd ..

sudo ldconfig

## Increase port size
sudo /sbin/sysctl -w net.core.rmem_max=33554432 
