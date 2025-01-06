# Ooblex 🚀

[![Share on HackerNews](https://img.shields.io/badge/Share_on-HackerNews-orange.svg)](https://news.ycombinator.com/submitlink?u=https://github.com/ooblex/ooblex)
[![GitHub Stars](https://img.shields.io/github/stars/ooblex/ooblex?style=social)](https://github.com/ooblex/ooblex)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fooblex%2Fooblex)](https://twitter.com/intent/tweet?text=Check%20out%20Ooblex%20-%20An%20open-source%20platform%20for%20real-time%20AI%20streaming%20and%20inference%20https://github.com/ooblex/ooblex)

## 🎯 What is Ooblex?

Ooblex is an open-source platform for building real-time AI streaming applications with ultra-low latency. Perfect for:
- 🤖 Conversational AI agents with audio/video capabilities
- 🎥 Smart security cameras and IoT devices
- 🧠 Distributed ML inference at scale
- 🌐 Edge-to-cloud AI processing
- 🚁 Remote autonomous systems

## ✨ Key Features

- 🔄 WebRTC streaming with AV1 compression
- 🎯 Ultra-low latency (<100ms) processing
- 🔌 Plugin system for AI services (OpenAI, Google Gemini, etc.)
- 🏗️ Modular architecture for distributed processing
- 📱 Edge device support (Raspberry Pi, Intel NUC, etc.)
- 🔧 Easy integration with Python ML frameworks
- 🎛️ Flexible deployment options (edge, cloud, hybrid)

## 🏗️ System Architecture
![Flow](untitled_diagram.png)

## 🚀 Quick Start

### Server Requirements
- Solid internet connection
- Domain name (for SSL/WebRTC)
- Basic: 4+ CPU cores, 8GB+ RAM
- ML Processing: 16GB+ RAM, GPU recommended
- Edge Deployment: Works on Raspberry Pi 3+ w/AI accelerator

### Installation (Ubuntu 18.04+)

```
cd ~
sudo apt-get update
sudo apt-get install git
git clone https://github.com/ooblex/ooblex/

cd ooblex
sudo chmod +x *.sh

sudo ./install_opencv.sh
```
We should now be building OpenCV, which can take 30 minutes to a few hours to install.
```
sudo ./install_nginx.sh
```
During the NGINX install, it will request information about a domain name to generate a free SSL certificate. As mentioned in the requirements above, you will need to have a domain name pointing at your server for this to work. Also, if asked, redirecting from HTTP to HTTPS is preferred.
```
sudo ./install_tensorflow.sh
sudo ./install_gstreamer.sh
sudo ./install_rabbitmq.sh
sudo ./install_janus.sh
sudo ./install_webrtc.sh
sudo ./install_redis.sh
```
The dependencies for Ooblex are largely now installed as we have now installed several core components. If you are deploying to the cloud, your cloud provider may offer some of these tools as hosted services, which would be useful for large managed clustered deployments.

We can enter the code directory now and configure some final settings.

Edit the config.py file so that it uses 'your' REDIS and RabbitMQ uri is used. Also, you will need enter the domain name used when configuring the SSL certificate earlier.

[recent note: The REDIS and RABBITMQ settings in the config.py are setup for localhost access currently, which may not work for production purposes. You can access the rabbitMQ admin dashboard via http://localhost:15672 with user/password guest/guest as a side note, to monitor/configure/add any missing data queues (ie, "broadcast-all").  You can also configure the domain to be localhost, but you'll need to disable SSL in a few places to get that working, and it will then only run locally, so not advised.]

You can edit the config.py with the following command:
```
sudo nano config.py
```
or use vim if you are cool enough to know how to use that.

Next, let's test things out:
```
cd ~/ooblex/code
python3 api.py
```
Running the api.py file with Python should now start the main API server, however there are several other services that we will need to run.  You can stop this server for now, or push it to the background ie) control-Z & bg

Next, lets configure and copy the files in the HTML folder to your /var/www/html folder.

You will need to MODIFY both the index.html file AND the Ooblex.v0.js file contained within the JS folder
You will need to update the domain names from api.ooblex.com to whatever your domain is.
A one-liner to do this all possibly is this is:

```
sudo find ~/ooblex/html -type f -exec sed -i 's/api.ooblex.com/YOU-DOMAIN-HERE/g' {} \;
sudo cp ~/ooblex/html/* /var/www/html -r
```

At this point, going to your domain in your browser should show a yellow website. 

Next, in ~/ooblex/code, we can try running our next service:

```
cd ~/ooblex/code
python3 decoder.py
```

Any error about image folders can be ignored at this point. If things otherwise works, great! You will likely want to modify the file so that the REDIS and AMPQ connection strings point to your own servers, rather the Ooblex's demo servers.

You will also need to test mjpeg.py.  This will also require the SSL certificate location to be properly configured.

```
python3 mjpeg.py
```

webrtc.py will fail if Janus is also not started and configured first.  Please see the Janus_config folder for instructions on how to configure Janus.  Start the Janus once configured using the following command, and then test to ensure webrtc.py starts.

```
sudo /opt/janus/bin/janus -o
```

Lastly, once the system is all configured, and each of the several servers so far are running all together, it is possible to modify the brain.py file, which contains the tensor threads.

The brain.py is configured to operate with a popular video-based facial recognition Tensorflow models, along with some face transformation models (Trump, Taylor), which can be downloaded as needed and implemnted. 

Due to the size of the models, they cannot be hosted directly on github, but they can be downloaded from here: https://api.ooblex.com/models/

Automate the download of all models with the following:
```
cd ~/ooblex/
wget --recursive --no-parent -nH https://api.ooblex.com/models
```

You will need to have the following servers running at the same time as well for things to now run, so together,
or a one liner for everything

```
cd  ~/ooblex/code/
sudo /opt/janus/bin/janus -o 
# wait a moment for Janus to start, then
sudo python3 api.py & python3 brain.py & python3 decoder.py & python3 mjpeg.py & python3 webrtc.py &
```

If setup correctly, along with the remote model files, the HTML files we hosted (the yellow website) should enable the Ooblex Demo to work. Check the console log for errors if not working.

### Tensor Threads

The tensorthread_shell.py is a template for your own AI scripts: it is what brain.py uses for its own demo code. It is quite accessible if you're familiar with Python and Tensorflow already. Working with IBM's Watson Studio, exporting a Python-based version of a trained model can be directly imported into this tensorthread_shell.py file for rapid deployment of a high performing, low-latency, serialized model-- virtually no coding needed.

brain.py, or tensorthread_shell.py for that matter, can be distributed and run across many servers. Just repeat the above setup steps but instead just run brain.py (or tensorthread_shell.py), without any of the other services loaded on that new server. The Tensor Threads will work on AI tasks from the main REDIS/RabbitMQ queue and accelerate the overall system's processing performance!  It can easily also be run on Windows or deployed to virtually any system that supports Python and Tensorflow (or TensorRT).

### OpenVINO

Tensor Threads do not need to use TensorFlow, nor AI logic in general, so what is loaded into them can vary based on your needs. We have successfully validated that OpenVINO, Caffe, PyTorch, and TensorRT, can replace Tensorflow within these Tensor Threads for deployment of the code onto even embedded devices. 

### Advanced

While the above install is pretty rudementry, Ooblex was designed as a set of services that are loosely tied to each other. Running each component on its own server, or multiple copies of a single component across many servers, is easy to accomplish and allows for quite a bit of performance scaling. 

In this demo installation and deployment, we look at Ooblex working as a whole, but individual components can also bring value on their own. The WebRTC gateway is optional if receieving data from a remote security camera. The WebRTC component can also be used indepedently of the Tensor Threads, if live media streaming capabilities is all that is required.

Information on the core server files at play:
```
mjpeg.py -- JPEG streaming server for low-latency output. 

api.py -- The main REST API server, used for external communication with the Ooblex deployment and orchestrating many of the intenral system components.

brain.py -- This contains the Tensor Thread code as a wrapper for a Python-based TensorThread model. It is pre-configured with example logic. This particular script can be loaded onto many different machines, and as long as they have access to the RabbitMQ and REDIS server, they will assist in processing.

tensorthread_shell.py -- THIS IS A SIMPLIFIED version of the above brain.py file. It can be used as a bare bones framework for any PYTHON-based machine learning code.  Plug and play largely, with a very basic amount of sample code / boilerplate.

webrtc.py -- This is the main API layer for the the WebRTC service

decoder.py -- This is the main live media deocder thread, configured for live webRTC video ingestion.
```
pixel_shuffler.py, npy files, and model.py files support the alread-configured AI models loaded in brain.py. These can be modified or removed as needed, depending on changes to brain.py

tensorthread_shell.py does not actually run-- but it is used as an example / boilerplate to create your own custom tensorthread processes. Each Tensorthread can store in memory dozens or hundreds of models, assuming sufficient system RAM. 

### Future Work

We would like Tensor Threads to not need any touching at all, and would prefer models to be hosted in an on-demand database that Tensor Threads can pull from. This on-demand database would be accessible via a webinterface, which can allow for models to be uploaded as microservices or pulled directly from model training tools like IBM Watson. 

This system would also intelligently optimize the model being pulled to meet the architecture being deployed to: Intel CPU, FPGA, ARM, TPU, & Nvidia. Intelligent in-memory model caching and orchestrating of the Tensor Threads would adapt resources and processign demands as needed.

We would like to increase accessibility for data-ingestion devices, using simple images, including for the Raspberry Pi using WebRTC. We would also like to maintain cloud-hosted deployable one-click images of the system, so that anyone can have it up and running within moments.

We would like to develop a toolkit for embedded devices and mobile devices, to allow for flexible deployment of Ooblex, along with centralized remote management of deployments. 

Last but not least, increased readibility of the code, better seperation of the services, better virtual server support, and general code clean up is still all greatly needed.


## 🧩 Integration Examples

### Conversational AI Agent
```python
from ooblex import TensorThread
from openai import OpenAI

class ConversationalAgent(TensorThread):
    def process_frame(self, frame, audio):
        # Process audio/video with OpenAI
        response = self.openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": frame},
                {"type": "text", "text": audio}
            ]}]
        )
        return response.choices[0].message.content
```

## 🛠️ Use Cases

1. **AI Assistant Platforms**
   - Real-time video/audio processing
   - Multi-modal conversation handling
   - Distributed inference processing

2. **Smart Security Systems**
   - Low-latency video analysis
   - Real-time threat detection
   - Edge processing capabilities

3. **Autonomous Systems**
   - Drone/robot control
   - Real-time decision making
   - Remote operation capabilities
     
### Status of project

Pictures from some years ago, presenting a demo to the IBM office, showcasing all the cool things that we were able to make and do with Ooblex, are below.  Thank you.

![image](https://user-images.githubusercontent.com/2575698/230468519-3da3cfaf-1a1a-4bfa-94ce-d784ccfe0171.png)

![image](https://user-images.githubusercontent.com/2575698/230468482-c3c4eb30-1597-4e9b-8b65-73a4838b41ac.png)

