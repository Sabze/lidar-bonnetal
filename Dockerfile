FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

ADD requirements.txt setup/requirements.txt

RUN pip3 install -r setup/requirements.txt
RUN apt-get update && apt-get install -y \
	python3-pyqt5 \
	python3-tk \
	git














