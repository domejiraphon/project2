# Use pytorch GPU base image
# FROM gcr.io/cloud-aiplatform/training/pytorch-gpu.1-7
#FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest
FROM pytorch/pytorch:latest

# set working directory
WORKDIR /app
#Install vim text editor
#RUN apt-get update
#RUN apt-get install vim -y
ADD ./coco_sampled /app/data
ADD ./style-images /app/style-images
# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


#Switch to a non-root user
#RUN useradd dome && chown -R dome /app
#USER dome

# Copies the trainer code to the docker image.
ADD ./neural_style /app/neural_style



# Set up the entry point to invoke the trainer.
#CMD exec /bin/bash -c "python neural_style/neural_style.py eval \
#    --content-image ./images/content-images/download.jpeg --model ./saved_models/mosaic.pth\
#     --output-image ./output/out.jpg --cuda 0; trap : TERM INT; sleep infinity & wait"
#CMD python neural_style/neural_style.py eval --content-image ./images/content-images/a.png --model ./saved_models/candy.pth --output-image /output/out.jpg --cuda 0
#ENTRYPOINT ["python neural_style/neural_style.py train"]
