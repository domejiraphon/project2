# Use pytorch GPU base image
# FROM gcr.io/cloud-aiplatform/training/pytorch-gpu.1-7
#FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest
FROM pytorch/pytorch:latest

# set working directory
WORKDIR /app
#Install vim text editor
#RUN apt-get update
#RUN apt-get install vim -y

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


#Switch to a non-root user
#RUN useradd dome && chown -R dome /app
#USER dome

# Expose any ports the app is expecting in the environment
ENV PORT 8001
EXPOSE $PORT
ENV FLASK_ENV=development

# Copies the trainer code to the docker image.
ADD ./neural_style /app/neural_style
ADD ./static /app/static
ADD ./saved_models /app/saved_models 
ADD ./templates /app/templates
ADD ./flask_server.py /app/flask_server.py


CMD ["python", "flask_server.py" ]

