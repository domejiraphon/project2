# project2
pipeline.py is used to create a pipeline.yaml file, which specifies the components of our pipeline. The pipeline can be executed simply by pressing the run button, and it is self-contained. It should be noted that the pipeline is tested on Google Cloud Platform, not IBM Cloud. 

## To create yaml file
```
python pipeline.py
```
This will create the pipeline.yaml file using Kubeflow Pipelines SDK package

## Create our pipeline
Create an Experiment environment on Kubeflow dashboard, then upload our pipeline.yaml there to run. 

## Deploy to endpoint
The serve.yaml file is used to deploy the service to a public endpoint.

After connecting to the Kubeflow Cluster on your CLI, simply apply the script:

```
kubectl apply -f serve.yaml
```


Our deployed endpoint is
http://34.121.84.159:82/

> **Note that**: The website should work in general, but you may need to refresh a few times if you encounter any issues. It is possible that these issues are due to how Google Cloud Platform (GCP) handles traffic, as the logs do not show any errors.
