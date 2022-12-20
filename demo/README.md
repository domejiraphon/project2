# DEMO
This is the demo of our applications running on Kubernetes Cluster, using IBMCloud. 

## Inference Deployment
The inference-style.yaml will deploy the inference on to the Cluster using Kubernetes Deployment
After connecting to the Kubeflow Cluster on your CLI, simply apply the script:

```
kubectl apply -f inference-style.yaml
```

## Deploy to endpoint
The service.yaml file is used to deploy the service to a public endpoint.

```
kubectl apply -f service.yaml
```


Our deployed endpoint is
http://df45f84b-eu-gb.lb.appdomain.cloud:8001/

