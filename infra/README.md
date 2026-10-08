# Cloud deployment starters

These Python modules are **scaffolding only**. They contain TODOs and an intentionally unimplemented provisioning function. They do not define cloud resources, configure accounts, create VMs, start local VMs, or deploy the application. No infrastructure framework or cloud SDK is required yet.

| Cloud | Starter | Future managed Kubernetes service | Future image registry |
| --- | --- | --- | --- |
| Azure | `azure.py` | AKS | Azure Container Registry |
| AWS | `aws.py` | EKS | Amazon ECR |
| GCP | `gcp.py` | GKE | Artifact Registry |

Locate a starter without running any deployment:

```sh
python -m infra --cloud azure
python -m infra --cloud aws
python -m infra --cloud gcp
```

Each `provision_infrastructure()` raises `NotImplementedError` if called. Implement only the selected provider when a real deployment is requested. At that point, choose a Python infrastructure-as-code framework, supply cloud/region settings and credentials, define worker VM/node pool capacity and networking/IAM, and connect the image registry and existing Kubernetes chart. Cloud node autoscaling and API pod autoscaling are separate requirements.

The existing `deploy/clouds/*.yaml` files are placeholder image values for an already-created cluster. The Helm chart and k6 scenario are application deployment/load-test assets; neither provisions cloud infrastructure. Deployment, latency and 1,000-user capacity remain unverified in any cloud.
