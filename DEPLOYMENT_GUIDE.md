# 🚀 Production Deployment Guide

This guide covers deploying Wound-AI API to production environments.

---

## 📋 Table of Contents

- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Security Hardening](#security-hardening)
- [Monitoring Setup](#monitoring-setup)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)

---

## ✅ Pre-Deployment Checklist

### Required

- [ ] Model checkpoint file (`artifacts/checkpoints/best_efficientnet.pt`)
- [ ] Environment variables configured (`.env` file)
- [ ] SSL/TLS certificates (for HTTPS)
- [ ] Backup strategy defined
- [ ] Monitoring tools configured
- [ ] Log aggregation setup
- [ ] Resource limits defined

### Recommended

- [ ] Load testing completed
- [ ] Disaster recovery plan documented
- [ ] API documentation reviewed
- [ ] Rate limiting configured
- [ ] Health check endpoints tested
- [ ] Security audit performed

---

## 🐳 Docker Deployment

### 1. Single Server Deployment

#### Step 1: Prepare Server

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

#### Step 2: Setup Application

```bash
# Clone repository
git clone https://github.com/yourusername/wound-classification-mlops.git
cd wound-classification-mlops

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

**Critical environment variables:**

```bash
CHECKPOINT_PATH=/app/artifacts/checkpoints/best_efficientnet.pt
LOG_LEVEL=WARNING
ENABLE_DOCS=false  # Disable in production
WOUND_API_TESTING=0
```

#### Step 3: Upload Model Checkpoint

```bash
# Create artifacts directory
mkdir -p artifacts/checkpoints

# Upload checkpoint (example with scp)
scp best_efficientnet.pt user@server:/path/to/artifacts/checkpoints/
```

#### Step 4: Start Services

```bash
# Build and start
docker compose up --build -d

# Verify running
docker compose ps

# Check logs
docker compose logs -f wound-api
```

#### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test prediction (with sample image)
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"
```

---

### 2. Docker Swarm (Multi-Node)

For high availability across multiple servers:

```bash
# Initialize swarm on manager node
docker swarm init --advertise-addr <MANAGER-IP>

# Join worker nodes (run on each worker)
docker swarm join --token <TOKEN> <MANAGER-IP>:2377

# Deploy stack
docker stack deploy -c docker-compose.yml wound-ai

# Scale service
docker service scale wound-ai_wound-api=3

# Check service status
docker service ls
docker service ps wound-ai_wound-api
```

---

## ☁️ Cloud Deployments

### AWS Deployment (EC2 + ECS)

#### Option A: Single EC2 Instance

```bash
# Launch EC2 instance (Ubuntu 22.04, t3.medium or larger)
# Security group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

# SSH into instance
ssh ubuntu@<EC2-PUBLIC-IP>

# Follow Docker deployment steps above
```

#### Option B: ECS Fargate (Serverless)

1. **Push image to ECR:**

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT-ID>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build -t wound-ai-api .
docker tag wound-ai-api:latest <ACCOUNT-ID>.dkr.ecr.us-east-1.amazonaws.com/wound-ai-api:latest

# Push to ECR
docker push <ACCOUNT-ID>.dkr.ecr.us-east-1.amazonaws.com/wound-ai-api:latest
```

2. **Create ECS task definition:**

```json
{
  "family": "wound-ai-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "wound-api",
      "image": "<ACCOUNT-ID>.dkr.ecr.us-east-1.amazonaws.com/wound-ai-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "CHECKPOINT_PATH", "value": "/app/artifacts/checkpoints/best_efficientnet.pt"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/wound-ai-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

3. **Create ECS service with Application Load Balancer**

---

### Google Cloud Platform (GCP)

#### Cloud Run Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/wound-ai-api

# Deploy to Cloud Run
gcloud run deploy wound-ai-api \
  --image gcr.io/PROJECT-ID/wound-ai-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

---

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name wound-ai-rg --location eastus

# Create container registry
az acr create --resource-group wound-ai-rg --name woundaiacr --sku Basic

# Build and push image
az acr build --registry woundaiacr --image wound-ai-api:latest .

# Deploy container
az container create \
  --resource-group wound-ai-rg \
  --name wound-ai-api \
  --image woundaiacr.azurecr.io/wound-ai-api:latest \
  --cpu 2 --memory 2 \
  --ports 8000 \
  --environment-variables \
    CHECKPOINT_PATH=/app/artifacts/checkpoints/best_efficientnet.pt \
    LOG_LEVEL=INFO
```

---

## 🔒 Security Hardening

### 1. SSL/TLS Configuration

#### Using Nginx Reverse Proxy

Create `nginx.conf`:

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
}
```

#### Using Let's Encrypt

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal (cron job)
0 0 * * * certbot renew --quiet
```

---

### 2. API Authentication (Optional)

Add JWT authentication to `src/api.py`:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Protect endpoints
@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    token: dict = Depends(verify_token)
):
    # ... existing code
```

---

### 3. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Check status
sudo ufw status
```

---

## 📊 Monitoring Setup

### 1. Prometheus + Grafana

See `monitoring/prometheus.yml` for Prometheus configuration.

#### Start monitoring stack:

```bash
# Uncomment monitoring services in docker-compose.yml
docker compose up prometheus grafana -d

# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin
```

#### Import dashboard:

1. Go to Grafana → Dashboards → Import
2. Use dashboard ID: `11074` (FastAPI dashboard)
3. Configure Prometheus data source

---

### 2. Application Performance Monitoring (APM)

#### Add Prometheus metrics to API:

```python
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration')

# Instrument app
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

---

### 3. Log Aggregation

#### ELK Stack (Elasticsearch, Logstash, Kibana)

```yaml
# Add to docker-compose.yml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
  environment:
    - discovery.type=single-node
  ports:
    - "9200:9200"

logstash:
  image: docker.elastic.co/logstash/logstash:8.5.0
  volumes:
    - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

kibana:
  image: docker.elastic.co/kibana/kibana:8.5.0
  ports:
    - "5601:5601"
  depends_on:
    - elasticsearch
```

---

## 💾 Backup & Recovery

### 1. Model Checkpoint Backup

```bash
# Manual backup
tar -czf wound-ai-backup-$(date +%Y%m%d).tar.gz artifacts/

# Automated daily backup (cron)
0 2 * * * tar -czf /backup/wound-ai-backup-$(date +\%Y\%m\%d).tar.gz /app/artifacts/
```

### 2. S3 Backup (AWS)

```bash
# Install AWS CLI
pip install awscli

# Sync to S3 daily
aws s3 sync artifacts/ s3://wound-ai-backups/artifacts/ --delete
```

### 3. Database Backup (if applicable)

```bash
# Backup SQLite database
cp wound_ai.db wound_ai.db.backup

# Or for PostgreSQL
pg_dump wound_ai > wound_ai_backup.sql
```

---

## 🐛 Troubleshooting

### Issue: Container won't start

**Check logs:**
```bash
docker compose logs wound-api
```

**Common causes:**
- Model checkpoint missing
- Incorrect environment variables
- Port already in use
- Insufficient memory

---

### Issue: Slow inference

**Solutions:**
- Increase CPU/memory limits in `docker-compose.yml`
- Use batch prediction for multiple images
- Enable CPU optimizations:

```python
# In api.py
torch.set_num_threads(4)
model = model.to(memory_format=torch.channels_last)
```

---

### Issue: Out of Memory (OOM)

**Solutions:**
- Reduce batch size
- Increase container memory limit
- Use model quantization:

```python
import torch
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## 📞 Production Support

### Health Check Endpoint

```bash
# Check API health
curl http://localhost:8000/health

# Expected response
{
  "status": "ok",
  "model_loaded": true,
  "uptime_seconds": 3600.0
}
```

### Logs Location

```bash
# API logs
tail -f logs/api.log

# Docker logs
docker compose logs -f --tail=100 wound-api
```

### Restart Service

```bash
# Graceful restart
docker compose restart wound-api

# Force rebuild
docker compose down
docker compose up --build -d
```

---

## 🔗 Additional Resources

- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Docker Production Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)

---

**For issues or questions, contact: your.email@example.com**
