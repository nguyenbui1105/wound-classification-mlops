# Docker CI/CD Setup Guide

This guide explains how to configure GitHub Secrets and test the Docker build workflow.

## Prerequisites

1. **Docker Hub Account**: Create one at https://hub.docker.com if you don't have it
2. **GitHub Repository**: Your code must be pushed to GitHub

## Required GitHub Secrets

You must add the following secrets to your GitHub repository before the workflow can run.

### Steps to Add Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each of the following:

### Required Secrets for Docker Build & Push

| Secret Name | Description | How to Get It |
|-------------|-------------|---------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username | Your username from https://hub.docker.com |
| `DOCKERHUB_TOKEN` | Docker Hub Personal Access Token (PAT) | Create at https://hub.docker.com/settings/security → **New Access Token** |

### Example: Creating Docker Hub PAT

1. Go to https://hub.docker.com/settings/security
2. Click **New Access Token**
3. Description: `GitHub Actions - Wound AI`
4. Access permissions: **Read & Write**
5. Click **Generate**
6. **Copy the token immediately** (you won't see it again)
7. Paste it into GitHub Secrets as `DOCKERHUB_TOKEN`

---

## Optional Secrets for Deployment (Currently Commented Out)

These are needed only if you uncomment the `deploy` job in the workflow:

| Secret Name | Description | How to Get It |
|-------------|-------------|---------------|
| `SSH_HOST` | Remote server IP or domain | Your VPS/server IP (e.g., `192.168.1.100` or `server.example.com`) |
| `SSH_USER` | SSH username | Usually `ubuntu`, `root`, or your custom user |
| `SSH_KEY` | Private SSH key for authentication | Run `cat ~/.ssh/id_rsa` on your local machine or generate a new key pair |

---

## Testing the Workflow Locally (Before Push)

### Option 1: Test Docker Build Locally

```powershell
# Build the image locally to verify Dockerfile works
docker build -t wound-ai-api:test .

# Run it locally
docker run -p 8000:8000 -e WOUND_API_TESTING=1 wound-ai-api:test

# Test the API
curl http://localhost:8000/health
```

### Option 2: Test with Docker Compose

```powershell
# Use your existing docker-compose.yml
docker compose up --build

# In another terminal
curl http://localhost:8000/health
```

---

## Workflow Behavior

### On Pull Request
- ✅ **Builds** the Docker image
- ❌ **Does NOT push** to Docker Hub
- Purpose: Verify Dockerfile builds successfully

### On Push to `main`, `master`, or `develop`
- ✅ **Builds** the Docker image
- ✅ **Pushes** to Docker Hub with two tags:
  - `latest`
  - `<commit-sha>` (e.g., `a1b2c3d4e5f6...`)

### Deploy Job (Currently Disabled)
- 🔒 Commented out by default
- Will auto-deploy when uncommented
- Only runs on push to `main` or `master`

---

## First-Time Setup Checklist

- [ ] Create Docker Hub account
- [ ] Generate Docker Hub Personal Access Token (PAT)
- [ ] Add `DOCKERHUB_USERNAME` to GitHub Secrets
- [ ] Add `DOCKERHUB_TOKEN` to GitHub Secrets
- [ ] Test Docker build locally
- [ ] Commit and push workflow file to GitHub
- [ ] Verify workflow runs successfully in GitHub Actions tab

---

## After First Push

1. **Push your code to GitHub:**
   ```powershell
   git add .
   git commit -m "Add Docker CI/CD workflow"
   git push origin main
   ```

2. **Monitor the workflow:**
   - Go to your GitHub repo → **Actions** tab
   - Click on the running workflow
   - Watch the "docker" job execute

3. **Verify Docker Hub:**
   - Go to https://hub.docker.com
   - Check your repositories
   - You should see `wound-ai-api` with two tags:
     - `latest`
     - `<your-commit-sha>`

4. **Pull and run your published image:**
   ```powershell
   docker pull <your-dockerhub-username>/wound-ai-api:latest
   docker run -p 8000:8000 -e WOUND_API_TESTING=1 <your-dockerhub-username>/wound-ai-api:latest
   ```

---

## Troubleshooting

### Workflow Fails: "Error: Username and password required"

**Cause:** GitHub Secrets not set or incorrect

**Fix:**
- Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are set in GitHub repo settings
- Ensure token has **Read & Write** permissions
- Regenerate token if expired

### Workflow Fails: "denied: requested access to the resource is denied"

**Cause:** Token permissions insufficient or repository name mismatch

**Fix:**
- Check Docker Hub token has write access
- Verify repository name format: `username/wound-ai-api`
- Ensure `DOCKERHUB_USERNAME` matches your actual Docker Hub username (case-sensitive)

### Build Fails: "ERROR [internal] load metadata for docker.io/library/python:3.11-slim"

**Cause:** Network issue or Docker Hub rate limit

**Fix:**
- Wait and retry (rate limits reset after ~6 hours)
- Authenticate to Docker Hub before pull (workflow already does this)

### Image Size Too Large

**Current image:** ~800MB (CPU-only PyTorch)

**Optimization tips:**
- Multi-stage builds (already minimal)
- Use `.dockerignore` to exclude unnecessary files
- Clean up pip cache in Dockerfile (already done)

---

## Enabling Auto-Deployment (Optional)

When you're ready to auto-deploy to a remote server:

1. **Set up SSH key authentication on your server**
2. **Add deployment secrets** (`SSH_HOST`, `SSH_USER`, `SSH_KEY`)
3. **Update deployment path** in workflow (line with `cd /path/to/your/deployment/directory`)
4. **Uncomment the `deploy` job** in `.github/workflows/docker-build.yml`
5. **Push changes** to GitHub

---

## Security Best Practices

- ✅ Never commit Docker Hub tokens to git
- ✅ Use GitHub Secrets for all credentials
- ✅ Rotate tokens every 90 days
- ✅ Use minimal token permissions (Read & Write only)
- ✅ Keep SSH keys private (never share or commit)
- ✅ Use separate SSH keys for CI/CD (not your personal key)

---

## Additional Resources

- [Docker Hub Tokens](https://docs.docker.com/docker-hub/access-tokens/)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [SSH Action for Deployment](https://github.com/appleboy/ssh-action)

---

## Summary

Your Docker workflow is ready! After adding the two required secrets (`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`), every push to `main`, `master`, or `develop` will automatically build and publish your Docker image to Docker Hub.
