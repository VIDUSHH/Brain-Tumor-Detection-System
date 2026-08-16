# Hugging Face Spaces deployment
#
# This folder contains everything needed to run the app on a free Hugging Face
# Docker Space (https://huggingface.co/docs/hub/spaces-sdks-docker).
#
# Free-tier Spaces: 2 vCPU / 16 GB RAM / 50 GB disk, no bundle-size limit,
# sleeps only after 48h of inactivity (far better than Render's 15 min).
#
# The deployment script `deploy.ps1` stages the Space repo for you:
#   - copies app/, runtime requirements, and the trained model
#   - initializes git-lfs so models/best_model.h5 (128 MB) is pushed properly
#   - creates the Space and pushes everything
#
# Requirements before running: a Hugging Face account and a login token.
#   huggingface-cli login   (or set the HF_TOKEN environment variable)

PORT=7860
