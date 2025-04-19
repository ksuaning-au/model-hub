#!/bin/bash
# pytest -s -v ./tests --cov=model_hub --cov-report=html
PYTHONPATH=./ pytest -s -v ./tests --cov=model_hub --cov-report=html