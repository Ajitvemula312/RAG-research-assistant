# Deployment

Run `docker compose -f deployment/docker-compose.yml up --build`. The API, MLflow server, and Ollama are separate services; model weights are mounted by Ollama and are not copied into the API image. Configure all paths and thresholds through `.env`.