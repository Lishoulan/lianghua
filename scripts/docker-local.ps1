$ErrorActionPreference = "Stop"

$env:COMPOSE_DISABLE_ENV_FILE = "1"
docker compose --env-file .env.docker.local -f docker-compose.local.yml @args
