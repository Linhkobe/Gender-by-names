## Purpose 

This folder holds runtime configs for the Docker stack:
- 'nginx.conf' : reverse proxy that exposes a stable public port (:8000) and routes to the active app service (blue/green).

## Blue/green traffic switch 
1) build and run BLUE (current) and GREEN (candidate) via docker-compose at the root of repository

2) Edit 'nginx.conf':
- Comment BLUE, uncomment GREEN in the 'upstream gender_api' block

3) Reload Nginx without downtime:
docker compose exec proxy nginx -s reload