.PHONY: build up down health openapi reports logs clean test

# Build the container image
build:
	docker compose build

# Start the API server and wait for readiness
up:
	docker compose up -d
	@echo "Waiting for readiness..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://localhost:8000/readyz > /dev/null 2>&1; then \
			echo "Service ready."; \
			exit 0; \
		fi; \
		sleep 1; \
	done
	@echo "Timeout waiting for readiness." && exit 1

# Stop the service
down:
	docker compose down

# Check health endpoints
health:
	@echo "=== /livez ===" && curl -s http://localhost:8000/livez | python3 -m json.tool
	@echo "=== /readyz ===" && curl -s http://localhost:8000/readyz | python3 -m json.tool

# Dump the OpenAPI schema
openapi:
	@curl -s http://localhost:8000/openapi.json

# Generate batch reports (runs in batch profile)
reports:
	docker compose --profile batch run --rm batch

# Tail API logs
logs:
	docker compose logs -f api

# Run tests locally (not in container)
test:
	uv run pytest service/tests/ -v

# Stop and remove volumes
clean:
	docker compose down -v
