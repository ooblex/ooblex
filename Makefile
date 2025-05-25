# Ooblex Makefile
# Simplified commands for common development tasks

.PHONY: help setup build start stop restart status logs clean test lint format dev prod k8s

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE := docker-compose
PYTHON := python3
PROJECT_NAME := ooblex

help: ## Show this help message
	@echo "Ooblex Development Commands"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Initial project setup
	@echo "Setting up Ooblex..."
	@./deploy.sh setup
	@echo "Setup complete! Run 'make dev' to start development environment"

build: ## Build Docker images
	@echo "Building Docker images..."
	@$(DOCKER_COMPOSE) build --parallel

dev: ## Start development environment
	@echo "Starting development environment..."
	@./deploy.sh start development

prod: ## Start production environment
	@echo "Starting production environment..."
	@./deploy.sh start production

start: dev ## Alias for 'make dev'

stop: ## Stop all services
	@echo "Stopping all services..."
	@$(DOCKER_COMPOSE) down

restart: ## Restart all services
	@echo "Restarting services..."
	@$(DOCKER_COMPOSE) restart

status: ## Show service status
	@$(DOCKER_COMPOSE) ps

logs: ## Follow logs for all services
	@$(DOCKER_COMPOSE) logs -f --tail=100

logs-%: ## Follow logs for specific service (e.g., make logs-api)
	@$(DOCKER_COMPOSE) logs -f --tail=100 $*

shell-%: ## Open shell in service container (e.g., make shell-api)
	@$(DOCKER_COMPOSE) exec $* /bin/bash

test: ## Run test suite
	@echo "Running tests..."
	@$(DOCKER_COMPOSE) exec api pytest tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@$(DOCKER_COMPOSE) exec api pytest tests/ --cov=ooblex --cov-report=html --cov-report=term

lint: ## Run linting checks
	@echo "Running linting checks..."
	@$(DOCKER_COMPOSE) exec api flake8 .
	@$(DOCKER_COMPOSE) exec api mypy .

format: ## Format code with black
	@echo "Formatting code..."
	@$(DOCKER_COMPOSE) exec api black .
	@$(DOCKER_COMPOSE) exec api isort .

migrate: ## Run database migrations
	@echo "Running database migrations..."
	@$(DOCKER_COMPOSE) exec api alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create name="migration_name")
	@echo "Creating new migration..."
	@$(DOCKER_COMPOSE) exec api alembic revision --autogenerate -m "$(name)"

clean: ## Clean up containers, volumes, and generated files
	@echo "Cleaning up..."
	@$(DOCKER_COMPOSE) down -v
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ .pytest_cache/ .mypy_cache/ 2>/dev/null || true

prune: clean ## Deep clean including Docker system prune
	@echo "Deep cleaning..."
	@docker system prune -af --volumes

backup: ## Create backup of data
	@echo "Creating backup..."
	@./deploy.sh backup

update: ## Update dependencies and rebuild
	@echo "Updating dependencies..."
	@$(DOCKER_COMPOSE) pull
	@$(DOCKER_COMPOSE) build --no-cache
	@$(DOCKER_COMPOSE) up -d

docs: ## Build documentation
	@echo "Building documentation..."
	@$(DOCKER_COMPOSE) exec api mkdocs build

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	@$(DOCKER_COMPOSE) exec api mkdocs serve --dev-addr 0.0.0.0:8000

monitoring: ## Open monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "RabbitMQ: http://localhost:15672 (admin/admin)"

gpu-check: ## Check GPU availability
	@echo "Checking GPU availability..."
	@docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

k8s: ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	@./deploy.sh k8s

k8s-delete: ## Delete Kubernetes deployment
	@echo "Deleting Kubernetes deployment..."
	@helm uninstall ooblex -n ooblex

# Development shortcuts
up: dev ## Alias for 'make dev'
down: stop ## Alias for 'make stop'
ps: status ## Alias for 'make status'

# CI/CD helpers
ci-test: ## Run CI test suite
	@docker-compose -f docker-compose.test.yml up --abort-on-container-exit --exit-code-from test

ci-build: ## Build for CI
	@docker-compose -f docker-compose.yml build

# Performance commands
bench: ## Run performance benchmarks
	@echo "Running benchmarks..."
	@$(DOCKER_COMPOSE) exec api python -m pytest benchmarks/ -v

profile-%: ## Profile specific service (e.g., make profile-ml-worker)
	@echo "Profiling $* service..."
	@$(DOCKER_COMPOSE) exec $* python -m cProfile -o profile.stats app.py