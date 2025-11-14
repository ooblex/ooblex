#!/bin/bash
# Modern deployment script for Ooblex
# Supports Docker, Docker Compose, and Kubernetes deployments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="ooblex"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_deps=()
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    # Check for Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # Check for Git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install the missing dependencies and try again."
        exit 1
    fi
    
    # Check for NVIDIA Docker runtime (optional)
    if docker info 2>/dev/null | grep -q nvidia; then
        log_info "NVIDIA Docker runtime detected - GPU support enabled"
        GPU_SUPPORT=true
    else
        log_warn "NVIDIA Docker runtime not found - GPU support disabled"
        GPU_SUPPORT=false
    fi
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE" ]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            log_info "Created .env file from .env.example"
            log_warn "Please review and update the configuration in .env"
        else
            log_error ".env.example file not found!"
            exit 1
        fi
    else
        log_info "Using existing .env file"
    fi
    
    # Create necessary directories
    mkdir -p models ssl services monitoring/grafana/provisioning logs
    
    # Generate self-signed SSL certificates if they don't exist
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        log_info "Generating self-signed SSL certificates..."
        mkdir -p ssl
        openssl req -x509 -newkey rsa:4096 -nodes -days 365 \
            -keyout ssl/key.pem -out ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Ooblex/CN=localhost" 2>/dev/null
        log_info "SSL certificates generated successfully"
    fi
}

download_models() {
    log_info "Checking ML models..."
    
    # Create models directory
    mkdir -p models
    
    # Check if models exist
    if [ -z "$(ls -A models)" ]; then
        log_warn "No models found in models/ directory"
        log_info "Please download or place your ML models in the models/ directory"
        log_info "Example models:"
        log_info "  - Face detection: models/face_detection.onnx"
        log_info "  - Face swap: models/face_swap.onnx"
        log_info "  - Style transfer: models/style_transfer.onnx"
    else
        log_info "Models found in models/ directory"
    fi
}

build_services() {
    log_info "Building Docker images..."
    
    # Build all services
    docker-compose build --parallel
    
    if [ $? -eq 0 ]; then
        log_info "Docker images built successfully"
    else
        log_error "Failed to build Docker images"
        exit 1
    fi
}

start_services() {
    local mode=${1:-"development"}
    
    log_info "Starting services in $mode mode..."
    
    case $mode in
        "development")
            docker-compose up -d
            ;;
        "production")
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
            ;;
        *)
            log_error "Unknown mode: $mode"
            exit 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        log_info "Services started successfully"
        show_status
    else
        log_error "Failed to start services"
        exit 1
    fi
}

stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_info "Services stopped"
}

show_status() {
    log_info "Service Status:"
    docker-compose ps
    
    echo ""
    log_info "Access URLs:"
    echo "  - Web Interface: https://localhost"
    echo "  - API Gateway: https://localhost:8800"
    echo "  - WebRTC Gateway: wss://localhost:8100"
    echo "  - MJPEG Stream: http://localhost:8081"
    echo "  - RabbitMQ Management: http://localhost:15672"
    echo "  - Grafana Dashboard: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
}

show_logs() {
    local service=${1:-""}
    
    if [ -z "$service" ]; then
        docker-compose logs -f --tail=100
    else
        docker-compose logs -f --tail=100 "$service"
    fi
}

update_services() {
    log_info "Updating services..."
    
    # Pull latest images
    docker-compose pull
    
    # Rebuild services
    build_services
    
    # Restart services
    docker-compose up -d
    
    log_info "Services updated successfully"
}

backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    
    log_info "Creating backup in $backup_dir..."
    mkdir -p "$backup_dir"
    
    # Backup database
    docker-compose exec -T postgres pg_dump -U ooblex ooblex | gzip > "$backup_dir/postgres_backup.sql.gz"
    
    # Backup Redis
    docker-compose exec -T redis redis-cli SAVE
    docker cp "$(docker-compose ps -q redis)":/data/dump.rdb "$backup_dir/redis_backup.rdb"
    
    # Backup environment
    cp .env "$backup_dir/.env.backup"
    
    log_info "Backup completed successfully"
}

restore_data() {
    local backup_dir=$1
    
    if [ -z "$backup_dir" ] || [ ! -d "$backup_dir" ]; then
        log_error "Invalid backup directory"
        exit 1
    fi
    
    log_info "Restoring from $backup_dir..."
    
    # Restore database
    if [ -f "$backup_dir/postgres_backup.sql.gz" ]; then
        gunzip -c "$backup_dir/postgres_backup.sql.gz" | docker-compose exec -T postgres psql -U ooblex ooblex
        log_info "Database restored"
    fi
    
    # Restore Redis
    if [ -f "$backup_dir/redis_backup.rdb" ]; then
        docker cp "$backup_dir/redis_backup.rdb" "$(docker-compose ps -q redis)":/data/dump.rdb
        docker-compose restart redis
        log_info "Redis restored"
    fi
    
    log_info "Restore completed"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl first."
        exit 1
    fi
    
    # Check for Helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm not found. Please install Helm first."
        exit 1
    fi
    
    # Deploy using Helm
    helm upgrade --install ooblex ./charts/ooblex \
        --namespace ooblex \
        --create-namespace \
        --values ./charts/ooblex/values.yaml
    
    log_info "Kubernetes deployment completed"
}

# Main script
main() {
    cd "$SCRIPT_DIR"
    
    case ${1:-"help"} in
        "setup")
            check_requirements
            setup_environment
            download_models
            ;;
        "build")
            build_services
            ;;
        "start")
            start_services "${2:-development}"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            start_services "${2:-development}"
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "update")
            update_services
            ;;
        "backup")
            backup_data
            ;;
        "restore")
            restore_data "${2:-}"
            ;;
        "k8s"|"kubernetes")
            deploy_kubernetes
            ;;
        "help"|*)
            echo "Ooblex Deployment Script"
            echo ""
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  setup              - Initial setup (check requirements, create configs)"
            echo "  build              - Build Docker images"
            echo "  start [mode]       - Start services (development|production)"
            echo "  stop               - Stop all services"
            echo "  restart [mode]     - Restart services"
            echo "  status             - Show service status"
            echo "  logs [service]     - Show logs (all services or specific service)"
            echo "  update             - Update and restart services"
            echo "  backup             - Backup data"
            echo "  restore <dir>      - Restore data from backup"
            echo "  k8s|kubernetes     - Deploy to Kubernetes"
            echo "  help               - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 setup                    # Initial setup"
            echo "  $0 start                    # Start in development mode"
            echo "  $0 start production         # Start in production mode"
            echo "  $0 logs api                 # Show API service logs"
            echo "  $0 backup                   # Create backup"
            echo "  $0 restore backups/20240115 # Restore from backup"
            ;;
    esac
}

# Run main function
main "$@"