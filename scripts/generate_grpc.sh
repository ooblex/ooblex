#!/bin/bash
# Generate gRPC Python code from proto files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/protos"
OUTPUT_DIR="$PROJECT_ROOT/services"

echo "Generating gRPC code from proto files..."

# Install required packages
pip install grpcio-tools grpcio-reflection

# Generate Python code for each service
for service in api ml-worker webrtc; do
    echo "Generating code for $service service..."
    
    python -m grpc_tools.protoc \
        -I"$PROTO_DIR" \
        --python_out="$OUTPUT_DIR/$service" \
        --grpc_python_out="$OUTPUT_DIR/$service" \
        "$PROTO_DIR/ooblex.proto"
    
    # Fix imports in generated files
    sed -i 's/import ooblex_pb2/from . import ooblex_pb2/g' "$OUTPUT_DIR/$service/ooblex_pb2_grpc.py"
done

echo "gRPC code generation completed!"

# Generate Go code (optional)
if command -v protoc &> /dev/null; then
    echo "Generating Go code..."
    
    mkdir -p "$PROJECT_ROOT/proto"
    
    protoc \
        -I"$PROTO_DIR" \
        --go_out="$PROJECT_ROOT/proto" \
        --go-grpc_out="$PROJECT_ROOT/proto" \
        "$PROTO_DIR/ooblex.proto"
fi

# Generate documentation (optional)
if command -v protoc-gen-doc &> /dev/null; then
    echo "Generating documentation..."
    
    protoc \
        -I"$PROTO_DIR" \
        --doc_out="$PROJECT_ROOT/docs" \
        --doc_opt=markdown,grpc-api.md \
        "$PROTO_DIR/ooblex.proto"
fi

echo "All done!"