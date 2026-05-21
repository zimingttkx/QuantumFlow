#!/bin/bash
# 生成 Proto Python 代码

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PROTO_DIR="$PROJECT_ROOT/quantumflow/grpc/proto"
GENERATED_DIR="$PROJECT_ROOT/quantumflow/grpc/generated"

echo "Proto directory: $PROTO_DIR"
echo "Generated directory: $GENERATED_DIR"

cd "$PROJECT_ROOT"

# 生成 Python 代码
python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$GENERATED_DIR" \
    --grpc_python_out="$GENERATED_DIR" \
    "$PROTO_DIR/quantumflow.proto"

# 修复导入路径问题
# grpc_tools.protoc 生成的 import 路径是 "quantumflow_pb2" 而不是 "quantumflow.grpc.generated.quantumflow_pb2"
# 需要修正导入语句

if [ -f "$GENERATED_DIR/quantumflow_pb2_grpc.py" ]; then
    # 替换 import quantumflow_pb2 为 from ... import quantumflow_pb2
    sed -i 's/import quantumflow_pb2/from quantumflow.grpc.generated import quantumflow_pb2/g' "$GENERATED_DIR/quantumflow_pb2_grpc.py"
    echo "Fixed imports in quantumflow_pb2_grpc.py"
fi

echo "Proto generation complete!"
