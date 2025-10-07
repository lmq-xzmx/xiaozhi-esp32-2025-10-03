#!/bin/bash

# Xiaozhi ESP32 Server 快速启动脚本
# 用于快速部署和验证优化方案

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查 kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装，请先安装 Kubernetes CLI"
        exit 1
    fi
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    
    # 检查 Kubernetes 连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到 Kubernetes 集群"
        exit 1
    fi
    
    log_success "所有依赖检查通过"
}

# 安装 Python 依赖
install_python_deps() {
    log_info "安装 Python 依赖..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        log_success "Python 依赖安装完成"
    else
        log_warning "requirements.txt 不存在，跳过 Python 依赖安装"
    fi
}

# 部署优化方案
deploy_optimization() {
    log_info "开始部署优化方案..."
    
    # 给脚本执行权限
    chmod +x scripts/optimize-for-100-devices.sh
    chmod +x scripts/monitoring-setup.sh
    
    # 执行优化部署
    log_info "执行优化配置部署..."
    echo "7" | ./scripts/optimize-for-100-devices.sh
    
    # 等待部署完成
    log_info "等待 Pod 启动..."
    sleep 30
    
    # 检查部署状态
    kubectl wait --for=condition=ready pod -l app=xiaozhi -n xiaozhi-system --timeout=300s
    
    log_success "优化方案部署完成"
}

# 部署监控系统
deploy_monitoring() {
    log_info "部署监控系统..."
    
    # 执行监控部署
    echo "5" | ./scripts/monitoring-setup.sh
    
    # 等待监控系统启动
    log_info "等待监控系统启动..."
    sleep 60
    
    # 检查监控部署状态
    kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=300s
    
    log_success "监控系统部署完成"
}

# 验证部署
validate_deployment() {
    log_info "验证部署状态..."
    
    # 运行部署验证
    python3 scripts/deployment-validator.py \
        --output-json validation-results.json \
        --output-text validation-report.txt
    
    # 检查验证结果
    if [ -f "validation-report.txt" ]; then
        log_success "部署验证完成，查看 validation-report.txt 获取详细结果"
        
        # 显示关键结果
        echo ""
        echo "=== 验证结果摘要 ==="
        grep -E "(PASS|FAIL|WARNING)" validation-report.txt | head -10
        echo ""
    else
        log_error "部署验证失败"
        return 1
    fi
}

# 运行性能测试
run_performance_test() {
    log_info "运行性能测试..."
    
    # 获取服务地址
    SERVICE_URL=$(kubectl get svc xiaozhi-lb -n xiaozhi-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    
    if [ "$SERVICE_URL" = "localhost" ]; then
        log_warning "无法获取外部 IP，使用端口转发进行测试"
        kubectl port-forward svc/xiaozhi-lb 8080:80 -n xiaozhi-system &
        PORT_FORWARD_PID=$!
        SERVICE_URL="http://localhost:8080"
        sleep 5
    else
        SERVICE_URL="http://$SERVICE_URL"
    fi
    
    # 运行轻量级性能测试
    python3 scripts/performance-test.py \
        --url "$SERVICE_URL" \
        --devices 20 \
        --duration 120 \
        --interval 3.0 \
        --output quick-performance-results.json
    
    # 清理端口转发
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
    
    log_success "性能测试完成，查看 performance_report.txt 获取结果"
}

# 显示访问信息
show_access_info() {
    log_info "获取访问信息..."
    
    echo ""
    echo "=== 系统访问信息 ==="
    
    # Grafana 访问信息
    GRAFANA_IP=$(kubectl get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "使用端口转发")
    if [ "$GRAFANA_IP" != "使用端口转发" ]; then
        echo "Grafana 监控面板: http://$GRAFANA_IP:3000"
    else
        echo "Grafana 监控面板: kubectl port-forward svc/grafana 3000:3000 -n monitoring"
    fi
    
    # Grafana 密码
    GRAFANA_PASSWORD=$(kubectl get secret grafana-admin -n monitoring -o jsonpath="{.data.password}" 2>/dev/null | base64 -d 2>/dev/null || echo "admin")
    echo "Grafana 用户名: admin"
    echo "Grafana 密码: $GRAFANA_PASSWORD"
    
    # Prometheus 访问信息
    echo "Prometheus: kubectl port-forward svc/prometheus 9090:9090 -n monitoring"
    
    # 主服务访问信息
    XIAOZHI_IP=$(kubectl get svc xiaozhi-lb -n xiaozhi-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "使用端口转发")
    if [ "$XIAOZHI_IP" != "使用端口转发" ]; then
        echo "Xiaozhi 服务: http://$XIAOZHI_IP"
    else
        echo "Xiaozhi 服务: kubectl port-forward svc/xiaozhi-lb 8080:80 -n xiaozhi-system"
    fi
    
    echo ""
}

# 显示后续步骤
show_next_steps() {
    echo ""
    echo "=== 后续步骤建议 ==="
    echo "1. 查看监控面板，观察系统运行状态"
    echo "2. 运行完整性能测试: python3 scripts/performance-test.py --devices 100 --duration 600"
    echo "3. 执行组件评估: python3 scripts/component-evaluator.py --full-evaluation"
    echo "4. 查看部署指南: cat scripts/deployment-guide.md"
    echo "5. 定期运行验证: python3 scripts/deployment-validator.py"
    echo ""
    echo "如需扩容到 1000 台设备，请参考边缘计算部署方案"
    echo ""
}

# 主函数
main() {
    echo ""
    echo "=========================================="
    echo "  Xiaozhi ESP32 Server 快速部署脚本"
    echo "  支持 100 台设备的优化方案"
    echo "=========================================="
    echo ""
    
    # 检查是否在正确目录
    if [ ! -f "scripts/optimize-for-100-devices.sh" ]; then
        log_error "请在 xiaozhi-server 项目根目录下运行此脚本"
        exit 1
    fi
    
    # 执行部署步骤
    check_dependencies
    install_python_deps
    deploy_optimization
    deploy_monitoring
    validate_deployment
    run_performance_test
    show_access_info
    show_next_steps
    
    log_success "快速部署完成！系统已优化支持 100 台设备并发访问"
}

# 错误处理
trap 'log_error "部署过程中发生错误，请检查日志"; exit 1' ERR

# 执行主函数
main "$@"