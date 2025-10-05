#!/bin/bash
# 专用服务器系统级优化脚本
# 适用于: 4核3GHz + 7.5GB内存的专用小智ESP32服务器
# 优化目标: 支持3-4台设备稳定并发，消除卡顿现象

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        exit 1
    fi
}

# 备份原始配置
backup_configs() {
    log_step "备份原始系统配置..."
    
    BACKUP_DIR="/root/xiaozhi-server/backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # 备份重要配置文件
    cp /etc/sysctl.conf "$BACKUP_DIR/sysctl.conf.bak" 2>/dev/null || true
    cp /etc/security/limits.conf "$BACKUP_DIR/limits.conf.bak" 2>/dev/null || true
    cp /etc/systemd/system.conf "$BACKUP_DIR/system.conf.bak" 2>/dev/null || true
    
    log_info "配置文件已备份到: $BACKUP_DIR"
}

# 系统信息检查
check_system_info() {
    log_step "检查系统信息..."
    
    # CPU信息
    CPU_CORES=$(nproc)
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    
    # 内存信息
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
    
    # 磁盘信息
    DISK_USAGE=$(df -h / | awk 'NR==2{print $5}')
    
    log_info "CPU: $CPU_MODEL ($CPU_CORES 核心)"
    log_info "内存: ${AVAILABLE_MEM}GB 可用 / ${TOTAL_MEM}GB 总计"
    log_info "磁盘使用率: $DISK_USAGE"
    
    # 检查最低要求
    if [[ $CPU_CORES -lt 4 ]]; then
        log_warn "CPU核心数不足4个，性能可能受限"
    fi
    
    if [[ $TOTAL_MEM -lt 7 ]]; then
        log_warn "内存不足7GB，可能影响性能"
    fi
}

# 检查内核参数是否存在
check_sysctl_param() {
    local param="$1"
    local path="/proc/sys/$(echo $param | tr '.' '/')"
    [ -f "$path" ] || [ -d "$path" ]
}

# 安全设置内核参数
safe_sysctl() {
    local param="$1"
    local value="$2"
    if check_sysctl_param "$param"; then
        echo "$param = $value" >> /etc/sysctl.d/99-xiaozhi-dedicated.conf
    else
        log_warn "内核参数 $param 在此系统中不支持，跳过"
    fi
}

# 内核参数优化
optimize_kernel_parameters() {
    log_step "优化内核参数..."
    
    # 创建专用的sysctl配置文件头部
    cat > /etc/sysctl.d/99-xiaozhi-dedicated.conf << 'EOF'
# 小智ESP32专用服务器内核优化配置
# 适用于4核3GHz + 7.5GB内存的专用服务器

# ===== 内存管理优化 =====
# 虚拟内存管理
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.dirty_expire_centisecs = 1500
vm.dirty_writeback_centisecs = 500

# 内存分配策略
vm.overcommit_memory = 1
vm.overcommit_ratio = 80

# 内存回收优化
vm.min_free_kbytes = 131072
vm.vfs_cache_pressure = 50

# 大页内存（如果支持）
vm.nr_hugepages = 128

# ===== CPU调度优化 =====
# 进程调度器优化（基础参数）
kernel.sched_autogroup_enabled = 0

# CPU频率调节
kernel.timer_migration = 0

# ===== 网络优化 =====
# TCP缓冲区大小
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216

# TCP窗口缩放
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# TCP连接优化
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.tcp_keepalive_probes = 9
net.ipv4.tcp_keepalive_intvl = 75

# 网络队列优化
net.core.netdev_max_backlog = 5000
net.core.netdev_budget = 600

# ===== 文件系统优化 =====
# 文件描述符限制
fs.file-max = 1048576
fs.nr_open = 1048576

# inotify优化
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512

# ===== 安全和稳定性 =====
# 内核安全
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2

# 进程限制
kernel.pid_max = 65536
kernel.threads-max = 131072

# ===== 专用服务器特殊优化 =====
# 禁用不必要的功能
kernel.printk = 3 4 1 3
kernel.hung_task_timeout_secs = 30
EOF

    # 安全设置高级内核参数（可能在某些系统中不存在）
    safe_sysctl "kernel.sched_migration_cost_ns" "5000000"
    safe_sysctl "kernel.sched_tunable_scaling" "0"
    safe_sysctl "kernel.softirq_time_limit_ns" "2000000"

    # 应用内核参数
    sysctl -p /etc/sysctl.d/99-xiaozhi-dedicated.conf
    log_info "内核参数优化完成"
}

# 系统限制优化
optimize_system_limits() {
    log_step "优化系统限制..."
    
    # 创建专用的limits配置
    cat >> /etc/security/limits.conf << 'EOF'

# 小智ESP32专用服务器限制优化
# 文件描述符限制
* soft nofile 65536
* hard nofile 65536
root soft nofile 65536
root hard nofile 65536

# 进程数限制
* soft nproc 32768
* hard nproc 32768
root soft nproc 32768
root hard nproc 32768

# 内存锁定限制
* soft memlock unlimited
* hard memlock unlimited
root soft memlock unlimited
root hard memlock unlimited

# 核心转储大小
* soft core unlimited
* hard core unlimited
EOF

    log_info "系统限制优化完成"
}

# CPU调度器优化
optimize_cpu_scheduler() {
    log_step "优化CPU调度器..."
    
    # 设置CPU调度器为性能模式
    if command -v cpupower >/dev/null 2>&1; then
        cpupower frequency-set -g performance 2>/dev/null || log_warn "无法设置CPU频率调节器"
    fi
    
    # 禁用CPU节能功能
    echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
    
    # 设置CPU亲和性脚本
    cat > /usr/local/bin/xiaozhi-cpu-affinity.sh << 'EOF'
#!/bin/bash
# 小智ESP32服务器CPU亲和性设置

# 获取Docker容器PID
CONTAINER_PID=$(docker inspect xiaozhi-esp32-server --format '{{.State.Pid}}' 2>/dev/null)

if [[ -n "$CONTAINER_PID" && "$CONTAINER_PID" != "0" ]]; then
    # 将主服务绑定到CPU 0-2
    taskset -cp 0-2 $CONTAINER_PID 2>/dev/null || true
    echo "已将xiaozhi-esp32-server绑定到CPU 0-2"
    
    # 将其他Docker服务绑定到CPU 3
    for service in xiaozhi-esp32-server-web xiaozhi-esp32-server-db xiaozhi-esp32-server-redis; do
        SERVICE_PID=$(docker inspect $service --format '{{.State.Pid}}' 2>/dev/null)
        if [[ -n "$SERVICE_PID" && "$SERVICE_PID" != "0" ]]; then
            taskset -cp 3 $SERVICE_PID 2>/dev/null || true
            echo "已将$service绑定到CPU 3"
        fi
    done
else
    echo "未找到xiaozhi-esp32-server容器"
fi
EOF

    chmod +x /usr/local/bin/xiaozhi-cpu-affinity.sh
    log_info "CPU调度器优化完成"
}

# 内存优化
optimize_memory() {
    log_step "优化内存管理..."
    
    # 创建内存优化脚本
    cat > /usr/local/bin/xiaozhi-memory-optimize.sh << 'EOF'
#!/bin/bash
# 小智ESP32服务器内存优化脚本

# 清理页面缓存（谨慎使用）
sync
echo 1 > /proc/sys/vm/drop_caches

# 内存碎片整理
echo 1 > /proc/sys/vm/compact_memory 2>/dev/null || true

# 检查内存使用情况
MEMORY_USAGE=$(free | awk '/^Mem:/{printf "%.1f", $3/$2 * 100}')
echo "当前内存使用率: ${MEMORY_USAGE}%"

# 如果内存使用率超过85%，发出警告
if (( $(echo "$MEMORY_USAGE > 85" | bc -l) )); then
    echo "警告: 内存使用率过高 (${MEMORY_USAGE}%)"
    # 可以在这里添加告警逻辑
fi
EOF

    chmod +x /usr/local/bin/xiaozhi-memory-optimize.sh
    
    # 创建定时任务
    cat > /etc/cron.d/xiaozhi-memory-optimize << 'EOF'
# 每30分钟执行一次内存优化
*/30 * * * * root /usr/local/bin/xiaozhi-memory-optimize.sh >> /var/log/xiaozhi-memory.log 2>&1
EOF

    log_info "内存优化配置完成"
}

# 磁盘I/O优化
optimize_disk_io() {
    log_step "优化磁盘I/O..."
    
    # 获取根分区设备
    ROOT_DEVICE=$(df / | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//')
    
    if [[ -n "$ROOT_DEVICE" ]]; then
        # 设置I/O调度器
        echo mq-deadline > /sys/block/$(basename $ROOT_DEVICE)/queue/scheduler 2>/dev/null || true
        
        # 优化读取预读
        echo 256 > /sys/block/$(basename $ROOT_DEVICE)/queue/read_ahead_kb 2>/dev/null || true
        
        # 设置队列深度
        echo 32 > /sys/block/$(basename $ROOT_DEVICE)/queue/nr_requests 2>/dev/null || true
        
        log_info "磁盘I/O优化完成 (设备: $ROOT_DEVICE)"
    else
        log_warn "无法确定根分区设备，跳过磁盘I/O优化"
    fi
}

# Docker优化
optimize_docker() {
    log_step "优化Docker配置..."
    
    # 创建Docker daemon配置
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ],
    "default-ulimits": {
        "nofile": {
            "Name": "nofile",
            "Hard": 65536,
            "Soft": 65536
        },
        "nproc": {
            "Name": "nproc",
            "Hard": 32768,
            "Soft": 32768
        }
    },
    "max-concurrent-downloads": 6,
    "max-concurrent-uploads": 6,
    "live-restore": true
}
EOF

    # 重启Docker服务
    systemctl restart docker
    log_info "Docker配置优化完成"
}

# 创建监控脚本
create_monitoring_script() {
    log_step "创建系统监控脚本..."
    
    cat > /usr/local/bin/xiaozhi-monitor.sh << 'EOF'
#!/bin/bash
# 小智ESP32服务器监控脚本

# 获取系统信息
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEMORY_USAGE=$(free | awk '/^Mem:/{printf "%.1f", $3/$2 * 100}')
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | tr -d '%')

# 获取Docker容器状态
CONTAINER_STATUS=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep xiaozhi)

# 输出监控信息
echo "=== 小智ESP32服务器监控报告 $(date) ==="
echo "CPU使用率: ${CPU_USAGE}%"
echo "内存使用率: ${MEMORY_USAGE}%"
echo "系统负载: ${LOAD_AVG}"
echo "磁盘使用率: ${DISK_USAGE}%"
echo ""
echo "Docker容器状态:"
echo "$CONTAINER_STATUS"
echo ""

# 检查告警条件
ALERT=false

if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "⚠️  警告: CPU使用率过高 (${CPU_USAGE}%)"
    ALERT=true
fi

if (( $(echo "$MEMORY_USAGE > 85" | bc -l) )); then
    echo "⚠️  警告: 内存使用率过高 (${MEMORY_USAGE}%)"
    ALERT=true
fi

if (( $(echo "$LOAD_AVG > 4" | bc -l) )); then
    echo "⚠️  警告: 系统负载过高 (${LOAD_AVG})"
    ALERT=true
fi

if [[ $DISK_USAGE -gt 85 ]]; then
    echo "⚠️  警告: 磁盘使用率过高 (${DISK_USAGE}%)"
    ALERT=true
fi

if [[ $ALERT == false ]]; then
    echo "✅ 系统状态正常"
fi

echo "=================================="
EOF

    chmod +x /usr/local/bin/xiaozhi-monitor.sh
    
    # 创建定时监控任务
    cat > /etc/cron.d/xiaozhi-monitor << 'EOF'
# 每5分钟执行一次系统监控
*/5 * * * * root /usr/local/bin/xiaozhi-monitor.sh >> /var/log/xiaozhi-monitor.log 2>&1
EOF

    log_info "监控脚本创建完成"
}

# 创建启动优化脚本
create_startup_script() {
    log_step "创建启动优化脚本..."
    
    cat > /usr/local/bin/xiaozhi-startup-optimize.sh << 'EOF'
#!/bin/bash
# 小智ESP32服务器启动优化脚本

# 等待系统完全启动
sleep 30

# 设置CPU亲和性
/usr/local/bin/xiaozhi-cpu-affinity.sh

# 执行内存优化
/usr/local/bin/xiaozhi-memory-optimize.sh

# 输出启动完成信息
echo "$(date): 小智ESP32服务器启动优化完成" >> /var/log/xiaozhi-startup.log
EOF

    chmod +x /usr/local/bin/xiaozhi-startup-optimize.sh
    
    # 创建systemd服务
    cat > /etc/systemd/system/xiaozhi-startup-optimize.service << 'EOF'
[Unit]
Description=XiaoZhi ESP32 Server Startup Optimization
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/xiaozhi-startup-optimize.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable xiaozhi-startup-optimize.service
    
    log_info "启动优化脚本创建完成"
}

# 性能测试
run_performance_test() {
    log_step "运行性能测试..."
    
    # 创建性能测试脚本
    cat > /tmp/xiaozhi-performance-test.sh << 'EOF'
#!/bin/bash
echo "=== 小智ESP32服务器性能测试 ==="

# CPU性能测试
echo "CPU性能测试..."
CPU_SCORE=$(timeout 10s yes > /dev/null & sleep 1; ps -o %cpu -p $! | tail -1 | tr -d ' ')
echo "CPU单核性能: ${CPU_SCORE}%"

# 内存性能测试
echo "内存性能测试..."
MEMORY_SPEED=$(dd if=/dev/zero of=/tmp/test bs=1M count=100 2>&1 | grep copied | awk '{print $(NF-1) " " $NF}')
echo "内存写入速度: $MEMORY_SPEED"
rm -f /tmp/test

# 磁盘性能测试
echo "磁盘性能测试..."
DISK_WRITE=$(dd if=/dev/zero of=/tmp/disktest bs=1M count=100 2>&1 | grep copied | awk '{print $(NF-1) " " $NF}')
DISK_READ=$(dd if=/tmp/disktest of=/dev/null bs=1M 2>&1 | grep copied | awk '{print $(NF-1) " " $NF}')
echo "磁盘写入速度: $DISK_WRITE"
echo "磁盘读取速度: $DISK_READ"
rm -f /tmp/disktest

# 网络性能测试（如果有网络连接）
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    echo "网络连接正常"
else
    echo "网络连接异常"
fi

echo "=== 性能测试完成 ==="
EOF

    chmod +x /tmp/xiaozhi-performance-test.sh
    /tmp/xiaozhi-performance-test.sh
    rm -f /tmp/xiaozhi-performance-test.sh
}

# 主函数
main() {
    log_info "开始小智ESP32专用服务器系统优化..."
    
    # 检查权限
    check_root
    
    # 系统检查
    check_system_info
    
    # 备份配置
    backup_configs
    
    # 执行优化
    optimize_kernel_parameters
    optimize_system_limits
    optimize_cpu_scheduler
    optimize_memory
    optimize_disk_io
    optimize_docker
    
    # 创建监控和管理脚本
    create_monitoring_script
    create_startup_script
    
    # 性能测试
    run_performance_test
    
    log_info "系统优化完成！"
    log_info "建议重启系统以确保所有优化生效"
    log_info "重启后可以运行以下命令查看优化效果："
    log_info "  - /usr/local/bin/xiaozhi-monitor.sh"
    log_info "  - /usr/local/bin/xiaozhi-cpu-affinity.sh"
    log_info "  - /usr/local/bin/xiaozhi-memory-optimize.sh"
}

# 运行主函数
main "$@"