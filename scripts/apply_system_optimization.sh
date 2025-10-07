#!/bin/bash

# P0级别系统优化应用脚本
# 目标：优化系统性能，支持100台设备并发访问

set -e

echo "=== 开始应用P0级别系统优化 ==="

# 检查是否为root用户
if [[ $EUID -ne 0 ]]; then
   echo "此脚本需要root权限运行"
   exit 1
fi

# 1. 应用内核参数优化
echo "1. 应用内核参数优化..."
cat > /etc/sysctl.d/99-xiaozhi-optimization.conf << 'EOF'
# P0级别系统优化 - 内核参数

# 网络优化
net.core.somaxconn = 65536
net.core.netdev_max_backlog = 5000
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216

# TCP参数优化
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 0
net.ipv4.tcp_sack = 1
net.ipv4.tcp_fack = 1
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 3

# 连接跟踪优化
net.netfilter.nf_conntrack_max = 1048576
net.netfilter.nf_conntrack_tcp_timeout_established = 7200

# 内存管理优化
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.dirty_expire_centisecs = 3000
vm.dirty_writeback_centisecs = 500
vm.overcommit_memory = 1
vm.max_map_count = 262144

# 文件系统优化
fs.file-max = 1048576
fs.nr_open = 1048576
fs.inotify.max_user_watches = 524288
fs.aio-max-nr = 1048576

# 进程和线程优化
kernel.pid_max = 4194304
kernel.threads-max = 1048576
kernel.sched_migration_cost_ns = 5000000

# 安全优化
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
EOF

# 应用内核参数
sysctl -p /etc/sysctl.d/99-xiaozhi-optimization.conf

# 2. 设置系统限制
echo "2. 设置系统限制..."
cat > /etc/security/limits.d/99-xiaozhi-optimization.conf << 'EOF'
# P0级别系统优化 - 用户限制
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited
root soft nofile 65536
root hard nofile 65536
root soft nproc 32768
root hard nproc 32768
root soft memlock unlimited
root hard memlock unlimited
EOF

# 3. 设置systemd默认限制
echo "3. 设置systemd默认限制..."
mkdir -p /etc/systemd/system.conf.d
cat > /etc/systemd/system.conf.d/99-xiaozhi-optimization.conf << 'EOF'
[Manager]
DefaultLimitNOFILE=65536
DefaultLimitNPROC=32768
DefaultLimitMEMLOCK=infinity
EOF

# 4. CPU优化
echo "4. 应用CPU优化..."
# 设置CPU调度器为性能模式
if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
    echo "设置CPU调度器为性能模式..."
    echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
    
    # 设置最小频率为最大频率
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
        if [ -f "$cpu" ]; then
            max_freq=$(cat "${cpu%min_freq}scaling_max_freq")
            echo "$max_freq" > "$cpu" 2>/dev/null || true
        fi
    done
fi

# 禁用CPU节能功能
echo "禁用CPU节能功能..."
# 禁用C-states
for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    [ -f "$state" ] && echo 1 > "$state" 2>/dev/null || true
done

# 5. 磁盘IO优化
echo "5. 应用磁盘IO优化..."
# 设置IO调度器
for scheduler in /sys/block/*/queue/scheduler; do
    if [ -f "$scheduler" ]; then
        echo noop > "$scheduler" 2>/dev/null || echo none > "$scheduler" 2>/dev/null || true
    fi
done

# 设置预读大小
for readahead in /sys/block/*/queue/read_ahead_kb; do
    [ -f "$readahead" ] && echo 256 > "$readahead" 2>/dev/null || true
done

# 6. 网络接口优化
echo "6. 应用网络接口优化..."
# 获取主网络接口
MAIN_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -n1)

if [ -n "$MAIN_INTERFACE" ] && command -v ethtool >/dev/null 2>&1; then
    echo "优化网络接口: $MAIN_INTERFACE"
    
    # 设置环形缓冲区大小
    ethtool -G "$MAIN_INTERFACE" rx 4096 tx 4096 2>/dev/null || true
    
    # 设置中断合并
    ethtool -C "$MAIN_INTERFACE" rx-usecs 50 tx-usecs 50 2>/dev/null || true
    
    # 启用RSS（如果支持）
    ethtool -X "$MAIN_INTERFACE" equal 4 2>/dev/null || true
fi

# 7. 内存优化
echo "7. 应用内存优化..."
# 禁用透明大页
if [ -f "/sys/kernel/mm/transparent_hugepage/enabled" ]; then
    echo never > /sys/kernel/mm/transparent_hugepage/enabled
fi

if [ -f "/sys/kernel/mm/transparent_hugepage/defrag" ]; then
    echo never > /sys/kernel/mm/transparent_hugepage/defrag
fi

# 8. 创建优化验证脚本
echo "8. 创建验证脚本..."
cat > /usr/local/bin/xiaozhi-optimization-check.sh << 'EOF'
#!/bin/bash

echo "=== 小智系统优化验证 ==="
echo "时间: $(date)"
echo ""

echo "=== 系统限制 ==="
echo "文件描述符限制: $(ulimit -n)"
echo "进程限制: $(ulimit -u)"
echo "内存锁定限制: $(ulimit -l)"
echo ""

echo "=== CPU状态 ==="
if [ -f "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]; then
    echo "CPU调度器: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "CPU最小频率: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)"
    echo "CPU最大频率: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
else
    echo "CPU频率控制不可用"
fi
echo ""

echo "=== 内存状态 ==="
echo "交换倾向: $(cat /proc/sys/vm/swappiness)"
echo "脏页比例: $(cat /proc/sys/vm/dirty_ratio)"
echo "最大内存映射: $(cat /proc/sys/vm/max_map_count)"
echo ""

echo "=== 网络状态 ==="
echo "监听队列长度: $(cat /proc/sys/net/core/somaxconn)"
echo "最大文件句柄: $(cat /proc/sys/fs/file-max)"
echo "连接跟踪最大数: $(cat /proc/sys/net/netfilter/nf_conntrack_max 2>/dev/null || echo '不可用')"
echo ""

echo "=== 磁盘IO状态 ==="
for scheduler in /sys/block/*/queue/scheduler; do
    if [ -f "$scheduler" ]; then
        device=$(echo "$scheduler" | cut -d'/' -f4)
        current=$(grep -o '\[.*\]' "$scheduler" | tr -d '[]')
        echo "磁盘 $device IO调度器: $current"
    fi
done
echo ""

echo "=== 透明大页状态 ==="
if [ -f "/sys/kernel/mm/transparent_hugepage/enabled" ]; then
    echo "透明大页: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
else
    echo "透明大页: 不可用"
fi
echo ""

echo "=== 当前负载 ==="
echo "负载平均: $(uptime | awk -F'load average:' '{print $2}')"
echo "内存使用: $(free -h | grep Mem | awk '{print $3"/"$2" ("$3/$2*100"%)"}')"
echo "磁盘使用: $(df -h / | tail -1 | awk '{print $3"/"$2" ("$5")"}')"
echo ""

echo "=== 网络连接统计 ==="
echo "TCP连接数: $(ss -t | wc -l)"
echo "监听端口数: $(ss -tln | wc -l)"
echo ""

echo "验证完成！"
EOF

chmod +x /usr/local/bin/xiaozhi-optimization-check.sh

# 9. 创建开机自启动服务
echo "9. 创建开机自启动服务..."
cat > /etc/systemd/system/xiaozhi-optimization.service << 'EOF'
[Unit]
Description=XiaoZhi System Optimization
After=network.target

[Service]
Type=oneshot
ExecStart=/root/xiaozhi-server/scripts/apply_system_optimization.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
systemctl daemon-reload
systemctl enable xiaozhi-optimization.service

# 10. 应用Docker优化
echo "10. 应用Docker优化..."
if command -v docker >/dev/null 2>&1; then
    # 创建Docker daemon配置
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << 'EOF'
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "soft": 65536,
      "hard": 65536
    },
    "nproc": {
      "soft": 32768,
      "hard": 32768
    }
  },
  "live-restore": true,
  "userland-proxy": false,
  "experimental": false,
  "metrics-addr": "127.0.0.1:9323",
  "iptables": true,
  "ip-forward": true,
  "ip-masq": true,
  "ipv6": false
}
EOF

    # 重启Docker服务
    systemctl restart docker 2>/dev/null || true
fi

echo ""
echo "=== P0级别系统优化应用完成 ==="
echo ""
echo "优化内容："
echo "✓ 内核参数优化（网络、内存、文件系统）"
echo "✓ 系统限制优化（文件描述符、进程数）"
echo "✓ CPU性能优化（调度器、频率）"
echo "✓ 磁盘IO优化（调度器、预读）"
echo "✓ 网络接口优化（缓冲区、中断）"
echo "✓ 内存优化（透明大页、交换）"
echo "✓ Docker优化（存储驱动、日志）"
echo "✓ 开机自启动配置"
echo ""
echo "验证命令: xiaozhi-optimization-check.sh"
echo ""
echo "注意：某些优化需要重启系统后生效"
echo "建议运行验证脚本检查优化状态"