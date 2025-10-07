#!/bin/bash
# Xiaozhi ESP32 Server - 监控系统部署脚本
# 部署Prometheus、Grafana、AlertManager等监控组件

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 创建监控命名空间
create_monitoring_namespace() {
    log_info "创建监控命名空间..."
    
    cat > k8s/monitoring/namespace.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    name: monitoring
EOF

    kubectl apply -f k8s/monitoring/namespace.yaml
    log_success "监控命名空间创建完成"
}

# 部署Prometheus
deploy_prometheus() {
    log_info "部署Prometheus..."
    
    # 创建Prometheus配置
    cat > k8s/monitoring/prometheus-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'xiaozhi-cluster'
        replica: 'prometheus-1'
    
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager:9093
        scheme: http
        timeout: 10s
        api_version: v1
    
    scrape_configs:
    # Kubernetes API Server
    - job_name: 'kubernetes-apiservers'
      kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
          - default
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        insecure_skip_verify: true
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https
    
    # Kubernetes Nodes
    - job_name: 'kubernetes-nodes'
      kubernetes_sd_configs:
      - role: node
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        insecure_skip_verify: true
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics
    
    # Kubernetes Pods
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
    
    # Xiaozhi Services
    - job_name: 'xiaozhi-services'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - xiaozhi-system
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: (vad-service|asr-service|llm-service|tts-service|intelligent-load-balancer)
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
    
    # Redis Cluster
    - job_name: 'redis-cluster'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - xiaozhi-system
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: redis-cluster
      - source_labels: [__address__]
        action: replace
        regex: ([^:]+)(?::\d+)?
        replacement: $1:9121
        target_label: __address__
    
    # Node Exporter
    - job_name: 'node-exporter'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: node-exporter
      - source_labels: [__address__]
        action: replace
        regex: ([^:]+)(?::\d+)?
        replacement: $1:9100
        target_label: __address__
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  xiaozhi-alerts.yml: |
    groups:
    - name: xiaozhi-performance
      interval: 30s
      rules:
      # CPU使用率告警
      - alert: HighCPUUsage
        expr: |
          (
            rate(container_cpu_usage_seconds_total{container!="POD",container!="",pod=~".*-(vad|asr|llm|tts)-.*"}[5m]) * 100
          ) > 80
        for: 2m
        labels:
          severity: warning
          service: "{{ $labels.container }}"
        annotations:
          summary: "{{ $labels.pod }} CPU使用率过高"
          description: "Pod {{ $labels.pod }} 的CPU使用率已超过80%，当前值: {{ $value }}%"
      
      # 内存使用率告警
      - alert: HighMemoryUsage
        expr: |
          (
            container_memory_working_set_bytes{container!="POD",container!="",pod=~".*-(vad|asr|llm|tts)-.*"} 
            / container_spec_memory_limit_bytes{container!="POD",container!=""} * 100
          ) > 85
        for: 2m
        labels:
          severity: warning
          service: "{{ $labels.container }}"
        annotations:
          summary: "{{ $labels.pod }} 内存使用率过高"
          description: "Pod {{ $labels.pod }} 的内存使用率已超过85%，当前值: {{ $value }}%"
      
      # 响应时间告警
      - alert: HighResponseTime
        expr: |
          histogram_quantile(0.95, 
            rate(http_request_duration_seconds_bucket{job="xiaozhi-services"}[5m])
          ) > 1
        for: 1m
        labels:
          severity: critical
          service: "{{ $labels.job }}"
        annotations:
          summary: "{{ $labels.job }} 响应时间过长"
          description: "服务 {{ $labels.job }} 的95%响应时间超过1秒，当前值: {{ $value }}s"
      
      # 错误率告警
      - alert: HighErrorRate
        expr: |
          (
            rate(http_requests_total{job="xiaozhi-services",status=~"5.."}[5m]) 
            / rate(http_requests_total{job="xiaozhi-services"}[5m]) * 100
          ) > 5
        for: 2m
        labels:
          severity: critical
          service: "{{ $labels.job }}"
        annotations:
          summary: "{{ $labels.job }} 错误率过高"
          description: "服务 {{ $labels.job }} 的错误率超过5%，当前值: {{ $value }}%"
      
      # 服务不可用告警
      - alert: ServiceDown
        expr: up{job="xiaozhi-services"} == 0
        for: 1m
        labels:
          severity: critical
          service: "{{ $labels.job }}"
        annotations:
          summary: "{{ $labels.instance }} 服务不可用"
          description: "服务实例 {{ $labels.instance }} 已下线超过1分钟"
      
      # Redis连接数告警
      - alert: RedisHighConnections
        expr: redis_connected_clients > 1000
        for: 2m
        labels:
          severity: warning
          service: "redis"
        annotations:
          summary: "Redis连接数过高"
          description: "Redis实例 {{ $labels.instance }} 的连接数超过1000，当前值: {{ $value }}"
      
      # Redis内存使用率告警
      - alert: RedisHighMemoryUsage
        expr: |
          (
            redis_memory_used_bytes / redis_memory_max_bytes * 100
          ) > 80
        for: 2m
        labels:
          severity: warning
          service: "redis"
        annotations:
          summary: "Redis内存使用率过高"
          description: "Redis实例 {{ $labels.instance }} 的内存使用率超过80%，当前值: {{ $value }}%"
      
      # GPU使用率告警
      - alert: HighGPUUsage
        expr: |
          nvidia_gpu_utilization > 90
        for: 5m
        labels:
          severity: warning
          service: "gpu"
        annotations:
          summary: "GPU使用率过高"
          description: "GPU {{ $labels.gpu }} 使用率超过90%，当前值: {{ $value }}%"
      
      # 磁盘空间告警
      - alert: DiskSpaceLow
        expr: |
          (
            node_filesystem_avail_bytes{mountpoint="/",fstype!="tmpfs"} 
            / node_filesystem_size_bytes{mountpoint="/",fstype!="tmpfs"} * 100
          ) < 20
        for: 5m
        labels:
          severity: warning
          service: "system"
        annotations:
          summary: "磁盘空间不足"
          description: "节点 {{ $labels.instance }} 的磁盘空间剩余不足20%，当前值: {{ $value }}%"
    
    - name: xiaozhi-business
      interval: 30s
      rules:
      # 设备连接数告警
      - alert: TooManyDevices
        expr: xiaozhi_connected_devices > 100
        for: 1m
        labels:
          severity: warning
          service: "business"
        annotations:
          summary: "连接设备数过多"
          description: "当前连接设备数: {{ $value }}，已超过预期的100台"
      
      # 队列积压告警
      - alert: QueueBacklog
        expr: xiaozhi_queue_size > 1000
        for: 2m
        labels:
          severity: warning
          service: "{{ $labels.service }}"
        annotations:
          summary: "{{ $labels.service }} 队列积压"
          description: "服务 {{ $labels.service }} 的队列大小: {{ $value }}，存在积压"
      
      # 缓存命中率低告警
      - alert: LowCacheHitRate
        expr: |
          (
            rate(xiaozhi_cache_hits_total[5m]) 
            / (rate(xiaozhi_cache_hits_total[5m]) + rate(xiaozhi_cache_misses_total[5m])) * 100
          ) < 70
        for: 5m
        labels:
          severity: warning
          service: "{{ $labels.service }}"
        annotations:
          summary: "{{ $labels.service }} 缓存命中率低"
          description: "服务 {{ $labels.service }} 的缓存命中率: {{ $value }}%，低于70%"
EOF

    # 部署Prometheus
    cat > k8s/monitoring/prometheus-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus/'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time=30d'
        - '--web.enable-lifecycle'
        - '--web.enable-admin-api'
        ports:
        - containerPort: 9090
          name: web
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus/
        - name: prometheus-rules
          mountPath: /etc/prometheus/rules/
        - name: prometheus-storage
          mountPath: /prometheus/
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9090
          initialDelaySeconds: 30
          timeoutSeconds: 30
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9090
          initialDelaySeconds: 30
          timeoutSeconds: 30
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-rules
        configMap:
          name: prometheus-rules
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-storage
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
  labels:
    app: prometheus
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
    name: web
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-storage
  namespace: monitoring
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups:
  - extensions
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: monitoring
EOF

    kubectl apply -f k8s/monitoring/prometheus-config.yaml
    kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
    
    log_success "Prometheus部署完成"
}

# 部署Grafana
deploy_grafana() {
    log_info "部署Grafana..."
    
    # 创建Grafana配置
    cat > k8s/monitoring/grafana-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: monitoring
data:
  prometheus.yaml: |
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus:9090
      access: proxy
      isDefault: true
      editable: true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards-config
  namespace: monitoring
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
    - name: 'default'
      orgId: 1
      folder: ''
      type: file
      disableDeletion: false
      updateIntervalSeconds: 10
      allowUiUpdates: true
      options:
        path: /var/lib/grafana/dashboards
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-xiaozhi
  namespace: monitoring
data:
  xiaozhi-overview.json: |
    {
      "dashboard": {
        "id": null,
        "title": "Xiaozhi ESP32 Server Overview",
        "tags": ["xiaozhi"],
        "style": "dark",
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "总体QPS",
            "type": "stat",
            "targets": [
              {
                "expr": "sum(rate(http_requests_total{job=\"xiaozhi-services\"}[5m]))",
                "legendFormat": "QPS"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "color": {
                  "mode": "thresholds"
                },
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 80},
                    {"color": "red", "value": 100}
                  ]
                }
              }
            },
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "平均响应时间",
            "type": "stat",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"xiaozhi-services\"}[5m]))",
                "legendFormat": "P95"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "s",
                "color": {
                  "mode": "thresholds"
                },
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 0.5},
                    {"color": "red", "value": 1.0}
                  ]
                }
              }
            },
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
          },
          {
            "id": 3,
            "title": "错误率",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(http_requests_total{job=\"xiaozhi-services\",status=~\"5..\"}[5m]) / rate(http_requests_total{job=\"xiaozhi-services\"}[5m]) * 100",
                "legendFormat": "Error Rate"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "percent",
                "color": {
                  "mode": "thresholds"
                },
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 1},
                    {"color": "red", "value": 5}
                  ]
                }
              }
            },
            "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
          },
          {
            "id": 4,
            "title": "连接设备数",
            "type": "stat",
            "targets": [
              {
                "expr": "xiaozhi_connected_devices",
                "legendFormat": "Devices"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "color": {
                  "mode": "thresholds"
                },
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 80},
                    {"color": "red", "value": 100}
                  ]
                }
              }
            },
            "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0}
          }
        ],
        "time": {
          "from": "now-1h",
          "to": "now"
        },
        "refresh": "5s"
      }
    }
EOF

    # 部署Grafana
    cat > k8s/monitoring/grafana-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.0.0
        ports:
        - containerPort: 3000
          name: web
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "xiaozhi123"
        - name: GF_USERS_ALLOW_SIGN_UP
          value: "false"
        - name: GF_SERVER_ROOT_URL
          value: "http://localhost:3000/"
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-datasources
          mountPath: /etc/grafana/provisioning/datasources
        - name: grafana-dashboards-config
          mountPath: /etc/grafana/provisioning/dashboards
        - name: grafana-dashboards
          mountPath: /var/lib/grafana/dashboards
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          timeoutSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 5
          timeoutSeconds: 10
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-storage
      - name: grafana-datasources
        configMap:
          name: grafana-datasources
      - name: grafana-dashboards-config
        configMap:
          name: grafana-dashboards-config
      - name: grafana-dashboards
        configMap:
          name: grafana-dashboard-xiaozhi
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
    name: web
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-storage
  namespace: monitoring
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
EOF

    kubectl apply -f k8s/monitoring/grafana-config.yaml
    kubectl apply -f k8s/monitoring/grafana-deployment.yaml
    
    log_success "Grafana部署完成"
}

# 部署AlertManager
deploy_alertmanager() {
    log_info "部署AlertManager..."
    
    # 创建AlertManager配置
    cat > k8s/monitoring/alertmanager-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'alertmanager@xiaozhi.com'
      smtp_auth_username: 'alertmanager@xiaozhi.com'
      smtp_auth_password: 'password'
    
    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
      routes:
      - match:
          severity: critical
        receiver: 'critical-alerts'
        group_wait: 5s
        repeat_interval: 30m
      - match:
          severity: warning
        receiver: 'warning-alerts'
        group_wait: 10s
        repeat_interval: 1h
    
    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://webhook-service:8080/alerts'
        send_resolved: true
    
    - name: 'critical-alerts'
      email_configs:
      - to: 'admin@xiaozhi.com'
        subject: '[CRITICAL] Xiaozhi Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Labels: {{ range .Labels.SortedPairs }}{{ .Name }}={{ .Value }} {{ end }}
          {{ end }}
      webhook_configs:
      - url: 'http://webhook-service:8080/critical-alerts'
        send_resolved: true
    
    - name: 'warning-alerts'
      email_configs:
      - to: 'ops@xiaozhi.com'
        subject: '[WARNING] Xiaozhi Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Labels: {{ range .Labels.SortedPairs }}{{ .Name }}={{ .Value }} {{ end }}
          {{ end }}
    
    inhibit_rules:
    - source_match:
        severity: 'critical'
      target_match:
        severity: 'warning'
      equal: ['alertname', 'cluster', 'service']
EOF

    # 部署AlertManager
    cat > k8s/monitoring/alertmanager-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:v0.25.0
        args:
        - '--config.file=/etc/alertmanager/alertmanager.yml'
        - '--storage.path=/alertmanager'
        - '--web.external-url=http://localhost:9093'
        ports:
        - containerPort: 9093
          name: web
        volumeMounts:
        - name: alertmanager-config
          mountPath: /etc/alertmanager
        - name: alertmanager-storage
          mountPath: /alertmanager
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9093
          initialDelaySeconds: 30
          timeoutSeconds: 30
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9093
          initialDelaySeconds: 5
          timeoutSeconds: 10
      volumes:
      - name: alertmanager-config
        configMap:
          name: alertmanager-config
      - name: alertmanager-storage
        persistentVolumeClaim:
          claimName: alertmanager-storage
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  selector:
    app: alertmanager
  ports:
  - port: 9093
    targetPort: 9093
    name: web
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: alertmanager-storage
  namespace: monitoring
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd
EOF

    kubectl apply -f k8s/monitoring/alertmanager-config.yaml
    kubectl apply -f k8s/monitoring/alertmanager-deployment.yaml
    
    log_success "AlertManager部署完成"
}

# 部署Node Exporter
deploy_node_exporter() {
    log_info "部署Node Exporter..."
    
    cat > k8s/monitoring/node-exporter.yaml << 'EOF'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
  labels:
    app: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9100"
    spec:
      hostPID: true
      hostIPC: true
      hostNetwork: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:v1.6.0
        args:
        - '--path.procfs=/host/proc'
        - '--path.sysfs=/host/sys'
        - '--path.rootfs=/host/root'
        - '--collector.filesystem.ignored-mount-points=^/(dev|proc|sys|var/lib/docker/.+)($|/)'
        - '--collector.filesystem.ignored-fs-types=^(autofs|binfmt_misc|cgroup|configfs|debugfs|devpts|devtmpfs|fusectl|hugetlbfs|mqueue|overlay|proc|procfs|pstore|rpc_pipefs|securityfs|sysfs|tracefs)$'
        ports:
        - containerPort: 9100
          hostPort: 9100
          name: metrics
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        - name: root
          mountPath: /host/root
          mountPropagation: HostToContainer
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      - name: root
        hostPath:
          path: /
      tolerations:
      - operator: Exists
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: node-exporter
  namespace: monitoring
  labels:
    app: node-exporter
spec:
  selector:
    app: node-exporter
  ports:
  - port: 9100
    targetPort: 9100
    name: metrics
  type: ClusterIP
EOF

    kubectl apply -f k8s/monitoring/node-exporter.yaml
    log_success "Node Exporter部署完成"
}

# 创建监控目录
create_monitoring_dirs() {
    log_info "创建监控配置目录..."
    mkdir -p k8s/monitoring
    log_success "监控配置目录创建完成"
}

# 检查部署状态
check_deployment_status() {
    log_info "检查监控组件部署状态..."
    
    echo ""
    echo "等待Pod启动..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=300s
    kubectl wait --for=condition=ready pod -l app=alertmanager -n monitoring --timeout=300s
    
    echo ""
    echo "监控组件状态:"
    kubectl get pods -n monitoring
    
    echo ""
    echo "监控服务状态:"
    kubectl get svc -n monitoring
    
    echo ""
    echo "访问信息:"
    echo "Prometheus: kubectl port-forward -n monitoring svc/prometheus 9090:9090"
    echo "Grafana: kubectl port-forward -n monitoring svc/grafana 3000:3000 (admin/xiaozhi123)"
    echo "AlertManager: kubectl port-forward -n monitoring svc/alertmanager 9093:9093"
    
    log_success "监控系统部署完成！"
}

# 主函数
main() {
    echo "Xiaozhi ESP32 Server - 监控系统部署"
    echo "===================================="
    
    # 选择部署选项
    echo ""
    echo "请选择部署选项:"
    echo "1) 完整监控系统部署 (推荐)"
    echo "2) 仅部署Prometheus"
    echo "3) 仅部署Grafana"
    echo "4) 仅部署AlertManager"
    echo "5) 仅部署Node Exporter"
    echo "6) 检查部署状态"
    echo "7) 退出"
    
    read -p "请输入选择 (1-7): " choice
    
    case $choice in
        1)
            log_info "开始完整监控系统部署..."
            create_monitoring_dirs
            create_monitoring_namespace
            deploy_prometheus
            deploy_grafana
            deploy_alertmanager
            deploy_node_exporter
            check_deployment_status
            ;;
        2)
            create_monitoring_dirs
            create_monitoring_namespace
            deploy_prometheus
            ;;
        3)
            create_monitoring_dirs
            create_monitoring_namespace
            deploy_grafana
            ;;
        4)
            create_monitoring_dirs
            create_monitoring_namespace
            deploy_alertmanager
            ;;
        5)
            create_monitoring_dirs
            create_monitoring_namespace
            deploy_node_exporter
            ;;
        6)
            check_deployment_status
            ;;
        7)
            log_info "退出监控部署脚本"
            exit 0
            ;;
        *)
            log_error "无效选择，请重新运行脚本"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"