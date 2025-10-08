#!/bin/bash

# 设置默认值
REDIS_PASSWORD_ARG=""
if [ -n "${SPRING_DATA_REDIS_PASSWORD}" ]; then
    REDIS_PASSWORD_ARG="--spring.data.redis.password=${SPRING_DATA_REDIS_PASSWORD}"
fi

# 启动Java后端（docker内监听8003端口）
java -jar /app/xiaozhi-esp32-api.jar \
  --server.port=8003 \
  --spring.datasource.druid.url="${SPRING_DATASOURCE_DRUID_URL}" \
  --spring.datasource.druid.username="${SPRING_DATASOURCE_DRUID_USERNAME}" \
  --spring.datasource.druid.password="${SPRING_DATASOURCE_DRUID_PASSWORD}" \
  --spring.data.redis.host="${SPRING_DATA_REDIS_HOST}" \
  --spring.data.redis.port="${SPRING_DATA_REDIS_PORT}" \
  ${REDIS_PASSWORD_ARG} &

# 等待Java应用启动
sleep 5

# 启动Nginx（前台运行保持容器存活）
exec nginx -g "daemon off;"