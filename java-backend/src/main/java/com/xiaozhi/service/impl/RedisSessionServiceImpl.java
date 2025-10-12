package com.xiaozhi.service.impl;

import com.xiaozhi.service.RedisSessionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Redis会话管理服务实现类
 */
@Service
public class RedisSessionServiceImpl implements RedisSessionService {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    private static final String DEVICE_SESSIONS_KEY = "device:sessions:";
    private static final String CURRENT_SESSION_KEY = "device:current_session:";
    private static final String SESSION_INFO_KEY = "session:info:";
    private static final long SESSION_EXPIRE_TIME = 24 * 60 * 60; // 24小时过期
    
    @Override
    public boolean createSession(String deviceId, String sessionId) {
        try {
            String sessionsKey = DEVICE_SESSIONS_KEY + deviceId;
            String sessionInfoKey = SESSION_INFO_KEY + deviceId + ":" + sessionId;
            
            // 添加会话到设备会话列表
            redisTemplate.opsForSet().add(sessionsKey, sessionId);
            redisTemplate.expire(sessionsKey, SESSION_EXPIRE_TIME, TimeUnit.SECONDS);
            
            // 创建会话信息
            Map<String, Object> sessionInfo = new HashMap<>();
            sessionInfo.put("sessionId", sessionId);
            sessionInfo.put("deviceId", deviceId);
            sessionInfo.put("createdAt", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            sessionInfo.put("lastActiveAt", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            sessionInfo.put("messageCount", 0);
            
            redisTemplate.opsForHash().putAll(sessionInfoKey, sessionInfo);
            redisTemplate.expire(sessionInfoKey, SESSION_EXPIRE_TIME, TimeUnit.SECONDS);
            
            // 设置为当前会话
            setCurrentSession(deviceId, sessionId);
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    @Override
    public String getCurrentSession(String deviceId) {
        try {
            String currentSessionKey = CURRENT_SESSION_KEY + deviceId;
            return (String) redisTemplate.opsForValue().get(currentSessionKey);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    @Override
    public boolean setCurrentSession(String deviceId, String sessionId) {
        try {
            String currentSessionKey = CURRENT_SESSION_KEY + deviceId;
            redisTemplate.opsForValue().set(currentSessionKey, sessionId, SESSION_EXPIRE_TIME, TimeUnit.SECONDS);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    @Override
    public List<String> getDeviceSessions(String deviceId) {
        try {
            String sessionsKey = DEVICE_SESSIONS_KEY + deviceId;
            Set<Object> sessions = redisTemplate.opsForSet().members(sessionsKey);
            if (sessions != null) {
                List<String> sessionList = new ArrayList<>();
                for (Object session : sessions) {
                    sessionList.add((String) session);
                }
                return sessionList;
            }
            return new ArrayList<>();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }
    
    @Override
    public boolean deleteSession(String deviceId, String sessionId) {
        try {
            String sessionsKey = DEVICE_SESSIONS_KEY + deviceId;
            String sessionInfoKey = SESSION_INFO_KEY + deviceId + ":" + sessionId;
            String currentSessionKey = CURRENT_SESSION_KEY + deviceId;
            
            // 从设备会话列表中移除
            redisTemplate.opsForSet().remove(sessionsKey, sessionId);
            
            // 删除会话信息
            redisTemplate.delete(sessionInfoKey);
            
            // 如果是当前会话，清除当前会话
            String currentSession = getCurrentSession(deviceId);
            if (sessionId.equals(currentSession)) {
                redisTemplate.delete(currentSessionKey);
            }
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    @Override
    public Map<String, Object> getSessionInfo(String deviceId, String sessionId) {
        try {
            String sessionInfoKey = SESSION_INFO_KEY + deviceId + ":" + sessionId;
            Map<Object, Object> sessionInfo = redisTemplate.opsForHash().entries(sessionInfoKey);
            
            Map<String, Object> result = new HashMap<>();
            for (Map.Entry<Object, Object> entry : sessionInfo.entrySet()) {
                result.put((String) entry.getKey(), entry.getValue());
            }
            return result;
        } catch (Exception e) {
            e.printStackTrace();
            return new HashMap<>();
        }
    }
    
    @Override
    public boolean updateSessionInfo(String deviceId, String sessionId, Map<String, Object> sessionInfo) {
        try {
            String sessionInfoKey = SESSION_INFO_KEY + deviceId + ":" + sessionId;
            
            // 更新最后活跃时间
            sessionInfo.put("lastActiveAt", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            
            redisTemplate.opsForHash().putAll(sessionInfoKey, sessionInfo);
            redisTemplate.expire(sessionInfoKey, SESSION_EXPIRE_TIME, TimeUnit.SECONDS);
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}