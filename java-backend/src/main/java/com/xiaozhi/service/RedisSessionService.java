package com.xiaozhi.service;

import java.util.List;
import java.util.Map;

/**
 * Redis会话管理服务接口
 */
public interface RedisSessionService {
    
    /**
     * 创建新会话
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 是否成功
     */
    boolean createSession(String deviceId, String sessionId);
    
    /**
     * 获取设备当前会话
     * @param deviceId 设备ID
     * @return 当前会话ID
     */
    String getCurrentSession(String deviceId);
    
    /**
     * 设置设备当前会话
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 是否成功
     */
    boolean setCurrentSession(String deviceId, String sessionId);
    
    /**
     * 获取设备所有会话
     * @param deviceId 设备ID
     * @return 会话ID列表
     */
    List<String> getDeviceSessions(String deviceId);
    
    /**
     * 删除会话
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 是否成功
     */
    boolean deleteSession(String deviceId, String sessionId);
    
    /**
     * 获取会话信息
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 会话信息
     */
    Map<String, Object> getSessionInfo(String deviceId, String sessionId);
    
    /**
     * 更新会话信息
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @param sessionInfo 会话信息
     * @return 是否成功
     */
    boolean updateSessionInfo(String deviceId, String sessionId, Map<String, Object> sessionInfo);
}