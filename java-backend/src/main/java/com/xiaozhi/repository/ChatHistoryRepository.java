package com.xiaozhi.repository;

import com.xiaozhi.entity.ChatHistory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * 聊天记录Repository
 */
@Repository
public interface ChatHistoryRepository extends JpaRepository<ChatHistory, Long> {
    
    /**
     * 获取设备会话列表（分页）
     */
    @Query(value = "SELECT " +
            "session_id as sessionId, " +
            "COUNT(*) as chatCount, " +
            "MIN(created_at) as createdAt, " +
            "MAX(created_at) as lastMessageAt " +
            "FROM ai_agent_chat_history " +
            "WHERE mac_address = :deviceId " +
            "GROUP BY session_id " +
            "ORDER BY MAX(created_at) DESC " +
            "LIMIT :limit OFFSET :offset", 
            nativeQuery = true)
    List<Object[]> findDeviceSessionsWithPagination(@Param("deviceId") String deviceId, 
                                                   @Param("limit") Integer limit, 
                                                   @Param("offset") Integer offset);
    
    /**
     * 获取设备会话总数
     */
    @Query(value = "SELECT COUNT(DISTINCT session_id) " +
            "FROM ai_agent_chat_history " +
            "WHERE mac_address = :deviceId", 
            nativeQuery = true)
    Long countDeviceSessions(@Param("deviceId") String deviceId);
    
    /**
     * 获取会话聊天记录
     */
    @Query(value = "SELECT " +
            "c.id, c.mac_address, c.agent_id, c.session_id, " +
            "c.chat_type, c.content, c.audio_id, c.created_at, " +
            "c.updated_at, c.device_id, c.student_id, " +
            "d.alias as device_alias, " +
            "u.real_name as student_name, " +
            "a.agent_name " +
            "FROM ai_agent_chat_history c " +
            "LEFT JOIN ai_device d ON c.device_id = d.id " +
            "LEFT JOIN sys_user u ON c.student_id = u.id " +
            "LEFT JOIN ai_agent a ON c.agent_id = a.id " +
            "WHERE c.mac_address = :deviceId AND c.session_id = :sessionId " +
            "ORDER BY c.created_at ASC", 
            nativeQuery = true)
    List<Object[]> findSessionChatHistory(@Param("deviceId") String deviceId, 
                                        @Param("sessionId") String sessionId);
    
    /**
     * 获取设备聊天记录（分页）
     */
    @Query(value = "SELECT " +
            "c.id, c.mac_address, c.agent_id, c.session_id, " +
            "c.chat_type, c.content, c.audio_id, c.created_at, " +
            "c.updated_at, c.device_id, c.student_id, " +
            "d.alias as device_alias, " +
            "u.real_name as student_name, " +
            "a.agent_name " +
            "FROM ai_agent_chat_history c " +
            "LEFT JOIN ai_device d ON c.device_id = d.id " +
            "LEFT JOIN sys_user u ON c.student_id = u.id " +
            "LEFT JOIN ai_agent a ON c.agent_id = a.id " +
            "WHERE c.mac_address = :deviceId " +
            "ORDER BY c.created_at DESC " +
            "LIMIT :limit OFFSET :offset", 
            nativeQuery = true)
    List<Object[]> findDeviceChatRecordsWithPagination(@Param("deviceId") String deviceId, 
                                                      @Param("limit") Integer limit, 
                                                      @Param("offset") Integer offset);
    
    /**
     * 获取设备聊天记录总数
     */
    @Query(value = "SELECT COUNT(*) " +
            "FROM ai_agent_chat_history " +
            "WHERE mac_address = :deviceId", 
            nativeQuery = true)
    Long countDeviceChatRecords(@Param("deviceId") String deviceId);
}