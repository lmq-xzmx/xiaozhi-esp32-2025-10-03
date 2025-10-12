package com.xiaozhi.service;

import com.xiaozhi.dto.ChatHistoryDownloadDTO;
import com.xiaozhi.dto.ChatHistoryReportDTO;
import com.xiaozhi.entity.ChatHistory;
import com.xiaozhi.vo.PageResult;
import com.xiaozhi.vo.SessionVO;
import org.springframework.core.io.Resource;

import java.util.List;

/**
 * 聊天记录服务接口
 */
public interface ChatHistoryService {
    
    /**
     * 获取设备会话列表
     * @param deviceId 设备ID
     * @param page 页码
     * @param perPage 每页数量
     * @return 会话列表
     */
    PageResult<SessionVO> getDeviceSessions(String deviceId, Integer page, Integer perPage);
    
    /**
     * 获取会话聊天记录
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 聊天记录列表
     */
    List<ChatHistory> getSessionChatHistory(String deviceId, String sessionId);
    
    /**
     * 获取设备聊天记录（分页）
     * @param deviceId 设备ID
     * @param page 页码
     * @param perPage 每页数量
     * @return 聊天记录分页结果
     */
    PageResult<ChatHistory> getDeviceChatRecords(String deviceId, Integer page, Integer perPage);
    
    /**
     * 聊天记录上报
     * @param reportDTO 上报数据
     * @return 上报是否成功
     */
    boolean reportChatHistory(ChatHistoryReportDTO reportDTO);
    
    /**
     * 生成聊天记录下载链接
     * @param deviceId 设备ID
     * @param sessionId 会话ID（可选）
     * @param format 导出格式
     * @return 下载信息
     */
    ChatHistoryDownloadDTO generateDownloadUrl(String deviceId, String sessionId, String format);
    
    /**
     * 下载聊天记录文件
     * @param deviceId 设备ID
     * @param sessionId 会话ID（可选）
     * @param format 导出格式
     * @return 文件资源
     */
    Resource downloadChatHistory(String deviceId, String sessionId, String format);
}