package com.xiaozhi.service.impl;

import com.xiaozhi.dto.ChatHistoryDownloadDTO;
import com.xiaozhi.dto.ChatHistoryReportDTO;
import com.xiaozhi.entity.ChatHistory;
import com.xiaozhi.repository.ChatHistoryRepository;
import com.xiaozhi.service.ChatHistoryService;
import com.xiaozhi.vo.PageResult;
import com.xiaozhi.vo.SessionVO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
/**
 * 聊天记录服务实现类
 */
@Service
public class ChatHistoryServiceImpl implements ChatHistoryService {
    
    @Autowired
    private ChatHistoryRepository chatHistoryRepository;
    
    @Override
    public PageResult<SessionVO> getDeviceSessions(String deviceId, Integer page, Integer perPage) {
        // 计算偏移量
        int offset = (page - 1) * perPage;
        
        // 获取会话列表
        List<Object[]> sessionData = chatHistoryRepository.findDeviceSessionsWithPagination(
                deviceId, perPage, offset);
        
        // 转换为SessionVO
        List<SessionVO> sessions = new ArrayList<>();
        for (Object[] row : sessionData) {
            SessionVO session = new SessionVO();
            session.setSessionId((String) row[0]);
            session.setChatCount(((Number) row[1]).intValue());
            session.setCreatedAt(((Timestamp) row[2]).toLocalDateTime());
            session.setLastMessageAt(((Timestamp) row[3]).toLocalDateTime());
            sessions.add(session);
        }
        
        // 获取总数
        Long total = chatHistoryRepository.countDeviceSessions(deviceId);
        
        // 创建分页信息
        PageResult.Pagination pagination = new PageResult.Pagination(page, perPage, total);
        
        return new PageResult<>(sessions, pagination);
    }
    
    @Override
    public List<ChatHistory> getSessionChatHistory(String deviceId, String sessionId) {
        List<Object[]> chatData = chatHistoryRepository.findSessionChatHistory(deviceId, sessionId);
        
        List<ChatHistory> chatHistories = new ArrayList<>();
        for (Object[] row : chatData) {
            ChatHistory chat = new ChatHistory();
            chat.setId(((Number) row[0]).longValue());
            chat.setMacAddress((String) row[1]);
            chat.setAgentId((String) row[2]);
            chat.setSessionId((String) row[3]);
            chat.setChatType(row[4] != null ? String.valueOf(row[4]) : null);
            chat.setContent((String) row[5]);
            chat.setAudioId((String) row[6]);
            chat.setCreatedAt(((Timestamp) row[7]).toLocalDateTime());
            chat.setUpdatedAt(row[8] != null ? ((Timestamp) row[8]).toLocalDateTime() : null);
            chat.setDeviceId((String) row[9]);
            chat.setStudentId(row[10] != null ? ((Number) row[10]).longValue() : null);
            chat.setDeviceAlias((String) row[11]);
            chat.setStudentName((String) row[12]);
            chat.setAgentName((String) row[13]);
            chatHistories.add(chat);
        }
        
        return chatHistories;
    }
    
    @Override
    public PageResult<ChatHistory> getDeviceChatRecords(String deviceId, Integer page, Integer perPage) {
        // 计算偏移量
        int offset = (page - 1) * perPage;
        
        // 获取聊天记录
        List<Object[]> chatData = chatHistoryRepository.findDeviceChatRecordsWithPagination(
                deviceId, perPage, offset);
        
        List<ChatHistory> chatHistories = new ArrayList<>();
        for (Object[] row : chatData) {
            ChatHistory chat = new ChatHistory();
            chat.setId(((Number) row[0]).longValue());
            chat.setMacAddress((String) row[1]);
            chat.setAgentId((String) row[2]);
            chat.setSessionId((String) row[3]);
            chat.setChatType(row[4] != null ? String.valueOf(row[4]) : null);
            chat.setContent((String) row[5]);
            chat.setAudioId((String) row[6]);
            chat.setCreatedAt(((Timestamp) row[7]).toLocalDateTime());
            chat.setUpdatedAt(row[8] != null ? ((Timestamp) row[8]).toLocalDateTime() : null);
            chat.setDeviceId((String) row[9]);
            chat.setStudentId(row[10] != null ? ((Number) row[10]).longValue() : null);
            chat.setDeviceAlias((String) row[11]);
            chat.setStudentName((String) row[12]);
            chat.setAgentName((String) row[13]);
            chatHistories.add(chat);
        }
        
        // 获取总数
        Long total = chatHistoryRepository.countDeviceChatRecords(deviceId);
        
        // 创建分页信息
        PageResult.Pagination pagination = new PageResult.Pagination(page, perPage, total);
        
        return new PageResult<>(chatHistories, pagination);
    }
    
    @Override
    public boolean reportChatHistory(ChatHistoryReportDTO reportDTO) {
        try {
            // 创建聊天记录实体
            ChatHistory chatHistory = new ChatHistory();
            chatHistory.setMacAddress(reportDTO.getMacAddress());
            chatHistory.setAgentId(reportDTO.getAgentId());
            chatHistory.setSessionId(reportDTO.getSessionId());
            chatHistory.setChatType(reportDTO.getChatType() != null ? reportDTO.getChatType().toString() : null);
            chatHistory.setContent(reportDTO.getContent());
            chatHistory.setAudioId(reportDTO.getAudioBase64()); // 暂时存储在audioId字段
            chatHistory.setDeviceId(reportDTO.getDeviceId());
            chatHistory.setStudentId(reportDTO.getStudentId());
            
            // 设置时间
            LocalDateTime now = reportDTO.getReportTime() != null ? reportDTO.getReportTime() : LocalDateTime.now();
            chatHistory.setCreatedAt(now);
            chatHistory.setUpdatedAt(now);
            
            // 保存到数据库
            chatHistoryRepository.save(chatHistory);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    @Override
    public ChatHistoryDownloadDTO generateDownloadUrl(String deviceId, String sessionId, String format) {
        String baseUrl = "http://localhost:8080/chat-history/download/" + deviceId;
        String downloadUrl = baseUrl + "?format=" + format;
        if (sessionId != null) {
            downloadUrl += "&sessionId=" + sessionId;
        }
        
        String fileName = String.format("chat_history_%s_%s.%s", 
            deviceId, 
            sessionId != null ? sessionId : "all", 
            format);
        
        return new ChatHistoryDownloadDTO(downloadUrl, fileName, 0L, format);
    }
    
    @Override
    public Resource downloadChatHistory(String deviceId, String sessionId, String format) {
        try {
            List<ChatHistory> chatHistories;
            if (sessionId != null) {
                chatHistories = getSessionChatHistory(deviceId, sessionId);
            } else {
                // 获取所有聊天记录
                PageResult<ChatHistory> result = getDeviceChatRecords(deviceId, 1, Integer.MAX_VALUE);
                chatHistories = result.getList();
            }
            
            byte[] content = generateFileContent(chatHistories, format);
            return new ByteArrayResource(content);
        } catch (Exception e) {
            throw new RuntimeException("生成下载文件失败", e);
        }
    }
    
    private byte[] generateFileContent(List<ChatHistory> chatHistories, String format) throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        OutputStreamWriter writer = new OutputStreamWriter(outputStream, StandardCharsets.UTF_8);
        
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        
        switch (format.toLowerCase()) {
            case "csv":
                writer.write("设备ID,会话ID,聊天类型,内容,创建时间\n");
                for (ChatHistory chat : chatHistories) {
                    writer.write(String.format("%s,%s,%s,\"%s\",%s\n",
                        chat.getDeviceId() != null ? chat.getDeviceId() : "",
                        chat.getSessionId() != null ? chat.getSessionId() : "",
                        chat.getChatType() != null ? chat.getChatType() : "",
                        chat.getContent() != null ? chat.getContent().replace("\"", "\"\"") : "",
                        chat.getCreatedAt() != null ? chat.getCreatedAt().format(formatter) : ""
                    ));
                }
                break;
            case "json":
                writer.write("[\n");
                for (int i = 0; i < chatHistories.size(); i++) {
                    ChatHistory chat = chatHistories.get(i);
                    writer.write(String.format("  {\n" +
                        "    \"deviceId\": \"%s\",\n" +
                        "    \"sessionId\": \"%s\",\n" +
                        "    \"chatType\": \"%s\",\n" +
                        "    \"content\": \"%s\",\n" +
                        "    \"createdAt\": \"%s\"\n" +
                        "  }%s\n",
                        chat.getDeviceId() != null ? chat.getDeviceId() : "",
                        chat.getSessionId() != null ? chat.getSessionId() : "",
                        chat.getChatType() != null ? chat.getChatType() : "",
                        chat.getContent() != null ? chat.getContent().replace("\"", "\\\"") : "",
                        chat.getCreatedAt() != null ? chat.getCreatedAt().format(formatter) : "",
                        i < chatHistories.size() - 1 ? "," : ""
                    ));
                }
                writer.write("]\n");
                break;
            case "txt":
                for (ChatHistory chat : chatHistories) {
                    writer.write(String.format("[%s] 设备:%s 会话:%s 类型:%s\n内容:%s\n\n",
                        chat.getCreatedAt() != null ? chat.getCreatedAt().format(formatter) : "",
                        chat.getDeviceId() != null ? chat.getDeviceId() : "",
                        chat.getSessionId() != null ? chat.getSessionId() : "",
                        chat.getChatType() != null ? chat.getChatType() : "",
                        chat.getContent() != null ? chat.getContent() : ""
                    ));
                }
                break;
            default:
                throw new IllegalArgumentException("不支持的格式: " + format);
        }
        
        writer.flush();
        writer.close();
        return outputStream.toByteArray();
    }
}