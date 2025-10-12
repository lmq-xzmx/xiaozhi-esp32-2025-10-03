package com.xiaozhi.controller;

import com.xiaozhi.common.Result;
import com.xiaozhi.dto.ChatHistoryDownloadDTO;
import com.xiaozhi.dto.ChatHistoryReportDTO;
import com.xiaozhi.entity.ChatHistory;
import com.xiaozhi.service.ChatHistoryService;
import com.xiaozhi.vo.PageResult;
import com.xiaozhi.vo.SessionVO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 聊天记录控制器
 */
@RestController
@RequestMapping("/chat-history")
@CrossOrigin(origins = "*")
public class ChatHistoryController {
    
    @Autowired
    private ChatHistoryService chatHistoryService;
    
    /**
     * 聊天记录上报
     * @param reportDTO 上报数据
     * @return 上报结果
     */
    @PostMapping("/report")
    public Result<String> reportChatHistory(@RequestBody ChatHistoryReportDTO reportDTO) {
        try {
            boolean success = chatHistoryService.reportChatHistory(reportDTO);
            if (success) {
                return Result.success("聊天记录上报成功");
            } else {
                return Result.error("聊天记录上报失败");
            }
        } catch (Exception e) {
            return Result.error("聊天记录上报失败: " + e.getMessage());
        }
    }

    /**
     * 获取设备会话列表
     * @param deviceId 设备ID
     * @param page 页码，默认1
     * @param perPage 每页数量，默认20
     * @return 会话列表
     */
    @GetMapping("/device/{deviceId}/sessions")
    public Result<PageResult<SessionVO>> getDeviceSessions(
            @PathVariable String deviceId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(name = "per_page", defaultValue = "20") Integer perPage) {
        try {
            PageResult<SessionVO> result = chatHistoryService.getDeviceSessions(deviceId, page, perPage);
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取设备会话列表失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取会话聊天记录
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 聊天记录列表
     */
    @GetMapping("/device/{deviceId}/session/{sessionId}/chat-history")
    public Result<List<ChatHistory>> getSessionChatHistory(
            @PathVariable String deviceId,
            @PathVariable String sessionId) {
        try {
            List<ChatHistory> result = chatHistoryService.getSessionChatHistory(deviceId, sessionId);
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取会话聊天记录失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取设备聊天记录（分页）
     * @param deviceId 设备ID
     * @param page 页码，默认1
     * @param perPage 每页数量，默认20
     * @return 聊天记录分页结果
     */
    @GetMapping("/chat-records/{deviceId}")
    public Result<PageResult<ChatHistory>> getDeviceChatRecords(
            @PathVariable String deviceId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "20") Integer perPage) {
        PageResult<ChatHistory> result = chatHistoryService.getDeviceChatRecords(deviceId, page, perPage);
        return Result.success(result);
    }

    /**
     * 生成聊天记录下载链接
     * @param deviceId 设备ID
     * @param sessionId 会话ID（可选）
     * @param format 导出格式（csv, json, txt），默认csv
     * @return 下载信息
     */
    @GetMapping("/download-url/{deviceId}")
    public Result<ChatHistoryDownloadDTO> generateDownloadUrl(
            @PathVariable String deviceId,
            @RequestParam(required = false) String sessionId,
            @RequestParam(defaultValue = "csv") String format) {
        try {
            ChatHistoryDownloadDTO downloadInfo = chatHistoryService.generateDownloadUrl(deviceId, sessionId, format);
            return Result.success(downloadInfo);
        } catch (Exception e) {
            return Result.error("生成下载链接失败: " + e.getMessage());
        }
    }

    /**
     * 下载聊天记录文件
     * @param deviceId 设备ID
     * @param sessionId 会话ID（可选）
     * @param format 导出格式（csv, json, txt），默认csv
     * @return 文件下载响应
     */
    @GetMapping("/download/{deviceId}")
    public ResponseEntity<Resource> downloadChatHistory(
            @PathVariable String deviceId,
            @RequestParam(required = false) String sessionId,
            @RequestParam(defaultValue = "csv") String format) {
        try {
            Resource resource = chatHistoryService.downloadChatHistory(deviceId, sessionId, format);
            String filename = String.format("chat_history_%s_%s.%s", 
                deviceId, 
                sessionId != null ? sessionId : "all", 
                format);
            
            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + filename + "\"")
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }
}