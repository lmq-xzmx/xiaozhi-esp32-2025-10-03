package com.xiaozhi.controller;

import com.xiaozhi.entity.DeviceStudentBindHistory;
import com.xiaozhi.service.DeviceStudentBindService;
import com.xiaozhi.common.Result;
import com.xiaozhi.dto.DeviceStudentBindDTO;
import com.xiaozhi.dto.BatchBindDTO;
import com.xiaozhi.vo.DeviceStudentVO;
import com.xiaozhi.vo.ChatStatisticsVO;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.validation.Valid;
import javax.servlet.http.HttpServletResponse;
import java.util.List;
import java.util.Map;

/**
 * 设备学员绑定控制器
 * 实现智能体-设备-学员的绑定关系管理
 */
@RestController
@RequestMapping("/device-student-bind")
@CrossOrigin(origins = "*")
public class DeviceStudentBindController {

    @Autowired
    private DeviceStudentBindService deviceStudentBindService;

    /**
     * 获取智能体的设备学员绑定列表
     */
    @GetMapping("/list/{agentId}")
    public Result<Map<String, Object>> getDeviceStudentList(
            @PathVariable String agentId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword) {
        
        return deviceStudentBindService.getDeviceStudentList(agentId, keyword, page, size);
    }

    /**
     * 绑定学员到设备
     */
    @PostMapping("/bind")
    public Result<String> bindStudentToDevice(@Valid @RequestBody DeviceStudentBindDTO bindDTO) {
        return deviceStudentBindService.bindStudentToDevice(bindDTO);
    }

    /**
     * 解绑设备的学员
     */
    @PostMapping("/unbind/{deviceId}")
    public Result<String> unbindStudentFromDevice(
            @PathVariable String deviceId,
            @RequestParam(required = false) String remark) {
        return deviceStudentBindService.unbindStudentFromDevice(deviceId, remark);
    }

    /**
     * 转移设备绑定（从一个学员转移到另一个学员）
     */
    @PostMapping("/transfer")
    public Result<String> transferDeviceBind(@Valid @RequestBody DeviceStudentBindDTO bindDTO) {
        return deviceStudentBindService.transferDeviceBind(bindDTO);
    }

    /**
     * 批量绑定学员到设备
     */
    @PostMapping("/batch-bind")
    public Result<Map<String, Object>> batchBindStudents(@Valid @RequestBody BatchBindDTO batchBindDTO) {
        return deviceStudentBindService.batchBindStudents(batchBindDTO);
    }

    /**
     * 从Excel导入绑定关系
     */
    @PostMapping("/import-excel")
    public Result<Map<String, Object>> importBindFromExcel(
            @RequestParam("file") MultipartFile file,
            @RequestParam String agentId) {
        try {
            // 这里需要处理文件上传和转换逻辑
            return deviceStudentBindService.importBindFromExcel(file.getOriginalFilename(), 1L, "admin");
        } catch (Exception e) {
            return Result.error("导入失败：" + e.getMessage());
        }
    }

    /**
     * 获取设备聊天统计信息
     */
    @GetMapping("/chat-statistics/{deviceId}")
    public Result<ChatStatisticsVO> getDeviceChatStatistics(@PathVariable String deviceId) {
        return deviceStudentBindService.getDeviceChatStatistics(deviceId);
    }

    /**
     * 获取智能体统计信息
     */
    @GetMapping("/agent-statistics/{agentId}")
    public Result<Map<String, Object>> getAgentStatistics(@PathVariable String agentId) {
        return deviceStudentBindService.getAgentStatistics(agentId);
    }

    /**
     * 获取绑定历史记录
     */
    @GetMapping("/bind-history/{deviceId}")
    public Result<List<DeviceStudentBindHistory>> getBindHistory(
            @PathVariable String deviceId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        
        return deviceStudentBindService.getBindHistory(deviceId, page, size);
    }

    /**
     * 搜索可绑定的学员
     */
    @GetMapping("/search-students")
    public Result<List<Map<String, Object>>> searchAvailableStudents(
            @RequestParam String keyword,
            @RequestParam(defaultValue = "10") Integer limit) {
        
        return deviceStudentBindService.searchAvailableStudents(keyword, limit);
    }

    /**
     * 获取设备详细信息（包含绑定信息）
     */
    @GetMapping("/device-detail/{deviceId}")
    public Result<DeviceStudentVO> getDeviceDetail(@PathVariable String deviceId) {
        return deviceStudentBindService.getDeviceDetail(deviceId);
    }

    /**
     * 验证绑定关系的有效性
     */
    @PostMapping("/validate-bind")
    public Result<Map<String, Object>> validateBind(@Valid @RequestBody DeviceStudentBindDTO bindDTO) {
        return deviceStudentBindService.validateBind(bindDTO);
    }

    /**
     * 导出绑定关系到Excel
     */
    @GetMapping("/export-excel/{agentId}")
    public void exportBindToExcel(
            @PathVariable String agentId,
            @RequestParam(required = false) String keyword,
            HttpServletResponse response) {
        try {
            deviceStudentBindService.exportBindToExcel(agentId, keyword, response);
        } catch (Exception e) {
            // 处理导出异常
            response.setStatus(500);
        }
    }
}