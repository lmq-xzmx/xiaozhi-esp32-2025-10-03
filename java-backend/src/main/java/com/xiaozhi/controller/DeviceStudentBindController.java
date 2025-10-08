package com.xiaozhi.controller;

import com.xiaozhi.entity.DeviceStudentBind;
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
import java.util.List;
import java.util.Map;

/**
 * 设备学员绑定控制器
 * 实现智能体-设备-学员的绑定关系管理
 */
@RestController
@RequestMapping("/xiaozhi/device-student-bind")
@CrossOrigin(origins = "*")
public class DeviceStudentBindController {

    @Autowired
    private DeviceStudentBindService deviceStudentBindService;

    /**
     * 获取智能体的设备学员绑定列表
     */
    @GetMapping("/list/{agentId}")
    public Result<List<DeviceStudentVO>> getDeviceStudentList(
            @PathVariable String agentId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword) {
        
        List<DeviceStudentVO> list = deviceStudentBindService.getDeviceStudentList(agentId, page, size, keyword);
        return Result.success(list);
    }

    /**
     * 绑定学员到设备
     */
    @PostMapping("/bind")
    public Result<String> bindStudentToDevice(@Valid @RequestBody DeviceStudentBindDTO bindDTO) {
        try {
            deviceStudentBindService.bindStudentToDevice(bindDTO);
            return Result.success("绑定成功");
        } catch (Exception e) {
            return Result.error("绑定失败：" + e.getMessage());
        }
    }

    /**
     * 解绑设备的学员
     */
    @PostMapping("/unbind/{deviceId}")
    public Result<String> unbindStudentFromDevice(
            @PathVariable String deviceId,
            @RequestParam(required = false) String remark) {
        try {
            deviceStudentBindService.unbindStudentFromDevice(deviceId, remark);
            return Result.success("解绑成功");
        } catch (Exception e) {
            return Result.error("解绑失败：" + e.getMessage());
        }
    }

    /**
     * 转移设备绑定（从一个学员转移到另一个学员）
     */
    @PostMapping("/transfer")
    public Result<String> transferDeviceBind(@Valid @RequestBody DeviceStudentBindDTO bindDTO) {
        try {
            deviceStudentBindService.transferDeviceBind(bindDTO);
            return Result.success("转移成功");
        } catch (Exception e) {
            return Result.error("转移失败：" + e.getMessage());
        }
    }

    /**
     * 批量绑定学员到设备
     */
    @PostMapping("/batch-bind")
    public Result<Map<String, Object>> batchBindStudents(@Valid @RequestBody BatchBindDTO batchBindDTO) {
        try {
            Map<String, Object> result = deviceStudentBindService.batchBindStudents(batchBindDTO);
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("批量绑定失败：" + e.getMessage());
        }
    }

    /**
     * 通过Excel文件批量导入绑定关系
     */
    @PostMapping("/import-excel")
    public Result<Map<String, Object>> importBindFromExcel(
            @RequestParam("file") MultipartFile file,
            @RequestParam String agentId) {
        try {
            Map<String, Object> result = deviceStudentBindService.importBindFromExcel(file, agentId);
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("导入失败：" + e.getMessage());
        }
    }

    /**
     * 获取设备的聊天记录统计
     */
    @GetMapping("/chat-statistics/{deviceId}")
    public Result<ChatStatisticsVO> getDeviceChatStatistics(@PathVariable String deviceId) {
        ChatStatisticsVO statistics = deviceStudentBindService.getDeviceChatStatistics(deviceId);
        return Result.success(statistics);
    }

    /**
     * 获取智能体的整体统计信息
     */
    @GetMapping("/agent-statistics/{agentId}")
    public Result<Map<String, Object>> getAgentStatistics(@PathVariable String agentId) {
        Map<String, Object> statistics = deviceStudentBindService.getAgentStatistics(agentId);
        return Result.success(statistics);
    }

    /**
     * 获取设备绑定历史记录
     */
    @GetMapping("/bind-history/{deviceId}")
    public Result<List<DeviceStudentBindHistory>> getBindHistory(
            @PathVariable String deviceId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        
        List<DeviceStudentBindHistory> history = deviceStudentBindService.getBindHistory(deviceId, page, size);
        return Result.success(history);
    }

    /**
     * 搜索可绑定的学员
     */
    @GetMapping("/search-students")
    public Result<List<Map<String, Object>>> searchAvailableStudents(
            @RequestParam String keyword,
            @RequestParam(defaultValue = "10") Integer limit) {
        
        List<Map<String, Object>> students = deviceStudentBindService.searchAvailableStudents(keyword, limit);
        return Result.success(students);
    }

    /**
     * 获取设备详细信息（包含绑定信息）
     */
    @GetMapping("/device-detail/{deviceId}")
    public Result<DeviceStudentVO> getDeviceDetail(@PathVariable String deviceId) {
        DeviceStudentVO deviceDetail = deviceStudentBindService.getDeviceDetail(deviceId);
        return Result.success(deviceDetail);
    }

    /**
     * 验证绑定关系的有效性
     */
    @PostMapping("/validate-bind")
    public Result<Map<String, Object>> validateBind(@Valid @RequestBody DeviceStudentBindDTO bindDTO) {
        Map<String, Object> validation = deviceStudentBindService.validateBind(bindDTO);
        return Result.success(validation);
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
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }
}