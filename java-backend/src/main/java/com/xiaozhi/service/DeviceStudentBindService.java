package com.xiaozhi.service;

import com.xiaozhi.entity.DeviceStudentBind;
import com.xiaozhi.entity.DeviceStudentBindHistory;
import com.xiaozhi.dto.DeviceStudentBindDTO;
import com.xiaozhi.dto.BatchBindDTO;
import com.xiaozhi.vo.DeviceStudentVO;
import com.xiaozhi.vo.ChatStatisticsVO;
import com.xiaozhi.common.Result;
import javax.servlet.http.HttpServletResponse;
import java.util.List;
import java.util.Map;

/**
 * 设备学员绑定服务接口
 */
public interface DeviceStudentBindService {
    
    /**
     * 创建设备学员绑定关系
     */
    Result<DeviceStudentBind> createBind(DeviceStudentBindDTO bindDTO);
    
    /**
     * 批量绑定学员到设备
     */
    Result<Map<String, Object>> batchBindStudents(BatchBindDTO batchBindDTO);
    
    /**
     * 解绑设备和学员
     */
    Result<String> unbindStudentFromDevice(String deviceId, String remark);
    
    /**
     * 更新绑定信息
     */
    Result<DeviceStudentBind> updateBind(String bindId, DeviceStudentBindDTO bindDTO);
    
    /**
     * 根据设备ID查询绑定信息
     */
    Result<DeviceStudentBind> getBindByDeviceId(String deviceId);
    
    /**
     * 根据学员ID查询绑定信息
     */
    Result<List<DeviceStudentBind>> getBindsByStudentId(String studentId);
    
    /**
     * 分页查询绑定列表
     */
    Result<Map<String, Object>> getDeviceStudentList(String agentId, String keyword, Integer page, Integer size);
    
    /**
     * 绑定学员到设备
     */
    Result<String> bindStudentToDevice(DeviceStudentBindDTO bindDTO);
    
    /**
     * 转移设备绑定
     */
    Result<String> transferDeviceBind(DeviceStudentBindDTO bindDTO);
    
    /**
     * 获取设备聊天统计信息
     */
    Result<ChatStatisticsVO> getDeviceChatStatistics(String deviceId);
    
    /**
     * 获取智能体统计信息
     */
    Result<Map<String, Object>> getAgentStatistics(String agentId);
    
    /**
     * 获取绑定历史记录
     */
    Result<List<DeviceStudentBindHistory>> getBindHistory(String deviceId, Integer page, Integer size);
    
    /**
     * 搜索可绑定的学员
     */
    Result<List<Map<String, Object>>> searchAvailableStudents(String keyword, Integer limit);
    
    /**
     * 获取设备详细信息
     */
    Result<DeviceStudentVO> getDeviceDetail(String deviceId);
    
    /**
     * 验证绑定关系
     */
    Result<Map<String, Object>> validateBind(DeviceStudentBindDTO bindDTO);
    
    /**
     * 从Excel导入绑定关系
     */
    Result<Map<String, Object>> importBindFromExcel(String filename, Long operatorId, String operatorName) throws Exception;
    
    /**
     * 导出绑定关系到Excel
     */
    void exportBindToExcel(String agentId, String keyword, HttpServletResponse response) throws Exception;
}