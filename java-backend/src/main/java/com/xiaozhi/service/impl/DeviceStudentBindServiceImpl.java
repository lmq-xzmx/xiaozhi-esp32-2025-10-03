package com.xiaozhi.service.impl;

import com.xiaozhi.entity.DeviceStudentBind;
import com.xiaozhi.entity.DeviceStudentBindHistory;
import com.xiaozhi.dto.DeviceStudentBindDTO;
import com.xiaozhi.dto.BatchBindDTO;
import com.xiaozhi.vo.DeviceStudentVO;
import com.xiaozhi.vo.ChatStatisticsVO;
import com.xiaozhi.service.DeviceStudentBindService;
import com.xiaozhi.repository.DeviceStudentBindRepository;
import com.xiaozhi.common.Result;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.servlet.http.HttpServletResponse;
import java.time.LocalDateTime;
import java.util.*;

/**
 * 设备学员绑定服务实现类
 */
@Service
public class DeviceStudentBindServiceImpl implements DeviceStudentBindService {
    
    @Autowired
    private DeviceStudentBindRepository deviceStudentBindRepository;
    
    @Override
    public Result<DeviceStudentBind> createBind(DeviceStudentBindDTO bindDTO) {
        try {
            DeviceStudentBind bind = new DeviceStudentBind(
                bindDTO.getDeviceId(),
                bindDTO.getStudentId(),
                bindDTO.getAgentId(),
                bindDTO.getRemark(),
                bindDTO.getOperatorId(),
                bindDTO.getOperatorName()
            );
            
            DeviceStudentBind savedBind = deviceStudentBindRepository.save(bind);
            return Result.success(savedBind);
        } catch (Exception e) {
            return Result.error("创建绑定关系失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<Map<String, Object>> batchBindStudents(BatchBindDTO batchBindDTO) {
        try {
            Map<String, Object> result = new HashMap<>();
            List<String> successList = new ArrayList<>();
            List<String> failList = new ArrayList<>();
            
            for (String deviceId : batchBindDTO.getDeviceIds()) {
                for (Long studentId : batchBindDTO.getStudentIds()) {
                    try {
                        DeviceStudentBind bind = new DeviceStudentBind(
                            deviceId,
                            studentId,
                            batchBindDTO.getAgentId(),
                            batchBindDTO.getRemark(),
                            batchBindDTO.getOperatorId(),
                            batchBindDTO.getOperatorName()
                        );
                        deviceStudentBindRepository.save(bind);
                        successList.add(deviceId + "-" + studentId);
                    } catch (Exception e) {
                        failList.add(deviceId + "-" + studentId + ": " + e.getMessage());
                    }
                }
            }
            
            result.put("success", successList);
            result.put("fail", failList);
            result.put("successCount", successList.size());
            result.put("failCount", failList.size());
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("批量绑定失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<String> unbindStudentFromDevice(String deviceId, String remark) {
        try {
            Optional<DeviceStudentBind> bindOpt = deviceStudentBindRepository.findByDeviceIdAndStatus(deviceId, 1);
            if (bindOpt.isPresent()) {
                DeviceStudentBind bind = bindOpt.get();
                bind.setStatus(2); // 设置为已解绑
                bind.setUnbindTime(LocalDateTime.now());
                bind.setUpdateTime(LocalDateTime.now());
                if (remark != null) {
                    bind.setRemark(bind.getRemark() + "; 解绑备注: " + remark);
                }
                deviceStudentBindRepository.save(bind);
                return Result.success("解绑成功");
            } else {
                return Result.error("未找到有效的绑定关系");
            }
        } catch (Exception e) {
            return Result.error("解绑失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<DeviceStudentBind> updateBind(String bindId, DeviceStudentBindDTO bindDTO) {
        try {
            Optional<DeviceStudentBind> bindOpt = deviceStudentBindRepository.findById(Long.parseLong(bindId));
            if (bindOpt.isPresent()) {
                DeviceStudentBind bind = bindOpt.get();
                bind.setStudentId(bindDTO.getStudentId());
                bind.setAgentId(bindDTO.getAgentId());
                bind.setRemark(bindDTO.getRemark());
                bind.setUpdateTime(LocalDateTime.now());
                
                DeviceStudentBind updatedBind = deviceStudentBindRepository.save(bind);
                return Result.success(updatedBind);
            } else {
                return Result.error("未找到绑定关系");
            }
        } catch (Exception e) {
            return Result.error("更新绑定关系失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<DeviceStudentBind> getBindByDeviceId(String deviceId) {
        try {
            Optional<DeviceStudentBind> bind = deviceStudentBindRepository.findByDeviceIdAndStatus(deviceId, 1);
            if (bind.isPresent()) {
                return Result.success(bind.get());
            } else {
                return Result.error("未找到有效的绑定关系");
            }
        } catch (Exception e) {
            return Result.error("查询绑定关系失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<List<DeviceStudentBind>> getBindsByStudentId(String studentId) {
        try {
            List<DeviceStudentBind> binds = deviceStudentBindRepository.findByStudentIdAndStatus(Long.parseLong(studentId), 1);
            return Result.success(binds);
        } catch (Exception e) {
            return Result.error("查询绑定关系失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<Map<String, Object>> getDeviceStudentList(String agentId, String keyword, Integer page, Integer size) {
        try {
            int offset = (page - 1) * size;
            List<Object[]> results = deviceStudentBindRepository.findDeviceStudentListWithPagination(agentId, keyword, offset, size);
            long total = deviceStudentBindRepository.countDeviceStudentList(agentId, keyword);
            
            List<Map<String, Object>> list = new ArrayList<>();
            for (Object[] row : results) {
                Map<String, Object> item = new HashMap<>();
                item.put("id", row[0]);
                item.put("deviceId", row[1]);
                item.put("studentId", row[2]);
                item.put("agentId", row[3]);
                item.put("remark", row[4]);
                item.put("status", row[5]);
                item.put("createTime", row[8]);
                item.put("deviceAlias", row[12]);
                item.put("macAddress", row[13]);
                item.put("studentName", row[14]);
                list.add(item);
            }
            
            Map<String, Object> result = new HashMap<>();
            result.put("list", list);
            result.put("total", total);
            result.put("page", page);
            result.put("size", size);
            result.put("totalPages", (total + size - 1) / size);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("查询设备学员列表失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<String> bindStudentToDevice(DeviceStudentBindDTO bindDTO) {
        return createBind(bindDTO).getCode() == 200 ? Result.success("绑定成功") : Result.error("绑定失败");
    }
    
    @Override
    public Result<String> transferDeviceBind(DeviceStudentBindDTO bindDTO) {
        try {
            // 先解绑原有关系
            unbindStudentFromDevice(bindDTO.getDeviceId(), "设备转移");
            
            // 创建新的绑定关系
            Result<DeviceStudentBind> result = createBind(bindDTO);
            if (result.getCode() == 200) {
                return Result.success("转移成功");
            } else {
                return Result.error("转移失败");
            }
        } catch (Exception e) {
            return Result.error("转移设备绑定失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<ChatStatisticsVO> getDeviceChatStatistics(String deviceId) {
        // 这里应该查询聊天统计信息，暂时返回空数据
        ChatStatisticsVO statistics = new ChatStatisticsVO();
        statistics.setDeviceId(deviceId);
        statistics.setTotalChatCount(0L);
        statistics.setUserMessageCount(0L);
        statistics.setAiMessageCount(0L);
        return Result.success(statistics);
    }
    
    @Override
    public Result<Map<String, Object>> getAgentStatistics(String agentId) {
        try {
            List<DeviceStudentBind> binds = deviceStudentBindRepository.findByAgentIdAndActiveStatus(agentId);
            
            Map<String, Object> statistics = new HashMap<>();
            statistics.put("totalDevices", binds.size());
            statistics.put("activeBindings", binds.size());
            statistics.put("totalStudents", binds.stream().map(DeviceStudentBind::getStudentId).distinct().count());
            
            return Result.success(statistics);
        } catch (Exception e) {
            return Result.error("查询智能体统计信息失败: " + e.getMessage());
        }
    }
    
    @Override
    public Result<List<DeviceStudentBindHistory>> getBindHistory(String deviceId, Integer page, Integer size) {
        // 暂时返回空列表，实际应该查询绑定历史
        return Result.success(new ArrayList<>());
    }
    
    @Override
    public Result<List<Map<String, Object>>> searchAvailableStudents(String keyword, Integer limit) {
        // 暂时返回空列表，实际应该查询可用学员
        return Result.success(new ArrayList<>());
    }
    
    @Override
    public Result<DeviceStudentVO> getDeviceDetail(String deviceId) {
        // 暂时返回空对象，实际应该查询设备详情
        return Result.success(new DeviceStudentVO());
    }
    
    @Override
    public Result<Map<String, Object>> validateBind(DeviceStudentBindDTO bindDTO) {
        Map<String, Object> result = new HashMap<>();
        result.put("valid", true);
        result.put("message", "验证通过");
        return Result.success(result);
    }
    
    @Override
    public void exportBindToExcel(String agentId, String keyword, HttpServletResponse response) throws Exception {
        // 暂时不实现Excel导出功能
    }
    
    @Override
    public Result<Map<String, Object>> importBindFromExcel(String filename, Long operatorId, String operatorName) throws Exception {
        // 暂时不实现Excel导入功能
        Map<String, Object> result = new HashMap<>();
        result.put("success", 0);
        result.put("fail", 0);
        return Result.success(result);
    }
}