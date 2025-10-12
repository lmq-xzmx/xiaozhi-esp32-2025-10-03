package com.xiaozhi.dto;

import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import java.util.List;

/**
 * 批量绑定DTO
 */
public class BatchBindDTO {
    
    @NotEmpty(message = "设备ID列表不能为空")
    private List<String> deviceIds;
    
    @NotEmpty(message = "学员ID列表不能为空")
    private List<Long> studentIds;
    
    private String agentId;
    
    private String remark;
    
    @NotNull(message = "操作员ID不能为空")
    private Long operatorId;
    
    @NotNull(message = "操作员姓名不能为空")
    private String operatorName;
    
    // 构造函数
    public BatchBindDTO() {}
    
    public BatchBindDTO(List<String> deviceIds, List<Long> studentIds, String agentId, 
                       String remark, Long operatorId, String operatorName) {
        this.deviceIds = deviceIds;
        this.studentIds = studentIds;
        this.agentId = agentId;
        this.remark = remark;
        this.operatorId = operatorId;
        this.operatorName = operatorName;
    }
    
    // Getter和Setter方法
    public List<String> getDeviceIds() {
        return deviceIds;
    }
    
    public void setDeviceIds(List<String> deviceIds) {
        this.deviceIds = deviceIds;
    }
    
    public List<Long> getStudentIds() {
        return studentIds;
    }
    
    public void setStudentIds(List<Long> studentIds) {
        this.studentIds = studentIds;
    }
    
    public String getAgentId() {
        return agentId;
    }
    
    public void setAgentId(String agentId) {
        this.agentId = agentId;
    }
    
    public String getRemark() {
        return remark;
    }
    
    public void setRemark(String remark) {
        this.remark = remark;
    }
    
    public Long getOperatorId() {
        return operatorId;
    }
    
    public void setOperatorId(Long operatorId) {
        this.operatorId = operatorId;
    }
    
    public String getOperatorName() {
        return operatorName;
    }
    
    public void setOperatorName(String operatorName) {
        this.operatorName = operatorName;
    }
    
    @Override
    public String toString() {
        return "BatchBindDTO{" +
                "deviceIds=" + deviceIds +
                ", studentIds=" + studentIds +
                ", agentId='" + agentId + '\'' +
                ", remark='" + remark + '\'' +
                ", operatorId=" + operatorId +
                ", operatorName='" + operatorName + '\'' +
                '}';
    }
}