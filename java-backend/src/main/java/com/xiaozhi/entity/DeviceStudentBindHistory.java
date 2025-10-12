package com.xiaozhi.entity;

import java.time.LocalDateTime;

/**
 * 设备学员绑定历史记录实体类
 */
public class DeviceStudentBindHistory {
    
    private Long id;
    private Long bindId;
    private String deviceId;
    private Long studentId;
    private String agentId;
    private String remark;
    private Integer operation; // 1-绑定，2-解绑，3-修改
    private String operationDesc;
    private Long operatorId;
    private String operatorName;
    private LocalDateTime operationTime;
    private String beforeData; // 操作前数据（JSON格式）
    private String afterData;  // 操作后数据（JSON格式）
    
    // 构造函数
    public DeviceStudentBindHistory() {}
    
    public DeviceStudentBindHistory(Long bindId, String deviceId, Long studentId, String agentId,
                                  Integer operation, String operationDesc, Long operatorId, 
                                  String operatorName) {
        this.bindId = bindId;
        this.deviceId = deviceId;
        this.studentId = studentId;
        this.agentId = agentId;
        this.operation = operation;
        this.operationDesc = operationDesc;
        this.operatorId = operatorId;
        this.operatorName = operatorName;
        this.operationTime = LocalDateTime.now();
    }
    
    // Getter和Setter方法
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public Long getBindId() {
        return bindId;
    }
    
    public void setBindId(Long bindId) {
        this.bindId = bindId;
    }
    
    public String getDeviceId() {
        return deviceId;
    }
    
    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }
    
    public Long getStudentId() {
        return studentId;
    }
    
    public void setStudentId(Long studentId) {
        this.studentId = studentId;
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
    
    public Integer getOperation() {
        return operation;
    }
    
    public void setOperation(Integer operation) {
        this.operation = operation;
    }
    
    public String getOperationDesc() {
        return operationDesc;
    }
    
    public void setOperationDesc(String operationDesc) {
        this.operationDesc = operationDesc;
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
    
    public LocalDateTime getOperationTime() {
        return operationTime;
    }
    
    public void setOperationTime(LocalDateTime operationTime) {
        this.operationTime = operationTime;
    }
    
    public String getBeforeData() {
        return beforeData;
    }
    
    public void setBeforeData(String beforeData) {
        this.beforeData = beforeData;
    }
    
    public String getAfterData() {
        return afterData;
    }
    
    public void setAfterData(String afterData) {
        this.afterData = afterData;
    }
    
    @Override
    public String toString() {
        return "DeviceStudentBindHistory{" +
                "id=" + id +
                ", bindId=" + bindId +
                ", deviceId='" + deviceId + '\'' +
                ", studentId=" + studentId +
                ", agentId='" + agentId + '\'' +
                ", remark='" + remark + '\'' +
                ", operation=" + operation +
                ", operationDesc='" + operationDesc + '\'' +
                ", operatorId=" + operatorId +
                ", operatorName='" + operatorName + '\'' +
                ", operationTime=" + operationTime +
                ", beforeData='" + beforeData + '\'' +
                ", afterData='" + afterData + '\'' +
                '}';
    }
}