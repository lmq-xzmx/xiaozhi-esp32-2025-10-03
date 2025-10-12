package com.xiaozhi.entity;

import javax.persistence.*;
import java.time.LocalDateTime;

/**
 * 设备学员绑定实体类
 */
@Entity
@Table(name = "device_student_bind")
public class DeviceStudentBind {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "device_id")
    private String deviceId;
    
    @Column(name = "student_id")
    private Long studentId;
    
    @Column(name = "agent_id")
    private String agentId;
    
    @Column(name = "remark")
    private String remark;
    
    @Column(name = "status")
    private Integer status; // 0-未绑定，1-已绑定，2-已解绑
    
    @Column(name = "operator_id")
    private Long operatorId;
    
    @Column(name = "operator_name")
    private String operatorName;
    
    @Column(name = "create_time")
    private LocalDateTime createTime;
    
    @Column(name = "update_time")
    private LocalDateTime updateTime;
    
    @Column(name = "bind_time")
    private LocalDateTime bindTime;
    
    @Column(name = "unbind_time")
    private LocalDateTime unbindTime;
    
    // 构造函数
    public DeviceStudentBind() {}
    
    public DeviceStudentBind(String deviceId, Long studentId, String agentId, String remark, 
                           Long operatorId, String operatorName) {
        this.deviceId = deviceId;
        this.studentId = studentId;
        this.agentId = agentId;
        this.remark = remark;
        this.operatorId = operatorId;
        this.operatorName = operatorName;
        this.status = 1; // 默认已绑定
        this.createTime = LocalDateTime.now();
        this.updateTime = LocalDateTime.now();
        this.bindTime = LocalDateTime.now();
    }
    
    // Getter和Setter方法
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
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
    
    public Integer getStatus() {
        return status;
    }
    
    public void setStatus(Integer status) {
        this.status = status;
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
    
    public LocalDateTime getCreateTime() {
        return createTime;
    }
    
    public void setCreateTime(LocalDateTime createTime) {
        this.createTime = createTime;
    }
    
    public LocalDateTime getUpdateTime() {
        return updateTime;
    }
    
    public void setUpdateTime(LocalDateTime updateTime) {
        this.updateTime = updateTime;
    }
    
    public LocalDateTime getBindTime() {
        return bindTime;
    }
    
    public void setBindTime(LocalDateTime bindTime) {
        this.bindTime = bindTime;
    }
    
    public LocalDateTime getUnbindTime() {
        return unbindTime;
    }
    
    public void setUnbindTime(LocalDateTime unbindTime) {
        this.unbindTime = unbindTime;
    }
    
    @Override
    public String toString() {
        return "DeviceStudentBind{" +
                "id=" + id +
                ", deviceId='" + deviceId + '\'' +
                ", studentId=" + studentId +
                ", agentId='" + agentId + '\'' +
                ", remark='" + remark + '\'' +
                ", status=" + status +
                ", operatorId=" + operatorId +
                ", operatorName='" + operatorName + '\'' +
                ", createTime=" + createTime +
                ", updateTime=" + updateTime +
                ", bindTime=" + bindTime +
                ", unbindTime=" + unbindTime +
                '}';
    }
}