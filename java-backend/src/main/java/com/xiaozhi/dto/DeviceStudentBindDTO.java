package com.xiaozhi.dto;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;
import java.time.LocalDateTime;

/**
 * 设备学员绑定数据传输对象
 */
public class DeviceStudentBindDTO {

    @NotBlank(message = "设备ID不能为空")
    private String deviceId;

    @NotNull(message = "学员ID不能为空")
    private Long studentId;

    @NotBlank(message = "智能体ID不能为空")
    private String agentId;

    private String remark;

    private Long operatorId;

    private String operatorName;

    // 构造函数
    public DeviceStudentBindDTO() {}

    public DeviceStudentBindDTO(String deviceId, Long studentId, String agentId) {
        this.deviceId = deviceId;
        this.studentId = studentId;
        this.agentId = agentId;
    }

    // Getter和Setter方法
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
        return "DeviceStudentBindDTO{" +
                "deviceId='" + deviceId + '\'' +
                ", studentId=" + studentId +
                ", agentId='" + agentId + '\'' +
                ", remark='" + remark + '\'' +
                ", operatorId=" + operatorId +
                ", operatorName='" + operatorName + '\'' +
                '}';
    }
}