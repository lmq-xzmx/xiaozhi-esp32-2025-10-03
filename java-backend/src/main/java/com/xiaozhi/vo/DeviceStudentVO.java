package com.xiaozhi.vo;

import java.time.LocalDateTime;

/**
 * 设备学员视图对象
 * 用于前端展示设备和学员的绑定信息
 */
public class DeviceStudentVO {

    // 设备信息
    private String deviceId;
    private String macAddress;
    private String deviceAlias;
    private LocalDateTime lastConnectedAt;
    private String board;
    private String appVersion;
    private Integer bindStatus; // 0-未绑定，1-已绑定，2-已解绑

    // 智能体信息
    private String agentId;
    private String agentName;
    private String agentCode;

    // 学员信息
    private Long studentId;
    private String studentUsername;
    private String studentRealName;
    private String schoolName;
    private String currentGrade;
    private String className;
    private String contactPhone;
    private String contactEmail;
    private String studentId_; // 学号

    // 绑定信息
    private LocalDateTime bindTime;
    private String bindRemark;

    // 聊天统计信息
    private Long totalChatCount;
    private Long userMessageCount;
    private Long aiMessageCount;
    private LocalDateTime lastChatTime;
    private LocalDateTime firstChatTime;

    // 构造函数
    public DeviceStudentVO() {}

    // Getter和Setter方法
    public String getDeviceId() {
        return deviceId;
    }

    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }

    public String getMacAddress() {
        return macAddress;
    }

    public void setMacAddress(String macAddress) {
        this.macAddress = macAddress;
    }

    public String getDeviceAlias() {
        return deviceAlias;
    }

    public void setDeviceAlias(String deviceAlias) {
        this.deviceAlias = deviceAlias;
    }

    public LocalDateTime getLastConnectedAt() {
        return lastConnectedAt;
    }

    public void setLastConnectedAt(LocalDateTime lastConnectedAt) {
        this.lastConnectedAt = lastConnectedAt;
    }

    public String getBoard() {
        return board;
    }

    public void setBoard(String board) {
        this.board = board;
    }

    public String getAppVersion() {
        return appVersion;
    }

    public void setAppVersion(String appVersion) {
        this.appVersion = appVersion;
    }

    public Integer getBindStatus() {
        return bindStatus;
    }

    public void setBindStatus(Integer bindStatus) {
        this.bindStatus = bindStatus;
    }

    public String getAgentId() {
        return agentId;
    }

    public void setAgentId(String agentId) {
        this.agentId = agentId;
    }

    public String getAgentName() {
        return agentName;
    }

    public void setAgentName(String agentName) {
        this.agentName = agentName;
    }

    public String getAgentCode() {
        return agentCode;
    }

    public void setAgentCode(String agentCode) {
        this.agentCode = agentCode;
    }

    public Long getStudentId() {
        return studentId;
    }

    public void setStudentId(Long studentId) {
        this.studentId = studentId;
    }

    public String getStudentUsername() {
        return studentUsername;
    }

    public void setStudentUsername(String studentUsername) {
        this.studentUsername = studentUsername;
    }

    public String getStudentRealName() {
        return studentRealName;
    }

    public void setStudentRealName(String studentRealName) {
        this.studentRealName = studentRealName;
    }

    public String getSchoolName() {
        return schoolName;
    }

    public void setSchoolName(String schoolName) {
        this.schoolName = schoolName;
    }

    public String getCurrentGrade() {
        return currentGrade;
    }

    public void setCurrentGrade(String currentGrade) {
        this.currentGrade = currentGrade;
    }

    public String getClassName() {
        return className;
    }

    public void setClassName(String className) {
        this.className = className;
    }

    public String getContactPhone() {
        return contactPhone;
    }

    public void setContactPhone(String contactPhone) {
        this.contactPhone = contactPhone;
    }

    public String getContactEmail() {
        return contactEmail;
    }

    public void setContactEmail(String contactEmail) {
        this.contactEmail = contactEmail;
    }

    public String getStudentId_() {
        return studentId_;
    }

    public void setStudentId_(String studentId_) {
        this.studentId_ = studentId_;
    }

    public LocalDateTime getBindTime() {
        return bindTime;
    }

    public void setBindTime(LocalDateTime bindTime) {
        this.bindTime = bindTime;
    }

    public String getBindRemark() {
        return bindRemark;
    }

    public void setBindRemark(String bindRemark) {
        this.bindRemark = bindRemark;
    }

    public Long getTotalChatCount() {
        return totalChatCount;
    }

    public void setTotalChatCount(Long totalChatCount) {
        this.totalChatCount = totalChatCount;
    }

    public Long getUserMessageCount() {
        return userMessageCount;
    }

    public void setUserMessageCount(Long userMessageCount) {
        this.userMessageCount = userMessageCount;
    }

    public Long getAiMessageCount() {
        return aiMessageCount;
    }

    public void setAiMessageCount(Long aiMessageCount) {
        this.aiMessageCount = aiMessageCount;
    }

    public LocalDateTime getLastChatTime() {
        return lastChatTime;
    }

    public void setLastChatTime(LocalDateTime lastChatTime) {
        this.lastChatTime = lastChatTime;
    }

    public LocalDateTime getFirstChatTime() {
        return firstChatTime;
    }

    public void setFirstChatTime(LocalDateTime firstChatTime) {
        this.firstChatTime = firstChatTime;
    }

    // 便利方法
    public String getBindStatusText() {
        if (bindStatus == null) return "未知";
        switch (bindStatus) {
            case 0: return "未绑定";
            case 1: return "已绑定";
            case 2: return "已解绑";
            default: return "未知";
        }
    }

    public boolean isBound() {
        return bindStatus != null && bindStatus == 1;
    }

    public boolean hasStudent() {
        return studentId != null && studentId > 0;
    }

    @Override
    public String toString() {
        return "DeviceStudentVO{" +
                "deviceId='" + deviceId + '\'' +
                ", macAddress='" + macAddress + '\'' +
                ", deviceAlias='" + deviceAlias + '\'' +
                ", agentId='" + agentId + '\'' +
                ", agentName='" + agentName + '\'' +
                ", studentId=" + studentId +
                ", studentRealName='" + studentRealName + '\'' +
                ", bindStatus=" + bindStatus +
                ", bindTime=" + bindTime +
                '}';
    }
}