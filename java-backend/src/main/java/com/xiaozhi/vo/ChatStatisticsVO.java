package com.xiaozhi.vo;

import java.time.LocalDateTime;

/**
 * 聊天统计视图对象
 */
public class ChatStatisticsVO {
    
    private String deviceId;
    private Long studentId;
    private String studentName;
    private String agentId;
    private String agentName;
    
    // 聊天统计数据
    private Long totalChatCount;        // 总聊天次数
    private Long userMessageCount;      // 用户消息数量
    private Long aiMessageCount;        // AI消息数量
    private Long todayChatCount;        // 今日聊天次数
    private Long weekChatCount;         // 本周聊天次数
    private Long monthChatCount;        // 本月聊天次数
    
    // 时间统计
    private LocalDateTime firstChatTime;    // 首次聊天时间
    private LocalDateTime lastChatTime;     // 最后聊天时间
    private Long totalChatDuration;         // 总聊天时长（秒）
    private Double avgChatDuration;         // 平均聊天时长（秒）
    
    // 活跃度统计
    private Integer activeDays;             // 活跃天数
    private Double chatFrequency;           // 聊天频率（次/天）
    private String activityLevel;           // 活跃度等级（高/中/低）
    
    // 构造函数
    public ChatStatisticsVO() {}
    
    public ChatStatisticsVO(String deviceId, Long studentId, String studentName, 
                           String agentId, String agentName) {
        this.deviceId = deviceId;
        this.studentId = studentId;
        this.studentName = studentName;
        this.agentId = agentId;
        this.agentName = agentName;
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
    
    public String getStudentName() {
        return studentName;
    }
    
    public void setStudentName(String studentName) {
        this.studentName = studentName;
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
    
    public Long getTodayChatCount() {
        return todayChatCount;
    }
    
    public void setTodayChatCount(Long todayChatCount) {
        this.todayChatCount = todayChatCount;
    }
    
    public Long getWeekChatCount() {
        return weekChatCount;
    }
    
    public void setWeekChatCount(Long weekChatCount) {
        this.weekChatCount = weekChatCount;
    }
    
    public Long getMonthChatCount() {
        return monthChatCount;
    }
    
    public void setMonthChatCount(Long monthChatCount) {
        this.monthChatCount = monthChatCount;
    }
    
    public LocalDateTime getFirstChatTime() {
        return firstChatTime;
    }
    
    public void setFirstChatTime(LocalDateTime firstChatTime) {
        this.firstChatTime = firstChatTime;
    }
    
    public LocalDateTime getLastChatTime() {
        return lastChatTime;
    }
    
    public void setLastChatTime(LocalDateTime lastChatTime) {
        this.lastChatTime = lastChatTime;
    }
    
    public Long getTotalChatDuration() {
        return totalChatDuration;
    }
    
    public void setTotalChatDuration(Long totalChatDuration) {
        this.totalChatDuration = totalChatDuration;
    }
    
    public Double getAvgChatDuration() {
        return avgChatDuration;
    }
    
    public void setAvgChatDuration(Double avgChatDuration) {
        this.avgChatDuration = avgChatDuration;
    }
    
    public Integer getActiveDays() {
        return activeDays;
    }
    
    public void setActiveDays(Integer activeDays) {
        this.activeDays = activeDays;
    }
    
    public Double getChatFrequency() {
        return chatFrequency;
    }
    
    public void setChatFrequency(Double chatFrequency) {
        this.chatFrequency = chatFrequency;
    }
    
    public String getActivityLevel() {
        return activityLevel;
    }
    
    public void setActivityLevel(String activityLevel) {
        this.activityLevel = activityLevel;
    }
    
    @Override
    public String toString() {
        return "ChatStatisticsVO{" +
                "deviceId='" + deviceId + '\'' +
                ", studentId=" + studentId +
                ", studentName='" + studentName + '\'' +
                ", agentId='" + agentId + '\'' +
                ", agentName='" + agentName + '\'' +
                ", totalChatCount=" + totalChatCount +
                ", userMessageCount=" + userMessageCount +
                ", aiMessageCount=" + aiMessageCount +
                ", todayChatCount=" + todayChatCount +
                ", weekChatCount=" + weekChatCount +
                ", monthChatCount=" + monthChatCount +
                ", firstChatTime=" + firstChatTime +
                ", lastChatTime=" + lastChatTime +
                ", totalChatDuration=" + totalChatDuration +
                ", avgChatDuration=" + avgChatDuration +
                ", activeDays=" + activeDays +
                ", chatFrequency=" + chatFrequency +
                ", activityLevel='" + activityLevel + '\'' +
                '}';
    }
}