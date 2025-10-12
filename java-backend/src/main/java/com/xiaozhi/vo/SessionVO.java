package com.xiaozhi.vo;

import java.time.LocalDateTime;

/**
 * 会话信息VO
 */
public class SessionVO {
    private String sessionId;
    private Integer chatCount;
    private LocalDateTime createdAt;
    private LocalDateTime lastMessageAt;

    public SessionVO() {}

    public SessionVO(String sessionId, Integer chatCount, LocalDateTime createdAt, LocalDateTime lastMessageAt) {
        this.sessionId = sessionId;
        this.chatCount = chatCount;
        this.createdAt = createdAt;
        this.lastMessageAt = lastMessageAt;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public Integer getChatCount() {
        return chatCount;
    }

    public void setChatCount(Integer chatCount) {
        this.chatCount = chatCount;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getLastMessageAt() {
        return lastMessageAt;
    }

    public void setLastMessageAt(LocalDateTime lastMessageAt) {
        this.lastMessageAt = lastMessageAt;
    }
}