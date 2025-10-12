package com.xiaozhi.dto;

/**
 * 聊天记录下载DTO
 */
public class ChatHistoryDownloadDTO {
    private String downloadUrl;
    private String fileName;
    private Long fileSize;
    private String format; // csv, json, txt
    private String sessionId;
    private String deviceId;

    public ChatHistoryDownloadDTO() {}

    public ChatHistoryDownloadDTO(String downloadUrl, String fileName, Long fileSize, String format) {
        this.downloadUrl = downloadUrl;
        this.fileName = fileName;
        this.fileSize = fileSize;
        this.format = format;
    }

    public String getDownloadUrl() {
        return downloadUrl;
    }

    public void setDownloadUrl(String downloadUrl) {
        this.downloadUrl = downloadUrl;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public Long getFileSize() {
        return fileSize;
    }

    public void setFileSize(Long fileSize) {
        this.fileSize = fileSize;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public String getDeviceId() {
        return deviceId;
    }

    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }
}