package com.xiaozhi.controller;

import com.xiaozhi.common.Result;
import com.xiaozhi.service.RedisSessionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 会话管理控制器
 */
@RestController
@RequestMapping("/session")
@CrossOrigin(origins = "*")
public class SessionController {
    
    @Autowired
    private RedisSessionService redisSessionService;
    
    /**
     * 创建新会话
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 创建结果
     */
    @PostMapping("/create")
    public Result<String> createSession(@RequestParam String deviceId, @RequestParam String sessionId) {
        try {
            boolean success = redisSessionService.createSession(deviceId, sessionId);
            if (success) {
                return Result.success("会话创建成功");
            } else {
                return Result.error("会话创建失败");
            }
        } catch (Exception e) {
            return Result.error("会话创建失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取设备当前会话
     * @param deviceId 设备ID
     * @return 当前会话ID
     */
    @GetMapping("/current/{deviceId}")
    public Result<String> getCurrentSession(@PathVariable String deviceId) {
        try {
            String sessionId = redisSessionService.getCurrentSession(deviceId);
            return Result.success(sessionId);
        } catch (Exception e) {
            return Result.error("获取当前会话失败: " + e.getMessage());
        }
    }
    
    /**
     * 设置设备当前会话
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 设置结果
     */
    @PostMapping("/current")
    public Result<String> setCurrentSession(@RequestParam String deviceId, @RequestParam String sessionId) {
        try {
            boolean success = redisSessionService.setCurrentSession(deviceId, sessionId);
            if (success) {
                return Result.success("当前会话设置成功");
            } else {
                return Result.error("当前会话设置失败");
            }
        } catch (Exception e) {
            return Result.error("当前会话设置失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取设备所有会话
     * @param deviceId 设备ID
     * @return 会话ID列表
     */
    @GetMapping("/list/{deviceId}")
    public Result<List<String>> getDeviceSessions(@PathVariable String deviceId) {
        try {
            List<String> sessions = redisSessionService.getDeviceSessions(deviceId);
            return Result.success(sessions);
        } catch (Exception e) {
            return Result.error("获取设备会话列表失败: " + e.getMessage());
        }
    }
    
    /**
     * 删除会话
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 删除结果
     */
    @DeleteMapping("/delete")
    public Result<String> deleteSession(@RequestParam String deviceId, @RequestParam String sessionId) {
        try {
            boolean success = redisSessionService.deleteSession(deviceId, sessionId);
            if (success) {
                return Result.success("会话删除成功");
            } else {
                return Result.error("会话删除失败");
            }
        } catch (Exception e) {
            return Result.error("会话删除失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取会话信息
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @return 会话信息
     */
    @GetMapping("/info/{deviceId}/{sessionId}")
    public Result<Map<String, Object>> getSessionInfo(@PathVariable String deviceId, @PathVariable String sessionId) {
        try {
            Map<String, Object> sessionInfo = redisSessionService.getSessionInfo(deviceId, sessionId);
            return Result.success(sessionInfo);
        } catch (Exception e) {
            return Result.error("获取会话信息失败: " + e.getMessage());
        }
    }
    
    /**
     * 更新会话信息
     * @param deviceId 设备ID
     * @param sessionId 会话ID
     * @param sessionInfo 会话信息
     * @return 更新结果
     */
    @PostMapping("/update")
    public Result<String> updateSessionInfo(@RequestParam String deviceId, 
                                           @RequestParam String sessionId, 
                                           @RequestBody Map<String, Object> sessionInfo) {
        try {
            boolean success = redisSessionService.updateSessionInfo(deviceId, sessionId, sessionInfo);
            if (success) {
                return Result.success("会话信息更新成功");
            } else {
                return Result.error("会话信息更新失败");
            }
        } catch (Exception e) {
            return Result.error("会话信息更新失败: " + e.getMessage());
        }
    }
}