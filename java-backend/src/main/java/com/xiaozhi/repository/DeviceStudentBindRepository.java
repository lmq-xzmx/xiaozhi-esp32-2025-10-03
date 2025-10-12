package com.xiaozhi.repository;

import com.xiaozhi.entity.DeviceStudentBind;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * 设备学员绑定Repository接口
 */
@Repository
public interface DeviceStudentBindRepository extends JpaRepository<DeviceStudentBind, Long> {
    
    /**
     * 根据设备ID查找绑定关系
     */
    Optional<DeviceStudentBind> findByDeviceIdAndStatus(String deviceId, Integer status);
    
    /**
     * 根据学员ID查找绑定关系
     */
    List<DeviceStudentBind> findByStudentIdAndStatus(Long studentId, Integer status);
    
    /**
     * 根据智能体ID查找绑定关系
     */
    @Query("SELECT dsb FROM DeviceStudentBind dsb WHERE dsb.agentId = :agentId AND dsb.status = 1")
    List<DeviceStudentBind> findByAgentIdAndActiveStatus(@Param("agentId") String agentId);
    
    /**
     * 分页查询绑定列表
     */
    @Query(value = "SELECT dsb.*, d.device_alias, d.mac_address, s.real_name as student_name " +
           "FROM device_student_bind dsb " +
           "LEFT JOIN devices d ON dsb.device_id = d.device_id " +
           "LEFT JOIN students s ON dsb.student_id = s.id " +
           "WHERE dsb.agent_id = :agentId " +
           "AND (:keyword IS NULL OR d.device_alias LIKE %:keyword% OR s.real_name LIKE %:keyword%) " +
           "ORDER BY dsb.create_time DESC " +
           "LIMIT :offset, :size", nativeQuery = true)
    List<Object[]> findDeviceStudentListWithPagination(@Param("agentId") String agentId, 
                                                       @Param("keyword") String keyword,
                                                       @Param("offset") int offset, 
                                                       @Param("size") int size);
    
    /**
     * 统计绑定数量
     */
    @Query(value = "SELECT COUNT(*) FROM device_student_bind dsb " +
           "LEFT JOIN devices d ON dsb.device_id = d.device_id " +
           "LEFT JOIN students s ON dsb.student_id = s.id " +
           "WHERE dsb.agent_id = :agentId " +
           "AND (:keyword IS NULL OR d.device_alias LIKE %:keyword% OR s.real_name LIKE %:keyword%)", 
           nativeQuery = true)
    long countDeviceStudentList(@Param("agentId") String agentId, @Param("keyword") String keyword);
}