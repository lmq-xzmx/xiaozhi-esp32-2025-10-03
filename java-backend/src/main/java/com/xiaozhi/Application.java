package com.xiaozhi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

/**
 * 小智设备学员绑定系统主类
 */
@SpringBootApplication
@ComponentScan(basePackages = "com.xiaozhi")
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}