/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it 
 * under the terms of CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_log.cpp
 * \brief Super Kernel Log Module Implementation (Merged File Logger)
 */

#include "sk_log.h"
#include "sk_common.h"
#include "securec.h"
#include "base/err_mgr.h"
#include <iostream>
#include <cstdarg>
#include <cstdlib>
#include <limits.h>

// ==================== Original Error Report Function ====================
constexpr size_t LIMIT_PREDEFINED_MESSAGE = 1024U;

void ReportErrorMessageInner(const std::string &code, const char* fmt, ...)
{
    std::vector<char> buf(LIMIT_PREDEFINED_MESSAGE, '\0');
    va_list argList;
    va_start(argList, fmt);
    auto ret = vsnprintf_s(buf.data(), LIMIT_PREDEFINED_MESSAGE, LIMIT_PREDEFINED_MESSAGE - 1U, fmt, argList);
    if (ret == -1) {
        SK_DLOGE("Construct error message failed, maybe the length of error message exceed limits: %zu", 
            LIMIT_PREDEFINED_MESSAGE);
    }
    va_end(argList);
    const std::vector<const char*> msgKey = {"message"};
    const std::vector<const char*> msgValue = {buf.data()};
    REPORT_PREDEFINED_ERR_MSG(code.c_str(), msgKey, msgValue);
}

// ==================== FileHandleManager Implementation ====================
namespace sk {
namespace logger {

bool FileHandleManager::RegisterFile(const std::string& name, const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (handles_.find(name) != handles_.end()) {
        return true; // Already registered
    }
    
    FileHandleInfo handle;
    handle.filePath = path;
    handle.fileStream.open(path, std::ios::app | std::ios::out);
    
    if (!handle.fileStream.is_open()) {
        SK_DLOGE("Failed to open log file: %s", path.c_str());
        return false;
    }
    
    handle.createTime = std::chrono::system_clock::now();
    
    // Get current file size for existing files
    handle.fileStream.seekp(0, std::ios::end);
    handle.currentSize = static_cast<size_t>(handle.fileStream.tellp());
    
    handles_.emplace(name, std::move(handle));
    return true;
}

bool FileHandleManager::SwitchToFile(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = handles_.find(name);
    if (it == handles_.end()) {
        SK_DLOGE("File handle not found: %s", name.c_str());
        return false;
    }
    
    currentHandle_ = name;
    return true;
}

void FileHandleManager::SwitchToDefault() {
    std::lock_guard<std::mutex> lock(mutex_);
    currentHandle_ = "default";
}

bool FileHandleManager::Write(const std::string& name, const std::string& content) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = handles_.find(name);
    if (it == handles_.end() || !it->second.fileStream.is_open()) {
        return false;
    }
    
    auto& handle = it->second;
    handle.fileStream << content;
    handle.fileStream.flush();
    
    handle.currentSize += content.size();
    handle.writeCount++;
    
    return true;
}

bool FileHandleManager::WriteToCurrent(const std::string& content) {
    return Write(currentHandle_, content);
}

std::string FileHandleManager::GetCurrentHandle() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return currentHandle_;
}

size_t FileHandleManager::GetFileSize(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = handles_.find(name);
    if (it == handles_.end()) {
        return 0;
    }
    return it->second.currentSize;
}

void FileHandleManager::CloseFile(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = handles_.find(name);
    if (it != handles_.end()) {
        if (it->second.fileStream.is_open()) {
            it->second.fileStream.flush();
            it->second.fileStream.close();
        }
        handles_.erase(it);
    }
}

bool FileHandleManager::InitializeDefault(const std::string& baseDir, pid_t pid, aclmdlRI model) {
    // Use unified path generator with model
    std::string dirPath = GetSkMetaPath(model);
    if (!CreateDirectoryRecursive(dirPath)) {
        return false;
    }
    
    // 为不同 model 创建唯一的 handle 名称
    std::string modelStr = ModelRIToString(model);
    std::string handleName = "model_" + SanitizePathComponent(modelStr);
    
    std::string defaultPath = dirPath + "/super_kernel.log";
    return RegisterFile(handleName, defaultPath);
}

// Thread-local current handle initialization
thread_local std::string FileHandleManager::currentHandle_ = "default";

// 线程局部 modelRI 初始化
thread_local aclmdlRI FileLogger::currentModelRI_ = nullptr;

FileHandleManager::FileHandleManager() {}

FileHandleManager::~FileHandleManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : handles_) {
        if (pair.second.fileStream.is_open()) {
            pair.second.fileStream.flush();
            pair.second.fileStream.close();
        }
    }
}

// ==================== LogContextGuard Implementation ====================
LogContextGuard::LogContextGuard(const std::string& fileName, const std::string& filePath)
    : previousHandle_(FileHandleManager::Instance().GetCurrentHandle())
    , active_(false)
{
    if (!FileHandleManager::Instance().RegisterFile(fileName, filePath)) {
        SK_DLOGE("Failed to register log file: %s", fileName.c_str());
        return;
    }
    
    if (!FileHandleManager::Instance().SwitchToFile(fileName)) {
        SK_DLOGE("Failed to switch to log file: %s", fileName.c_str());
        return;
    }
    
    active_ = true;
}

LogContextGuard::~LogContextGuard() {
    if (active_ && !previousHandle_.empty()) {
        // Restore to previous handle, not just default
        FileHandleManager::Instance().SwitchToFile(previousHandle_);
    }
}

LogContextGuard::LogContextGuard(LogContextGuard&& other) noexcept
    : previousHandle_(std::move(other.previousHandle_))
    , active_(other.active_)
{
    other.active_ = false;
}

LogContextGuard& LogContextGuard::operator=(LogContextGuard&& other) noexcept {
    if (this != &other) {
        if (active_) {
            FileHandleManager::Instance().SwitchToDefault();
        }
        previousHandle_ = std::move(other.previousHandle_);
        active_ = other.active_;
        other.active_ = false;
    }
    return *this;
}

// ==================== FileLogger Implementation ====================
FileLogger& FileLogger::Instance() {
    static FileLogger instance;
    return instance;
}

bool FileLogger::Initialize(const LoggerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 如果已初始化且 modelRI 相同，直接返回
    if (initialized_.load() && config_.modelRI == config.modelRI) {
        return true;
    }
    
    // 如果 modelRI 改变，需要为新 modelRI 创建日志文件
    bool isNewModel = (initialized_.load() && config_.modelRI != config.modelRI);
    
    config_ = config;
    
    if (!config_.enabled) {
        SK_DLOGI("File logger is disabled");
        initialized_.store(true);
        return true;
    }
    
    // Use common utility to create sk_meta directory structure
    std::string logDir = CreateSkMetaDirectory(config_.modelRI);
    if (logDir.empty() && config_.enabled) {
        SK_DLOGE("Failed to create sk_meta directory");
        return false;
    }
    
    // Extract PID from created directory
    pid_ = getpid();
    
    // 为新 modelRI 注册日志文件
    std::string modelStr = ModelRIToString(config_.modelRI);
    std::string handleName = "model_" + SanitizePathComponent(modelStr);
    std::string defaultPath = logDir + "/super_kernel.log";
    
    if (!FileHandleManager::Instance().RegisterFile(handleName, defaultPath)) {
        SK_DLOGE("Failed to register log file for model: %s", modelStr.c_str());
        return false;
    }
    
    // 切换到新的日志文件
    FileHandleManager::Instance().SwitchToFile(handleName);
    
    initialized_.store(true);
    
    // Convert to absolute path for better visibility
    char absPath[PATH_MAX] = {0};  // Initialize buffer with zeros
    if (realpath(logDir.c_str(), absPath) != nullptr) {
        // Ensure null termination
        absPath[PATH_MAX - 1] = '\0';
        SK_DLOGI("File logger initialized: dir=%s, model=%s", absPath, modelStr.c_str());
    } else {
        SK_DLOGI("File logger initialized: dir=%s, model=%s", logDir.c_str(), modelStr.c_str());
    }
    return true;
}

// 获取当前有效的 modelRI
aclmdlRI FileLogger::GetEffectiveModelRI() const {
    // 优先使用线程局部变量
    if (currentModelRI_ != nullptr) {
        return currentModelRI_;
    }
    // 其次使用配置中的 modelRI
    return config_.modelRI;
}

bool FileLogger::RegisterLogFile(const std::string& name, const std::string& subPath) {
    if (!initialized_.load() || !config_.enabled) {
        return false;
    }
    
    // 使用 GetEffectiveModelRI 获取当前有效的 modelRI
    aclmdlRI useModel = GetEffectiveModelRI();
    
    // Use unified path generator
    std::string basePath = GetSkMetaPath(useModel);
    std::string filePath = basePath + "/" + name;
    
    // If subPath provided, insert it before filename
    if (!subPath.empty()) {
        filePath = basePath + "/" + SanitizePathComponent(subPath) + "/" + name;
    }
    
    // Ensure directory exists using common utility
    size_t lastSlash = filePath.find_last_of('/');
    if (lastSlash != std::string::npos) {
        std::string dir = filePath.substr(0, lastSlash);
        if (!CreateDirectoryRecursive(dir)) {
            return false;
        }
    }
    
    // 为不同 model 创建唯一的 handle 名称
    std::string modelStr = ModelRIToString(useModel);
    std::string handleName = "model_" + SanitizePathComponent(modelStr) + "_" + name;
    
    return FileHandleManager::Instance().RegisterFile(handleName, filePath);
}

bool FileLogger::SwitchToFile(const std::string& name) {
    if (!initialized_.load() || !config_.enabled) {
        return false;
    }
    return FileHandleManager::Instance().SwitchToFile(name);
}

void FileLogger::SwitchToDefault() {
    FileHandleManager::Instance().SwitchToDefault();
}

std::unique_ptr<LogContextGuard> FileLogger::CreateContext(const std::string& fileName,
                                                            aclmdlRI model) {
    if (!initialized_.load() || !config_.enabled) {
        return nullptr;
    }
    
    // Use provided model or GetEffectiveModelRI
    aclmdlRI useModel = (model != nullptr) ? model : GetEffectiveModelRI();
    
    // Use common utility to create directory
    std::string dirPath = CreateSkMetaDirectory(useModel);
    
    if (dirPath.empty()) {
        SK_DLOGE("Failed to create directory for context");
        return nullptr;
    }
    
    std::string filePath = dirPath + "/" + fileName;
    
    // 为不同 model 创建唯一的 handle 名称
    std::string modelStr = ModelRIToString(useModel);
    std::string handleName = "model_" + SanitizePathComponent(modelStr) + "_" + fileName;
    
    return std::make_unique<LogContextGuard>(handleName, filePath);
}

void FileLogger::SetEnabled(bool enabled) {
    config_.enabled = enabled;
}

bool FileLogger::IsEnabled() const {
    return config_.enabled;
}

void FileLogger::SetMinLevel(LogLevel level) {
    config_.minLevel = level;
}

void FileLogger::SetModelRI(aclmdlRI modelRI) {
    config_.modelRI = modelRI;
}

bool FileLogger::IsInitialized() const {
    return initialized_.load();
}

std::string FileLogger::FormatMessage(LogLevel level, const char* funcName,
                                      const char* fileName, int lineNum,
                                      const char* format, ...) {
    std::ostringstream oss;
    
    // Timestamp
    if (config_.enableTimestamp) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        oss << " ";
    }
    
    // Process/Thread ID
    if (config_.enablePidTid) {
        oss << "[" << pid_ << ":" << std::this_thread::get_id() << "] ";
    }
    
    // Log level
    oss << "[" << LogLevelToString(level) << "] ";
    
    // File name and line number (extract just the filename from full path)
    if (fileName != nullptr) {
        const char* baseName = strrchr(fileName, '/');
        if (baseName != nullptr) {
            baseName++; // Skip '/'
        } else {
            baseName = fileName;
        }
        oss << "[" << baseName << ":" << lineNum << "] ";
    }
    
    // Function name
    if (funcName != nullptr) {
        oss << "[" << funcName << "] ";
    }
    
    // Log content - use secure function
    va_list args;
    va_start(args, format);
    char buffer[4096];
    int ret = vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, format, args);
    va_end(args);
    
    if (ret < 0) {
        // Format error, use error message
        oss << "[FORMAT_ERROR] ";
        oss << "Failed to format log message\n";
    } else {
        oss << buffer << "\n";
    }
    
    return oss.str();
}

void FileLogger::WriteLog(const std::string& message) {
    if (message.empty()) {
        return;
    }
    
    // 获取当前 handle，判断是否在 LogContextGuard 上下文中
    std::string currentHandle = FileHandleManager::Instance().GetCurrentHandle();
    std::string targetHandle;
    
    // 如果当前 handle 是 "default" 或以 "model_" 开头但没有额外后缀，
    // 说明没有通过 LogContextGuard 切换到非 default 日志
    // 此时需要根据当前 modelRI 计算正确的 handle
    if (currentHandle == "default" || currentHandle.find('_') == currentHandle.rfind('_')) {
        // 没有在 LogContextGuard 上下文中，使用 modelRI 计算 handle
        aclmdlRI useModel = GetEffectiveModelRI();
        std::string modelStr = ModelRIToString(useModel);
        targetHandle = "model_" + SanitizePathComponent(modelStr);
    } else {
        // 在 LogContextGuard 上下文中，使用当前 handle
        targetHandle = currentHandle;
    }
    
    // Long log segmentation handling
    if (message.size() > config_.maxLineLength) {
        size_t offset = 0;
        size_t segmentNum = 0;
        
        while (offset < message.size()) {
            size_t length = std::min(config_.maxLineLength, message.size() - offset);
            std::string segment;
            
            if (segmentNum == 0) {
                segment = message.substr(offset, length);
            } else {
                segment = "[CONT:" + std::to_string(segmentNum) + "] " + 
                          message.substr(offset, length);
            }
            
            FileHandleManager::Instance().Write(targetHandle, segment);
            offset += length;
            segmentNum++;
        }
    } else {
        FileHandleManager::Instance().Write(targetHandle, message);
    }
}

} // namespace logger
} // namespace sk

// ==================== Global Initialization Helper Functions ====================

bool InitializeSkFileLogger(bool enabled, aclmdlRI model,
                             sk::logger::LogLevel minLevel) {
    sk::logger::LoggerConfig config;
    config.enabled = enabled;
    config.modelRI = model;
    config.minLevel = minLevel;
    
    if (!sk::logger::FileLogger::Instance().Initialize(config)) {
        // Disable file logging on initialization failure
        sk::logger::FileLogger::Instance().SetEnabled(false);
        return false;
    }
    return true;
}
