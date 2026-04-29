/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Lightweight ge_log.h wrapper for graph_metadef/graph sources.
 * Provides GELOGE/GELOGW/GELOGI/GELOGD macros directly via dlog_pub.h
 * without pulling in the Ascend error-codes/tensor chain that conflicts
 * with the af:: namespace migration.
 */

#ifndef COMMON_GRAPH_DEBUG_GE_LOG_H_
#define COMMON_GRAPH_DEBUG_GE_LOG_H_

#include <cinttypes>
#include <cstdint>
#include <string>

#include "base/dlog_pub.h"
#include "base/err_msg.h"

#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#else
#include "mmpa/mmpa_api.h"
#endif

#ifndef GE_MODULE_NAME
#define GE_MODULE_NAME static_cast<int32_t>(45)
#endif
#ifndef GE_MODULE_NAME_U16
#define GE_MODULE_NAME_U16 static_cast<uint16_t>(45)
#endif

class GeLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

namespace ge_log_impl {
inline uint64_t GetTid() {
  return GeLog::GetTid();
}
}  // namespace ge_log_impl

#ifndef GE_GET_ERRORNO_STR
inline std::string GE_GET_ERRORNO_STR_FUNC(uint32_t) { return ""; }
#define GE_GET_ERRORNO_STR(value) GE_GET_ERRORNO_STR_FUNC(static_cast<uint32_t>(value))
#endif

#ifndef GELOGE
#define GELOGE(ERROR_CODE, fmt, ...)                                                         \
  do {                                                                                       \
    dlog_error(GE_MODULE_NAME, "%" PRIu64 " %s: ErrorNo: %u(%s)" fmt,                     \
               ge_log_impl::GetTid(), &__FUNCTION__[0U],                                     \
               static_cast<uint32_t>(ERROR_CODE),                                            \
               GE_GET_ERRORNO_STR(ERROR_CODE).c_str(), ##__VA_ARGS__);                       \
  } while (false)
#endif

#ifndef GELOGW
#define GELOGW(fmt, ...)                                                                     \
  do {                                                                                       \
    dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt,                                      \
              ge_log_impl::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                      \
  } while (false)
#endif

#ifndef GELOGI
#define GELOGI(fmt, ...)                                                                     \
  do {                                                                                       \
    dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt,                                      \
              ge_log_impl::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                      \
  } while (false)
#endif

#ifndef GELOGD
#define GELOGD(fmt, ...)                                                                     \
  do {                                                                                       \
    dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt,                                     \
               ge_log_impl::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                     \
  } while (false)
#endif

#ifndef GE_LOG_ERROR
#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)                                        \
  do {                                                                                       \
    dlog_error((MOD_NAME), "%" PRIu64 " %s: ErrorNo: %u(%s)" fmt,                         \
               ge_log_impl::GetTid(), &__FUNCTION__[0U],                                     \
               static_cast<uint32_t>(ERROR_CODE),                                            \
               GE_GET_ERRORNO_STR(ERROR_CODE).c_str(), ##__VA_ARGS__);                       \
  } while (false)
#endif

#ifndef GE_LOGE
#define GE_LOGE(fmt, ...) GE_LOG_ERROR(GE_MODULE_NAME, 0U, fmt, ##__VA_ARGS__)
#endif

#ifndef GE_CHK_BOOL_RET_STATUS
#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...)  \
  do {                                              \
    const bool _b = (expr);                         \
    if (!_b) {                                      \
      REPORT_INNER_ERR_MSG("E19999", __VA_ARGS__); \
      GELOGE((_status), __VA_ARGS__);               \
      return (_status);                             \
    }                                               \
  } while (false)
#endif

#ifndef GE_CHK_BOOL_RET_STATUS_NOLOG
#define GE_CHK_BOOL_RET_STATUS_NOLOG(expr, _status, ...)  \
  do {                                                    \
    const bool _b = (expr);                               \
    if (!_b) {                                            \
      return (_status);                                   \
    }                                                     \
  } while (false)
#endif

#ifndef GE_CHK_BOOL_RET_SPECIAL_STATUS
#define GE_CHK_BOOL_RET_SPECIAL_STATUS(expr, special_return, ...)  \
  do {                                                             \
    const bool _b = (expr);                                        \
    if (_b) {                                                      \
      GELOGI(__VA_ARGS__);                             \
      return (special_return);                                     \
    }                                                              \
  } while (false)
#endif

#ifndef GE_IF_BOOL_EXEC
#define GE_IF_BOOL_EXEC(expr, exec_expr)  \
  {                                       \
    if (expr) {                           \
      exec_expr;                          \
    }                                     \
  }
#endif

#ifndef GE_CHK_BOOL_EXEC
#define GE_CHK_BOOL_EXEC(expr, exec_expr, ...)  \
  {                                             \
    const bool _b = (expr);                     \
    if (!_b) {                                  \
      GELOGE(ge::FAILED, __VA_ARGS__);          \
      exec_expr;                                \
    }                                           \
  }
#endif

#ifndef GE_CHK_STATUS_RET
#define GE_CHK_STATUS_RET(expr, ...)           \
  do {                                         \
    const uint32_t _chk_status = (expr);       \
    if (_chk_status != 0U) {                   \
      GELOGE(0xFFFFFFFFU, __VA_ARGS__);        \
      return _chk_status;                      \
    }                                          \
  } while (false)
#endif

#ifndef GE_CHK_GRAPH_STATUS_RET
#define GE_CHK_GRAPH_STATUS_RET(expr, ...)     \
  do {                                         \
    if ((expr) != 0U) {                        \
      GELOGE(0xFFFFFFFFU, __VA_ARGS__);        \
      return 0xFFFFFFFFU;                      \
    }                                          \
  } while (false)
#endif

#ifndef GE_LOGI_IF
#define GE_LOGI_IF(condition, ...)  \
  if ((condition)) {                \
    GELOGI(__VA_ARGS__);            \
  }
#endif

#ifndef GE_LOGW_IF
#define GE_LOGW_IF(condition, ...)  \
  if ((condition)) {                \
    GELOGW(__VA_ARGS__);            \
  }
#endif

#ifndef GE_LOGE_IF
#define GE_LOGE_IF(condition, ...)    \
  if ((condition)) {                  \
    GELOGE(0xFFFFFFFFU, __VA_ARGS__); \
  }
#endif

#ifndef GE_CHK_STATUS_RET_NOLOG
#define GE_CHK_STATUS_RET_NOLOG(expr)   \
  do {                                  \
    const uint32_t _chk_status = (expr);\
    if (_chk_status != 0U) {            \
      return _chk_status;               \
    }                                   \
  } while (false)
#endif

#ifndef IsLogEnable
inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  return CheckLogLevel(module_name, log_level) == 1;
}
#endif

#ifndef GEEVENT
#define GEEVENT(fmt, ...) GELOGI(fmt, ##__VA_ARGS__)
#endif

#ifndef GE_CHK_STATUS
#define GE_CHK_STATUS(expr, ...)             \
  do {                                       \
    const uint32_t _chk_status = (expr);     \
    if (_chk_status != 0U) {                 \
      GELOGE(_chk_status, __VA_ARGS__);      \
    }                                        \
  } while (false)
#endif

#ifndef GE_CHK_BOOL_ONLY_LOG
#define GE_CHK_BOOL_ONLY_LOG(expr, ...)   \
  do {                                    \
    const bool _b = (expr);               \
    if (!_b) {                            \
      GELOGI(__VA_ARGS__);                \
    }                                     \
  } while (false)
#endif

#ifndef GE_MAKE_SHARED
#define GE_MAKE_SHARED(exec_expr0, exec_expr1)     \
  try {                                            \
    exec_expr0;                                    \
  } catch (const std::bad_alloc &) {               \
    GELOGE(0xFFFFFFFFU, "Make shared failed");    \
    exec_expr1;                                    \
  }
#endif

#endif  // COMMON_GRAPH_DEBUG_GE_LOG_H_
