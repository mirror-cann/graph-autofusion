/**
 * Shadow header: ensures GE_FUNC_VISIBILITY is defined before the system scope_guard.h
 * is pulled in, preventing the 'ScopeGuard in namespace ge does not name a type' error.
 */
#ifndef AUTOFUSE_COMMON_GE_COMMON_SCOPE_GUARD_SHADOW_H_
#define AUTOFUSE_COMMON_GE_COMMON_SCOPE_GUARD_SHADOW_H_

#ifndef GE_FUNC_VISIBILITY
#if defined(HOST_VISIBILITY) && defined(__GNUC__)
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include_next "common/ge_common/scope_guard.h"

#endif
