#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

log()
{
    commoninfo=$1

	timestr=$(date '+%Y-%m-%d %H:%M:%S.%N '|cut -b 1-23)
        echo "[${timestr}]""${commoninfo}"

    return 0
}

log_error()
{
	errorinfo="$1"
	log "[ERROR] ${errorinfo}"
	return 0
}


log_warning()
{
        warninginfo="$1"
        log "[WARNING] ${warninginfo}"
        return 0
}

log_info()
{
        loginfo="$1"
        log "[INFO] ${loginfo}"
        return 0
}
