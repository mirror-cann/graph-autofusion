/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "rt_sk_intf.h"
#include "rt_intf.h"

#if __has_include("acl/acl.h")
#include "acl/acl.h"
#endif

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <mutex>
#include "securec.h"

namespace {
constexpr aclError kAclSuccess = 0;
constexpr aclError kAclError = -1;
}

class RtsObjectStub {
public:

	void StartCapture() {
		std::lock_guard<std::mutex> lock(mu_);
		capturing_ = true;
		has_snapshot_ = false;
		initialized_ = false;
		ReleaseKernelArgsLocked();
		buckets_.clear();
	}

	void StopCapture() {
		std::lock_guard<std::mutex> lock(mu_);
		if (!capturing_) {
			return;
		}
		RefreshFromRuntimeLocked();
		capturing_ = false;
		has_snapshot_ = true;
		initialized_ = true;
	}

	void Capture() {
		StartCapture();
		StopCapture();
	}

    void Replay() {
        // No-op for stub
    }

	static RtsObjectStub& Instance() {
		static RtsObjectStub inst;
		return inst;
	}

	RtsObjectStub(const RtsObjectStub&) = delete;
	RtsObjectStub& operator=(const RtsObjectStub&) = delete;
	RtsObjectStub(RtsObjectStub&&) = delete;
	RtsObjectStub& operator=(RtsObjectStub&&) = delete;

	void RefreshFromRuntime() {
		std::lock_guard<std::mutex> lock(mu_);
		if (capturing_) {
			return;
		}
		RefreshFromRuntimeLocked();
		has_snapshot_ = true;
		initialized_ = true;
	}

	aclrtStream GetStream(uint32_t index) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (index >= buckets_.size()) {
			return nullptr;
		}
		return buckets_[index]->handle;
	}

	uint32_t GetStreamCount() {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		return static_cast<uint32_t>(buckets_.size());
	}

	void GetTasks(aclrtTask* tasks, uint32_t* numTasks) {
		GetTasksForStream(nullptr, tasks, numTasks);
	}

	void GetTasksForStream(aclrtStream stream, aclrtTask* tasks, uint32_t* numTasks) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!numTasks) {
			return;
		}
		const StreamBucket* bucket = nullptr;
		if (stream != nullptr) {
			bucket = FindBucket(stream);
			if (!bucket) {
				*numTasks = 0;
				return;
			}
		}

		if (!bucket) {
			uint32_t total = 0;
			for (const auto& item : buckets_) {
				total += static_cast<uint32_t>(item->tasks.size());
			}
			*numTasks = total;
			if (tasks) {
				uint32_t offset = 0;
				for (const auto& item : buckets_) {
					for (const auto& t : item->tasks) {
						tasks[offset++] = t;
					}
				}
			}
			return;
		}

		*numTasks = static_cast<uint32_t>(bucket->tasks.size());
		if (tasks) {
			for (size_t i = 0; i < bucket->tasks.size(); ++i) {
				tasks[i] = bucket->tasks[i];
			}
		}
	}

	bool GetTaskType(aclrtTask task, aclrtTaskType* type) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!type) {
			return false;
		}
		const aclrtTask* stored = FindTask(task);
		*type = stored ? stored->type : task.type;
		return true;
	}

	bool GetKernelParams(aclrtTask task, aclrtTaskKernelParams* params) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!params) {
			return false;
		}
		const aclrtTask* stored = FindTask(task);
		const aclrtTask& src = stored ? *stored : task;
		if (src.type != ACL_RT_TASK_KERNEL) {
			return false;
		}
		*params = src.kernel;
		return true;
	}


	bool SetKernelParams(aclrtTask task, const aclrtTaskKernelParams* params) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!params) {
			return false;
		}
		auto* stored = FindTaskMutable(task);
		if (!stored || stored->type != ACL_RT_TASK_KERNEL) {
			return false;
		}
		stored->kernel = *params;
		return true;
	}

bool GetEventParams(aclrtTask task, aclrtTaskEventParams* params) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!params) {
			return false;
		}
		const aclrtTask* stored = FindTask(task);
		const aclrtTask& src = stored ? *stored : task;
		if (src.type != ACL_RT_TASK_EVENT_RECORD &&
			src.type != ACL_RT_TASK_EVENT_WAIT &&
			src.type != ACL_RT_TASK_EVENT_RESET) {
			return false;
		}
		*params = src.event;
		return true;
	}

bool SetEventParams(aclrtTask task, const aclrtTaskEventParams* params) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!params) {
			return false;
		}
		auto* stored = FindTaskMutable(task);
		if (!stored || (stored->type != ACL_RT_TASK_EVENT_RECORD &&
			stored->type != ACL_RT_TASK_EVENT_WAIT &&
			stored->type != ACL_RT_TASK_EVENT_RESET)) {
			return false;
		}
		stored->event = *params;
		return true;
	}

bool GetMemValueParams(aclrtTask task, aclrtMemValueParams* params) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!params) {
			return false;
		}
		const aclrtTask* stored = FindTask(task);
		const aclrtTask& src = stored ? *stored : task;
		if (src.type != ACL_RT_TASK_VALUE_WRITE &&
			src.type != ACL_RT_TASK_VALUE_WAIT) {
			return false;
		}
		*params = src.memValue;
		return true;
	}

bool SetMemValueParams(aclrtTask task, const aclrtMemValueParams* params) {
		std::lock_guard<std::mutex> lock(mu_);
		EnsureInitializedLocked();
		if (!params) {
			return false;
		}
		auto* stored = FindTaskMutable(task);
		if (!stored || (stored->type != ACL_RT_TASK_VALUE_WRITE &&
			stored->type != ACL_RT_TASK_VALUE_WAIT)) {
			return false;
		}
		stored->memValue = *params;
		return true;
	}

private:
	RtsObjectStub() = default;

	void ReleaseKernelArgsLocked() {
		for (auto& bucket : buckets_) {
			for (auto& task : bucket->tasks) {
				if (task.type != ACL_RT_TASK_KERNEL) {
					continue;
				}
				if (task.kernel.devArgs) {
					(void)aclrtFree(task.kernel.devArgs);
					task.kernel.devArgs = nullptr;
					task.kernel.argsSize = 0;
				}
			}
		}
	}

	void* AllocKernelArgs(const void* hostArgs, size_t argSize) {
		if (!hostArgs || argSize == 0) {
			return nullptr;
		}
		void* devArgs = nullptr;
		if (aclrtMalloc(&devArgs, argSize, ACL_MEM_MALLOC_HUGE_FIRST) != kAclSuccess) {
			return nullptr;
		}
		if (aclrtMemcpy(devArgs, argSize, hostArgs, argSize, ACL_MEMCPY_HOST_TO_DEVICE) != kAclSuccess) {
			(void)aclrtFree(devArgs);
			return nullptr;
		}
		return devArgs;
	}

	void EnsureInitializedLocked() {
		if (has_snapshot_ || capturing_) {
			return;
		}
		RefreshFromRuntimeLocked();
		has_snapshot_ = true;
		initialized_ = true;
	}

	void RefreshFromRuntimeLocked() {
		rt_info* info = nullptr;
		size_t count = 0;
		rt_get_kernel_info(&info, &count);

		ReleaseKernelArgsLocked();
		buckets_.clear();
		buckets_.reserve(count);
		std::unordered_map<uint64_t, size_t> bucketIndex;

		uint32_t nextTaskId = 0;
		for (size_t i = 0; i < count; ++i) {
			auto it = bucketIndex.find(info[i].stream_id);
			if (it == bucketIndex.end()) {
				auto bucket = std::make_unique<StreamBucket>();
				bucket->stream_id = info[i].stream_id;
				bucket->handle = reinterpret_cast<aclrtStream>(info[i].stream_id);
				buckets_.push_back(std::move(bucket));
				bucketIndex[info[i].stream_id] = buckets_.size() - 1;
				it = bucketIndex.find(info[i].stream_id);
			}

			auto& bucket = buckets_[it->second];
			aclrtTask rec{};
			rec.task_id = nextTaskId++;
			switch (static_cast<rt_task_type>(info[i].task_type)) {
				case RT_TASK_EVENT_RECORD:
				case RT_TASK_EVENT_WAIT:
				case RT_TASK_EVENT_RESET:
					rec.type = static_cast<aclrtTaskType>(info[i].task_type);
					rec.event.type = rec.type;
					rec.event.flag = ACL_RT_TASK_VALID;
					rec.event.eventType = static_cast<aclrtEventType>(info[i].event_type);
					rec.event.eventId = info[i].event_id;
					rec.event.eventAddr = info[i].event_addr;
					rec.event.value = info[i].value;
					rec.event.valueSize = info[i].value_size;
					rec.event.waitFlag = info[i].wait_flag;
					rec.event.pExtend = nullptr;
					break;
				case RT_TASK_VALUE_WRITE:
				case RT_TASK_VALUE_WAIT:
					rec.type = static_cast<aclrtTaskType>(info[i].task_type);
					rec.memValue.type = rec.type;
					rec.memValue.flag = ACL_RT_TASK_VALID;
					rec.memValue.valueAddr = info[i].value_addr;
					rec.memValue.value = info[i].value;
					rec.memValue.valueSize = info[i].value_size;
					rec.memValue.waitFlag = info[i].wait_flag;
					rec.memValue.pExtend = nullptr;
					break;
				case RT_TASK_DEFAULT:
					rec.type = ACL_RT_TASK_DEFAULT;
					rec.def.type = ACL_RT_TASK_DEFAULT;
					rec.def.flag = ACL_RT_TASK_VALID;
					rec.def.pExtend = nullptr;
					break;
				case RT_TASK_KERNEL:
				default:
					rec.type = ACL_RT_TASK_KERNEL;
					rec.kernel.type = ACL_RT_TASK_KERNEL;
					rec.kernel.flag = ACL_RT_TASK_VALID;
					rec.kernel.funcHandle = reinterpret_cast<aclrtFuncHandle>(
						const_cast<char*>(info[i].func_name));
					rec.kernel.binHandle = reinterpret_cast<aclrtBinHandle>(info[i].bin_hdl);
					rec.kernel.cfg = nullptr;
					rec.kernel.argsHandle = nullptr;
					rec.kernel.taskGrp = nullptr;
					rec.kernel.devArgs = AllocKernelArgs(info[i].arg_data, info[i].arg_size);
					rec.kernel.argsSize = rec.kernel.devArgs ? info[i].arg_size : 0;
					rec.kernel.opInfoPtr = nullptr;
					rec.kernel.opInfoSize = 0;
					rec.kernel.numBlocks = info[i].numBlocks;
					rec.kernel.func_name = info[i].func_name;
					rec.kernel.sk_bin_hdl = info[i].bin_hdl;
					rec.kernel.sk_kernel_type = info[i].kernel_type;
					rec.kernel.sk_task_ratio[0] = info[i].task_ratio[0];
					rec.kernel.sk_task_ratio[1] = info[i].task_ratio[1];
					rec.kernel.pExtend = nullptr;
					break;
			}

			bucket->tasks.push_back(rec);
		}

		initialized_ = true;
	}

	struct StreamBucket {
		uint64_t stream_id = 0;
		aclrtStream handle = nullptr;
		std::vector<aclrtTask> tasks;
		int reserved = 0;
	};

	const aclrtTask* FindTask(const aclrtTask& task) const {
		for (const auto& bucket : buckets_) {
			for (const auto& item : bucket->tasks) {
				if (item.task_id == task.task_id) {
					return &item;
				}
			}
		}
		return nullptr;
	}

	aclrtTask* FindTaskMutable(const aclrtTask& task) {
		for (auto& bucket : buckets_) {
			for (auto& item : bucket->tasks) {
				if (item.task_id == task.task_id) {
					return &item;
				}
			}
		}
		return nullptr;
	}

	const StreamBucket* FindBucket(aclrtStream stream) const {
		for (const auto& item : buckets_) {
			if (item->handle == stream) {
				return item.get();
			}
		}
		return nullptr;
	}

	std::vector<std::unique_ptr<StreamBucket>> buckets_{};
	bool initialized_ = false;
	bool capturing_ = false;
	bool has_snapshot_ = false;
	std::mutex mu_;
};

extern "C" {

aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams) {
	(void)modelRI;
	auto& stub = RtsObjectStub::Instance();
	if (!numStreams) {
		return kAclError;
	}
	const uint32_t count = stub.GetStreamCount();
	*numStreams = count;
	if (streams) {
		std::vector<aclrtStream> tmp(count);
		for (uint32_t i = 0; i < count; ++i) {
			tmp[i] = stub.GetStream(i);
		}
		if (count > 0) {
			errno_t err = memcpy_s(streams, sizeof(aclrtStream) * count, tmp.data(), sizeof(aclrtStream) * count);
			if (err != 0) {
				printf("[error] memcpy_s streams failed\n");
				return kAclError;
			}
		}
	}
	return kAclSuccess;
}

aclError aclrtStreamGetTasks(aclrtStream stream, aclrtTask *tasks, uint32_t *numTasks) {
	auto& stub = RtsObjectStub::Instance();
	if (!numTasks) {
		return kAclError;
	}
	if (!tasks) {
		stub.GetTasksForStream(stream, nullptr, numTasks);
		return kAclSuccess;
	}
	uint32_t count = 0;
	stub.GetTasksForStream(stream, nullptr, &count);
	std::vector<aclrtTask> tmp(count);
	if (count > 0) {
		stub.GetTasksForStream(stream, tmp.data(), &count);
		errno_t err = memcpy_s(tasks, sizeof(aclrtTask) * count, tmp.data(), sizeof(aclrtTask) * count);
		if (err != 0) {
			printf("[error] memcpy_s tasks failed\n");
			return kAclError;
		}
	}
	*numTasks = count;
	return kAclSuccess;
}

aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type) {
	return RtsObjectStub::Instance().GetTaskType(task, type) ? kAclSuccess : kAclError;
}

aclError aclrtTaskGetKernelParams(aclrtTask task, aclrtTaskKernelParams *params) {
	return RtsObjectStub::Instance().GetKernelParams(task, params) ? kAclSuccess : kAclError;
}

aclError aclrtTaskSetKernelParams(aclrtTask task, aclrtTaskKernelParams *params) {
	return RtsObjectStub::Instance().SetKernelParams(task, params) ? kAclSuccess : kAclError;
}

aclError aclrtTaskGetEventParams(aclrtTask task, aclrtTaskEventParams *params) {
	return RtsObjectStub::Instance().GetEventParams(task, params) ? kAclSuccess : kAclError;
}

aclError aclrtTaskSetEventParams(aclrtTask task, aclrtTaskEventParams *params) {
	return RtsObjectStub::Instance().SetEventParams(task, params) ? kAclSuccess : kAclError;
}

aclError aclrtTaskGetMemValueParams(aclrtTask task, aclrtMemValueParams *params) {
	return RtsObjectStub::Instance().GetMemValueParams(task, params) ? kAclSuccess : kAclError;
}

aclError aclrtTaskSetMemValueParams(aclrtTask task, aclrtMemValueParams *params) {
	return RtsObjectStub::Instance().SetMemValueParams(task, params) ? kAclSuccess : kAclError;
}

aclError aclmdlRIUpdate(aclmdlRI modelRI) {
	(void)modelRI;
	RtsObjectStub::Instance().RefreshFromRuntime();
	return kAclSuccess;
}

void rt_sk_start_capture(void) {
	RtsObjectStub::Instance().StartCapture();
}

void rt_sk_stop_capture(void) {
	RtsObjectStub::Instance().StopCapture();
}

void rt_sk_capture_snapshot(void) {
	RtsObjectStub::Instance().StartCapture();
	RtsObjectStub::Instance().StopCapture();
}

}  // extern "C"