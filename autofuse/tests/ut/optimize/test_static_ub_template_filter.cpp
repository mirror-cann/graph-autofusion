/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <memory>

#include "ascir_ops.h"
#include "common/platform_context.h"
#include "optimize/buffer_allocate/buf_que_allocator.h"
#include "optimize/static_ub_template_filter.h"
#include "schedule_result.h"
#include "tests/depends/runtime/src/runtime_stub.h"
#include "tests/depends/slog/src/slog_stub.h"

namespace optimize {
namespace {

constexpr int64_t kDefaultUbSize = 1024;

af::AscGraph MakeQueueGraph(const std::string &name, const af::Expression &axis_size, ge::DataType dtype = ge::DT_UINT8,
                            int64_t buf_num = 1) {
  af::AscGraph graph(name.c_str());
  auto &axis = graph.CreateAxis("z0", axis_size);
  af::ascir_op::Data data_op("data", graph);
  af::ascir_op::Load load_op("load");
  graph.AddNode(load_op);
  load_op.x = data_op.y;
  auto node = graph.FindNode("load");
  if (node == nullptr) {
    ADD_FAILURE() << "failed to build load node";
    return graph;
  }
  auto &output_attr = node->outputs[0].attr;
  output_attr.dtype = dtype;
  output_attr.repeats = {axis.size};
  output_attr.vectorized_axis = {axis.id};
  output_attr.vectorized_strides = {af::sym::kSymbolOne};
  output_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  output_attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  output_attr.mem.position = af::Position::kPositionVecIn;
  output_attr.mem.reuse_id = 0;
  output_attr.que.id = 0;
  output_attr.que.buf_num = buf_num;
  return graph;
}

ascir::ScheduledResult MakeScheduledResult(af::AscGraph &&graph) {
  ascir::ScheduledResult result;
  ascir::ScheduleGroup group;
  group.impl_graphs.emplace_back(std::move(graph));
  result.schedule_groups.emplace_back(std::move(group));
  return result;
}

ascir::ScheduledResult MakeScheduledResultWithGroups(std::vector<ascir::ScheduleGroup> &&groups) {
  ascir::ScheduledResult result;
  result.schedule_groups = std::move(groups);
  return result;
}

af::AscGraph MakeUnallocatedLoadStoreGraph(const std::string &name, int64_t size) {
  af::AscGraph graph(name.c_str());
  const af::Expression s0 = graph.CreateSizeVar(size);
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data data_op("data", graph);
  data_op.ir_attr.SetIndex(0);
  data_op.y.dtype = ge::DT_UINT8;

  af::ascir_op::Load load_op("load");
  load_op.x = data_op.y;
  load_op.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load_op.attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load_op.y.dtype = ge::DT_UINT8;
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {s0};
  *load_op.y.strides = {af::ops::One};
  *load_op.y.vectorized_axis = {z0.id};
  *load_op.y.vectorized_strides = {af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.x = load_op.y;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store_op.y.dtype = ge::DT_UINT8;
  *store_op.y.axis = {z0.id};
  *store_op.y.repeats = {s0};
  *store_op.y.strides = {af::ops::One};
  *store_op.y.vectorized_axis = {z0.id};
  *store_op.y.vectorized_strides = {af::ops::One};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.ir_attr.SetIndex(0);
  return graph;
}

ascir::ScheduledResult MakeUnallocatedLoadStoreScheduledResult(const std::string &name, int64_t size) {
  return MakeScheduledResult(MakeUnallocatedLoadStoreGraph(name, size));
}

ascir::ScheduleGroup MakeScheduleGroup(std::vector<af::AscGraph> &&impl_graphs) {
  ascir::ScheduleGroup group;
  group.impl_graphs = std::move(impl_graphs);
  return group;
}

ascir::FusedScheduledResult MakeFusedResult(std::vector<ascir::ScheduledResult> &&results) {
  ascir::FusedScheduledResult fused_result;
  fused_result.fused_graph_name = "fused_ub_filter";
  fused_result.node_idx_to_scheduled_results.emplace_back(std::move(results));
  return fused_result;
}

void SetPlatformUb(int64_t ub_size, const std::string &soc_ver = "mock_soc") {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = soc_ver;
  platform_info.ub_size = ub_size;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
}

void SetUbOverride(int64_t ub_size) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformContext::GetInstance().SetUbSizeOverride(ub_size);
}

class RuntimeUbSizeFailStub : public ge::RuntimeStub {
 public:
  rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)val;
    (void)maxLen;
    return 1;
  }
};

class CaptureSlogStub : public ge::SlogStub {
 public:
  void Log(int module_id, int level, const char *fmt, va_list args) override {
    if (level < GetLevel()) {
      return;
    }
    char buffer[2048] = {};
    if (Format(buffer, sizeof(buffer), module_id, level, fmt, args) > 0) {
      logs_ += buffer;
      logs_ += '\n';
    }
  }

  const std::string &GetLogs() const {
    return logs_;
  }

 private:
  std::string logs_;
};

class StaticUbTemplateFilterTest : public testing::Test {
 protected:
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::Reset();
    ge::SlogStub::SetInstance(nullptr);
  }
};

TEST_F(StaticUbTemplateFilterTest, KeepsAllTemplatesWhenPlatformIsNotInitialized) {
  ge::PlatformContext::GetInstance().Reset();
  ge::RuntimeStub::SetInstance(std::make_shared<RuntimeUbSizeFailStub>());
  auto fused_result = MakeFusedResult({MakeScheduledResult(MakeQueueGraph("large", af::Symbol(4096)))});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
}

TEST_F(StaticUbTemplateFilterTest, KeepsAllTemplatesWhenUbSizeIsInvalid) {
  SetPlatformUb(0);
  ge::RuntimeStub::SetInstance(std::make_shared<RuntimeUbSizeFailStub>());
  auto fused_result = MakeFusedResult({MakeScheduledResult(MakeQueueGraph("large", af::Symbol(4096)))});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
}

TEST_F(StaticUbTemplateFilterTest, UsesRuntimeUbSizeWhenOnlyInjectedUbSizeIsInvalid) {
  SetUbOverride(0);
  auto kept = MakeScheduledResult(MakeQueueGraph("fit", af::Symbol(1), ge::DT_UINT8, 1));
  auto dropped = MakeScheduledResult(MakeQueueGraph("oversize", af::Symbol(300000), ge::DT_UINT8, 1));
  auto fused_result = MakeFusedResult({std::move(kept), std::move(dropped)});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  ASSERT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(), "fit");
}

TEST_F(StaticUbTemplateFilterTest, UsesUbSizeOverrideWithoutInitializingPlatformInfo) {
  SetUbOverride(kDefaultUbSize);
  auto kept = MakeScheduledResult(MakeQueueGraph("fit", af::Symbol(1), ge::DT_UINT8, 1));
  auto dropped = MakeScheduledResult(MakeQueueGraph("oversize", af::Symbol(2048), ge::DT_UINT8, 1));
  auto fused_result = MakeFusedResult({std::move(kept), std::move(dropped)});

  ge::PlatformInfo platform_info;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(platform_info));
  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  ASSERT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(), "fit");
}

TEST_F(StaticUbTemplateFilterTest, UsesUbSizeOverrideWhenInitializedPlatformUbSizeIsInvalid) {
  SetPlatformUb(0);
  ge::PlatformContext::GetInstance().SetUbSizeOverride(kDefaultUbSize);
  auto kept = MakeScheduledResult(MakeQueueGraph("fit", af::Symbol(1), ge::DT_UINT8, 1));
  auto dropped = MakeScheduledResult(MakeQueueGraph("oversize", af::Symbol(2048), ge::DT_UINT8, 1));
  auto fused_result = MakeFusedResult({std::move(kept), std::move(dropped)});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  ASSERT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(), "fit");
}

TEST_F(StaticUbTemplateFilterTest, UsesRuntimeUbSizeWhenPlatformInfoIsNotInitialized) {
  ge::PlatformContext::GetInstance().Reset();
  ge::RuntimeStub::Reset();
  auto kept = MakeScheduledResult(MakeQueueGraph("fit", af::Symbol(1), ge::DT_UINT8, 1));
  auto dropped = MakeScheduledResult(MakeQueueGraph("oversize", af::Symbol(300000), ge::DT_UINT8, 1));
  auto fused_result = MakeFusedResult({std::move(kept), std::move(dropped)});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  ASSERT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(), "fit");
}

TEST_F(StaticUbTemplateFilterTest, KeepsEmptyScheduledResults) {
  SetPlatformUb(kDefaultUbSize);
  auto fused_result = MakeFusedResult({});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_TRUE(fused_result.node_idx_to_scheduled_results[0].empty());
}

TEST_F(StaticUbTemplateFilterTest, DropsTemplateWhenMinimumUbUsageExceedsPlatformLimit) {
  SetPlatformUb(kDefaultUbSize);
  auto log_stub = std::make_shared<CaptureSlogStub>();
  log_stub->SetLevelInfo();
  ge::SlogStub::SetInstance(log_stub);
  auto kept = MakeScheduledResult(MakeQueueGraph("fit", af::Symbol(1), ge::DT_UINT8, 1));
  auto dropped = MakeScheduledResult(MakeQueueGraph("oversize", af::Symbol(2048), ge::DT_UINT8, 1));
  auto fused_result = MakeFusedResult({std::move(kept), std::move(dropped)});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  ASSERT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(), "fit");
  EXPECT_EQ(log_stub->GetLogs().find("keep template, graph=fit"), std::string::npos) << log_stub->GetLogs();
  EXPECT_EQ(log_stub->GetLogs().find("drop template, graph=oversize"), std::string::npos) << log_stub->GetLogs();
  EXPECT_NE(log_stub->GetLogs().find("total_dropped=1"), std::string::npos) << log_stub->GetLogs();
  EXPECT_NE(log_stub->GetLogs().find("platform_ub_size=1024"), std::string::npos) << log_stub->GetLogs();
  EXPECT_EQ(log_stub->GetLogs().find("reason=min_ub_usage exceeds platform_ub_size"), std::string::npos)
      << log_stub->GetLogs();
}

TEST_F(StaticUbTemplateFilterTest, HandlesNullFusedGraphNameInLogs) {
  SetPlatformUb(kDefaultUbSize);
  auto log_stub = std::make_shared<CaptureSlogStub>();
  log_stub->SetLevelInfo();
  ge::SlogStub::SetInstance(log_stub);
  ascir::FusedScheduledResult fused_result;
  fused_result.node_idx_to_scheduled_results.emplace_back();
  fused_result.node_idx_to_scheduled_results[0].emplace_back(
      MakeScheduledResult(MakeQueueGraph("fit", af::Symbol(1), ge::DT_UINT8, 1)));

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  EXPECT_NE(log_stub->GetLogs().find("graph=<null>"), std::string::npos) << log_stub->GetLogs();
}

TEST_F(StaticUbTemplateFilterTest, DropsStaticTemplateAfterPrepareImplGraphMemoryPlan) {
  ge::PlatformContext::GetInstance().Reset();
  auto kept = MakeUnallocatedLoadStoreScheduledResult("fit", 16);
  auto dropped = MakeUnallocatedLoadStoreScheduledResult("oversize", 300000);
  auto fused_result = MakeFusedResult({std::move(kept), std::move(dropped)});

  BufQueAllocator allocator;
  ASSERT_EQ(allocator.PrepareImplGraphMemoryPlan(fused_result), af::SUCCESS);
  ASSERT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);
  ASSERT_EQ(allocator.CollectFusedIoNodes(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(), "fit");
  EXPECT_FALSE(fused_result.input_nodes.empty());
  EXPECT_FALSE(fused_result.output_nodes.empty());
}

TEST_F(StaticUbTemplateFilterTest, CollectsIoNodesWhenFilterIsSkippedAfterPrepare) {
  ge::PlatformContext::GetInstance().Reset();
  auto fused_result = MakeFusedResult({MakeUnallocatedLoadStoreScheduledResult("fit", 16)});

  BufQueAllocator allocator;
  ASSERT_EQ(allocator.PrepareImplGraphMemoryPlan(fused_result), af::SUCCESS);
  ASSERT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);
  ASSERT_EQ(allocator.CollectFusedIoNodes(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.input_nodes.size(), 1UL);
  ASSERT_EQ(fused_result.output_nodes.size(), 1UL);
}

TEST_F(StaticUbTemplateFilterTest, CountsDroppedTemplatesAfterScheduledResultIsRemoved) {
  SetPlatformUb(kDefaultUbSize);
  auto log_stub = std::make_shared<CaptureSlogStub>();
  log_stub->SetLevelInfo();
  ge::SlogStub::SetInstance(log_stub);
  auto first_group = MakeScheduleGroup({MakeQueueGraph("oversize_group0", af::Symbol(2048), ge::DT_UINT8, 1)});
  auto second_group = MakeScheduleGroup({MakeQueueGraph("oversize_group1", af::Symbol(4096), ge::DT_UINT8, 1)});
  auto dropped = MakeScheduledResultWithGroups({std::move(first_group), std::move(second_group)});
  auto kept = MakeScheduledResult(MakeQueueGraph("fit", af::Symbol(1), ge::DT_UINT8, 1));
  auto fused_result = MakeFusedResult({std::move(dropped), std::move(kept)});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  EXPECT_NE(log_stub->GetLogs().find("total_dropped=2"), std::string::npos) << log_stub->GetLogs();
}

TEST_F(StaticUbTemplateFilterTest, KeepsTemplateWhenMinimumUbUsageEqualsPlatformLimit) {
  SetPlatformUb(1024);
  auto fused_result =
      MakeFusedResult({MakeScheduledResult(MakeQueueGraph("equal", af::Symbol(1024), ge::DT_UINT8, 1))});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
}

TEST_F(StaticUbTemplateFilterTest, ReplacesDynamicSizeVarsWithOneForMinimumUsage) {
  SetPlatformUb(32);
  af::AscGraph graph = MakeQueueGraph("dynamic_min", af::Symbol("s0"), ge::DT_UINT8, 1);
  (void)graph.CreateSizeVar("s0");
  auto fused_result = MakeFusedResult({MakeScheduledResult(std::move(graph))});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
}

TEST_F(StaticUbTemplateFilterTest, KeepsTemplateWhenExpressionHasUnknownSymbol) {
  SetPlatformUb(32);
  auto fused_result = MakeFusedResult({MakeScheduledResult(MakeQueueGraph("unknown", af::Symbol("unknown")))});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
}

TEST_F(StaticUbTemplateFilterTest, DoesNotReplaceUnrecognizedOriginVars) {
  SetPlatformUb(32);
  auto fused_result = MakeFusedResult({MakeScheduledResult(MakeQueueGraph("origin_unknown", af::Symbol("unknown")))});
  fused_result.origin_vars = {af::Symbol("unknown")};

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
}

TEST_F(StaticUbTemplateFilterTest, KeepsTemplateWhenExpressionCannotBeEvaluated) {
  SetPlatformUb(32);
  auto invalid_expr = af::sym::Div(af::Symbol(1), af::Symbol(0));
  auto fused_result = MakeFusedResult({MakeScheduledResult(MakeQueueGraph("div_zero", invalid_expr))});

  EXPECT_EQ(StaticUbTemplateFilter().Filter(fused_result), af::SUCCESS);

  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  EXPECT_EQ(fused_result.node_idx_to_scheduled_results[0].size(), 1U);
}

}  // namespace
}  // namespace optimize
