/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "api_perf_register/api_perf_factory.h"
#include "api_perf_register/utils/api_perf_utils.h"
#include "common/platform_context.h"
#include "codegen_api_param/codegen_api_param.h"
#include "v35/att/api_perf_register/ascendc_api_perf/reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
#include "v35/att/api_perf_register/perf_param_v2.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph_construct_utils.h"
#include "../../../../../ut/att/testcase/gen_model_info/api_perf_register/runtime_stub.h"

using namespace att;
using namespace af::sym;

namespace {
class UTestReduceMinMaxApiPerfV2 : public ::testing::Test {
 public:
  static ge::RuntimeStubV2Common stub_v_2;

  static void SetUpTestCase()
  {
    ge::RuntimeStub::Install(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
  }

  static void TearDownTestCase()
  {
    ge::RuntimeStub::UnInstall(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
  }

  void SetUp() override
  {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
  }
};

ge::RuntimeStubV2Common UTestReduceMinMaxApiPerfV2::stub_v_2;

NodeDetail MakeNodeDetail(const std::string &dtype, const std::vector<Expr> &dims)
{
  NodeDetail node_detail;
  node_detail.name = "ReduceMinMaxNode";
  node_detail.optype = "Reduce";
  node_detail.input_dtype = {dtype};
  node_detail.output_dtype = {dtype};
  node_detail.input_dims = dims;
  node_detail.output_dims = {dims.front()};
  return node_detail;
}

TensorShapeInfo MakeTensorShape(const std::string &dtype, uint32_t type_size, const std::vector<Expr> &dims,
                                const std::vector<Expr> &strides)
{
  TensorShapeInfo shape;
  shape.data_type = dtype;
  shape.data_type_size = type_size;
  shape.dims = dims;
  shape.repeats = dims;
  shape.strides = strides;
  shape.gm_strides = strides;
  return shape;
}

TensorShapeInfo MakeTensorShape(const std::string &dtype, uint32_t type_size, const std::vector<Expr> &dims,
                                const std::vector<Expr> &repeats, const std::vector<Expr> &strides)
{
  TensorShapeInfo shape = MakeTensorShape(dtype, type_size, dims, strides);
  shape.repeats = repeats;
  shape.origin_repeats = repeats;
  return shape;
}

std::string PipeString(const PerfOutputInfo &perf, PipeType pipe_type)
{
  const auto iter = perf.pipe_res.find(pipe_type);
  if (iter == perf.pipe_res.end()) {
    return "";
  }
  auto pipe_expr = iter->second.Replace(ConcursiveReplaceVars(perf.ternary_ops));
  pipe_expr.Simplify();
  return Str(pipe_expr);
}

std::string TernaryChoiceString(const PerfOutputInfo &perf, bool choice_a)
{
  if (perf.ternary_ops.empty()) {
    return "";
  }
  auto if_case = perf.ternary_ops.begin()->second.DeepCopyIfCase();
  auto choice = choice_a ? if_case->GetChoiceA() : if_case->GetChoiceB();
  auto choice_expr = choice->GetExpr().Replace(ConcursiveReplaceVars(perf.ternary_ops));
  choice_expr.Simplify();
  return Str(choice_expr);
}

Expr ResolvedPipeExpr(const PerfOutputInfo &perf, PipeType pipe_type)
{
  const auto iter = perf.pipe_res.find(pipe_type);
  if (iter == perf.pipe_res.end()) {
    return CreateExpr(0);
  }
  auto pipe_expr = iter->second.Replace(ConcursiveReplaceVars(perf.ternary_ops));
  pipe_expr.Simplify();
  return pipe_expr;
}

const PerfBreakdownItem *FindBreakdownItem(const PerfOutputInfo &perf, const std::string &name)
{
  for (const auto &group : perf.perf_breakdowns) {
    for (const auto &item : group.items) {
      if (item.name == name) {
        return &item;
      }
    }
  }
  return nullptr;
}

ascendcapi_v2::ReduceApiPerfContext MakeRaContext(const std::string &dtype, const std::vector<Expr> &dims,
                                                  bool is_reuse_source = true)
{
  ascendcapi_v2::ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(dtype, dims);
  context.pattern = ascendcapi_v2::ReducePattern::kRA;
  context.is_reuse_source = is_reuse_source;
  context.merge_mode = ascendcapi_v2::ReduceMergeMode::kNone;
  return context;
}

ascendcapi_v2::ReduceApiPerfContext MakeArContext(const std::string &dtype, const std::vector<Expr> &dims,
                                                  bool is_reuse_source = true)
{
  ascendcapi_v2::ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(dtype, dims);
  context.pattern = ascendcapi_v2::ReducePattern::kAR;
  context.is_reuse_source = is_reuse_source;
  context.merge_mode = ascendcapi_v2::ReduceMergeMode::kNone;
  return context;
}

void SetReduceSpecificParams(NodeInfo &node, codegen::ReducePattern pattern, codegen::ReduceMergeMode merge_mode,
                             const Expr &merge_size, const Expr &merge_times, bool is_reuse_source)
{
  node.reduce_specific_params.valid = true;
  node.reduce_specific_params.pattern = pattern;
  node.reduce_specific_params.merge_mode = merge_mode;
  node.reduce_specific_params.merge_size = merge_size;
  node.reduce_specific_params.merge_times = merge_times;
  node.reduce_specific_params.reuse = {true, is_reuse_source};
}
}  // namespace

// --- RegBase VF lookup tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, RegBaseReduceMinMaxUseVfTable)
{
  PerfOutputInfo min_perf;
  PerfOutputInfo max_perf;
  NodeDetail node_detail = MakeNodeDetail(kFloat16, {CreateExpr(128)});

  EXPECT_EQ(ascendcperf_v2::ReduceMinPerf(node_detail, min_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcperf_v2::ReduceMaxPerf(node_detail, max_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(max_perf, PipeType::AIV_VEC).empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RegBaseReduceMinMaxCoverIntegerDtypes)
{
  const std::vector<std::string> dtypes = {kInt64, kUInt64, kUInt16, kUInt32};
  for (const auto &dtype : dtypes) {
    PerfOutputInfo min_perf;
    PerfOutputInfo max_perf;
    NodeDetail node_detail = MakeNodeDetail(dtype, {CreateExpr(128)});
    EXPECT_EQ(ascendcperf_v2::ReduceMinPerf(node_detail, min_perf), ge::SUCCESS) << dtype;
    EXPECT_EQ(ascendcperf_v2::ReduceMaxPerf(node_detail, max_perf), ge::SUCCESS) << dtype;
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceMergedShapeUsesZeroStrideWhenLegacyPathHasNoNonZeroStride)
{
  const auto plan = codegen::BuildReduceMergedAxisPlan({false, true, false}, {true, false, false});
  EXPECT_TRUE(plan.valid);
  EXPECT_TRUE(plan.use_last_non_zero_stride);
  EXPECT_TRUE(plan.use_zero_stride);

  const auto shape = codegen::BuildReduceMergedShape({CreateExpr(2), CreateExpr(3), CreateExpr(4)},
                                                     {CreateExpr(1), CreateExpr(0), CreateExpr(1)},
                                                     {CreateExpr(0), CreateExpr(1), CreateExpr(1)}, 2U);
  EXPECT_TRUE(shape.valid);
  EXPECT_EQ(Str(shape.first), "2");
  EXPECT_EQ(Str(shape.last), "0");
}

// --- AscendC API layer tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceMinMaxArRaBranches)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext ar_context;
  ar_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(64)});
  ar_context.pattern = ReducePattern::kAR;
  ar_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext ra_context = ar_context;
  ra_context.pattern = ReducePattern::kRA;

  PerfOutputInfo ar_min_perf;
  PerfOutputInfo ar_max_perf;
  PerfOutputInfo ra_min_perf;
  PerfOutputInfo ra_max_perf;

  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(ar_context, ar_min_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(ar_context, ar_max_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(ra_context, ra_min_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(ra_context, ra_max_perf), ge::SUCCESS);

  EXPECT_FALSE(PipeString(ar_min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(ar_max_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(ra_min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(ra_max_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(ar_min_perf.ternary_ops.empty());
  EXPECT_FALSE(ar_max_perf.ternary_ops.empty());
  EXPECT_FALSE(ra_min_perf.ternary_ops.empty());
  EXPECT_FALSE(ra_max_perf.ternary_ops.empty());
  EXPECT_NE(PipeString(ar_max_perf, PipeType::AIV_VEC), PipeString(ra_max_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReducePerfProvidesReadableBreakdownItems)
{
  auto context = MakeRaContext(kInt64, {CreateExpr(16), CreateExpr(4)}, true);
  context.merge_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
  context.merge_size = CreateExpr(4);
  context.merge_times = CreateExpr(2);

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);

  const auto *body_item = FindBreakdownItem(perf, "reduce_body_perf");
  const auto *merge_item = FindBreakdownItem(perf, "reduce_merge_perf");
  const auto *total_item = FindBreakdownItem(perf, "reduce_total_perf");
  ASSERT_NE(body_item, nullptr);
  ASSERT_NE(merge_item, nullptr);
  ASSERT_NE(total_item, nullptr);
  EXPECT_NE(body_item->desc.find("single Reduce API body"), std::string::npos);
  EXPECT_NE(merge_item->desc.find("multi-reduce merge"), std::string::npos);
  EXPECT_NE(total_item->desc.find("Reduce API total"), std::string::npos);

  Expr total_expr = total_item->expr.Replace(ConcursiveReplaceVars(perf.ternary_ops));
  total_expr.Simplify();
  EXPECT_EQ(Str(total_expr), PipeString(perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceMaxB64AndMergeBranches)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext b64_context;
  b64_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(65)});
  b64_context.pattern = ReducePattern::kRA;
  b64_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo b64_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(b64_context, b64_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(b64_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(b64_perf.ternary_ops.empty());

  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(65)});
  context.pattern = ReducePattern::kRA;
  context.merge_size = CreateExpr(128);

  PerfOutputInfo copy_perf;
  context.merge_mode = ReduceMergeMode::kCopy;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, copy_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(copy_perf, PipeType::AIV_VEC).empty());

  PerfOutputInfo merge_perf;
  context.merge_mode = ReduceMergeMode::kMergeByElementwise;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, merge_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(merge_perf, PipeType::AIV_VEC).empty());
  EXPECT_NE(PipeString(copy_perf, PipeType::AIV_VEC), PipeString(merge_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceReuseSourceChangesFormula)
{
  auto reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, true);
  auto non_reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);
  EXPECT_NE(PipeString(reuse_perf, PipeType::AIV_VEC), PipeString(non_reuse_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceArAlignedNonReuseGreaterThanReuse)
{
  auto reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, true);
  auto non_reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);

  Expr reuse_cost = ResolvedPipeExpr(reuse_perf, PipeType::AIV_VEC);
  Expr non_reuse_cost = ResolvedPipeExpr(non_reuse_perf, PipeType::AIV_VEC);
  EXPECT_FALSE(PipeString(non_reuse_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(non_reuse_perf.ternary_ops.empty());
  EXPECT_NE(Str(reuse_cost), Str(non_reuse_cost));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceNonReuseCopyFallsBackForB64Dtype)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext reuse_context;
  reuse_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(64)});
  reuse_context.pattern = ReducePattern::kRA;
  reuse_context.is_reuse_source = true;
  reuse_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext non_reuse_context = reuse_context;
  non_reuse_context.is_reuse_source = false;

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);
  EXPECT_NE(PipeString(reuse_perf, PipeType::AIV_VEC), PipeString(non_reuse_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceMaxB64AlignedRaBranch)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // int64 + last=64: align_key = 64*8 % 32 = 0 → aligned branch, b64=true
  ReduceApiPerfContext aligned_b64_context;
  aligned_b64_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(64)});
  aligned_b64_context.pattern = ReducePattern::kRA;
  aligned_b64_context.is_reuse_source = true;
  aligned_b64_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo aligned_b64_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_b64_context, aligned_b64_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(aligned_b64_perf, PipeType::AIV_VEC).empty());
  // aligned branch: ternary_ops should NOT contain align case for b64 (last*8 % 32 == 0)
  EXPECT_FALSE(aligned_b64_perf.ternary_ops.empty());

  // Same dims but unaligned (last=65) should produce a different formula
  ReduceApiPerfContext unaligned_b64_context;
  unaligned_b64_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(65)});
  unaligned_b64_context.pattern = ReducePattern::kRA;
  unaligned_b64_context.is_reuse_source = true;
  unaligned_b64_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo unaligned_b64_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_b64_context, unaligned_b64_perf), ge::SUCCESS);
  EXPECT_NE(PipeString(aligned_b64_perf, PipeType::AIV_VEC), PipeString(unaligned_b64_perf, PipeType::AIV_VEC));
}

// --- RA Tree Reduction modeling tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64UnalignedTreeReduction)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // int64, first=1224, last=9 → large first triggers multi-round tree reduction
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(1224), CreateExpr(9)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());

  // Larger first should produce different (higher) formula than small first
  ReduceApiPerfContext small_context;
  small_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(9)});
  small_context.pattern = ReducePattern::kRA;
  small_context.is_reuse_source = true;
  small_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), ge::SUCCESS);
  EXPECT_NE(PipeString(perf, PipeType::AIV_VEC), PipeString(small_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalAlignedTreeReduction)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // float16, first=8, last=64 → aligned Concat subpath (dimA <= vlSize / 2), mainR=8, tailR=0
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(64)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalUnalignedTreeReduction)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // float16, first=8, last=65 → unaligned (65*2 % 32 ≠ 0), tree reduction applies
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(65)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalLargeFirstMultiRound)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // float16, first=256, last=128 → mainR=256, tailR=0, folds=8, avgFolds=4, mainTimes=2, foldZero=0
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(256), CreateExpr(128)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());

  // Multi-round should produce more cost than single-round
  ReduceApiPerfContext small_context;
  small_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(16), CreateExpr(128)});
  small_context.pattern = ReducePattern::kRA;
  small_context.is_reuse_source = true;
  small_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), ge::SUCCESS);
  // Multi-round should produce a different formula than single-round
  EXPECT_NE(PipeString(perf, PipeType::AIV_VEC), PipeString(small_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64AlignedTreeReduction)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // int64, first=8, last=64 → aligned (64*8%32==0), mainR=8, tailR=0, folds=3, avgFolds=3, mainTimes=1, foldZero=1
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(64)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64SymbolicDimRIncludesTreeReduceCost)
{
  const Expr z0t_size = CreateExpr("z0t_size");
  auto context = MakeRaContext(kInt64, {CreateExpr(136) * z0t_size, CreateExpr(12)});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);

  const std::string perf_expr = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_NE(perf_expr.find("z0t_size"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalSymbolicDimRIncludesTreeReduceCost)
{
  const Expr z0t_size = CreateExpr("z0t_size");
  auto concat_context = MakeRaContext(kFloat16, {CreateExpr(136) * z0t_size, CreateExpr(64)});
  auto over_vl_context = MakeRaContext(kFloat16, {CreateExpr(136) * z0t_size, CreateExpr(144)});
  auto unaligned_context = MakeRaContext(kFloat16, {CreateExpr(136) * z0t_size, CreateExpr(65)});

  PerfOutputInfo concat_perf;
  PerfOutputInfo over_vl_perf;
  PerfOutputInfo unaligned_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(concat_context, concat_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(over_vl_context, over_vl_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_context, unaligned_perf), ge::SUCCESS);

  EXPECT_NE(TernaryChoiceString(concat_perf, true).find("z0t_size"), std::string::npos);
  EXPECT_NE(TernaryChoiceString(over_vl_perf, true).find("z0t_size"), std::string::npos);
  EXPECT_NE(TernaryChoiceString(unaligned_perf, false).find("z0t_size"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, SymbolicDimRUsesFiniteTreeCases)
{
  const Expr z0t_size = CreateExpr("z0t_size");
  auto ra_context = MakeRaContext(kInt64, {CreateExpr(136) * z0t_size, CreateExpr(12)});
  auto ar_context = MakeArContext(kFloat16, {CreateExpr(16), z0t_size});

  PerfOutputInfo ra_perf;
  PerfOutputInfo ar_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(ra_context, ra_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(ar_context, ar_perf), ge::SUCCESS);

  const std::string ra_formula = PipeString(ra_perf, PipeType::AIV_VEC);
  const std::string ar_formula = PipeString(ar_perf, PipeType::AIV_VEC);
  EXPECT_NE(ra_formula.find("< 32768"), std::string::npos) << ra_formula;
  EXPECT_NE(ar_formula.find("z0t_size <= 128"), std::string::npos) << ar_formula;
  EXPECT_NE(ar_formula.find("< 32768"), std::string::npos) << ar_formula;
  EXPECT_EQ(ra_formula.find("Ceiling(((136 * z0t_size) * Rational(1 , 8)))"), std::string::npos) << ra_formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, SymbolicDimRNonReuseExactPowerUsesSplitBranch)
{
  const Expr z0t_size = CreateExpr("z0t_size");
  auto reuse_ra_context = MakeRaContext(kFloat16, {z0t_size, CreateExpr(144)}, true);
  auto non_reuse_ra_context = MakeRaContext(kFloat16, {z0t_size, CreateExpr(144)}, false);
  auto reuse_ar_context = MakeArContext(kFloat16, {CreateExpr(16), z0t_size}, true);
  auto non_reuse_ar_context = MakeArContext(kFloat16, {CreateExpr(16), z0t_size}, false);

  PerfOutputInfo reuse_ra_perf;
  PerfOutputInfo non_reuse_ra_perf;
  PerfOutputInfo reuse_ar_perf;
  PerfOutputInfo non_reuse_ar_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_ra_context, reuse_ra_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_ra_context, non_reuse_ra_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_ar_context, reuse_ar_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_ar_context, non_reuse_ar_perf), ge::SUCCESS);

  const std::string reuse_ra_formula = PipeString(reuse_ra_perf, PipeType::AIV_VEC);
  const std::string non_reuse_ra_formula = PipeString(non_reuse_ra_perf, PipeType::AIV_VEC);
  const std::string reuse_ar_formula = PipeString(reuse_ar_perf, PipeType::AIV_VEC);
  const std::string non_reuse_ar_formula = PipeString(non_reuse_ar_perf, PipeType::AIV_VEC);
  EXPECT_EQ(reuse_ra_formula.find("IsEqual(z0t_size, 16)"), std::string::npos) << reuse_ra_formula;
  EXPECT_NE(non_reuse_ra_formula.find("IsEqual(z0t_size, 16)"), std::string::npos) << non_reuse_ra_formula;
  EXPECT_EQ(reuse_ar_formula.find("IsEqual(z0t_size, 256)"), std::string::npos) << reuse_ar_formula;
  EXPECT_NE(non_reuse_ar_formula.find("IsEqual(z0t_size, 256)"), std::string::npos) << non_reuse_ar_formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, Bfloat16NormalReduceUsesHalfVectorRepeat)
{
  const Expr z1t_size = CreateExpr("z1t_size");
  auto context = MakeRaContext(kBfloat16, {CreateExpr(136), z1t_size});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);

  const std::string aligned_formula = TernaryChoiceString(perf, true);
  EXPECT_NE(aligned_formula.find("Rational(1 , 128)"), std::string::npos) << aligned_formula;
  EXPECT_EQ(aligned_formula.find("Rational(1 , 64)"), std::string::npos) << aligned_formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64SmallLastUsesOneVectorBlock)
{
  auto aligned_small_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(12)});
  auto aligned_one_block_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(64)});
  auto aligned_two_blocks_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(128)});
  auto unaligned_small_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)});
  auto unaligned_one_block_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(63)});
  auto unaligned_two_blocks_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(65)});

  PerfOutputInfo aligned_small_perf;
  PerfOutputInfo aligned_one_block_perf;
  PerfOutputInfo aligned_two_blocks_perf;
  PerfOutputInfo unaligned_small_perf;
  PerfOutputInfo unaligned_one_block_perf;
  PerfOutputInfo unaligned_two_blocks_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_small_context, aligned_small_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_one_block_context, aligned_one_block_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_two_blocks_context, aligned_two_blocks_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_small_context, unaligned_small_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_one_block_context, unaligned_one_block_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_two_blocks_context, unaligned_two_blocks_perf), ge::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(aligned_small_perf, true), TernaryChoiceString(aligned_one_block_perf, true));
  EXPECT_NE(TernaryChoiceString(aligned_small_perf, true), TernaryChoiceString(aligned_two_blocks_perf, true));
  EXPECT_EQ(TernaryChoiceString(unaligned_small_perf, false), TernaryChoiceString(unaligned_one_block_perf, false));
  EXPECT_NE(TernaryChoiceString(unaligned_small_perf, false), TernaryChoiceString(unaligned_two_blocks_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64ExpandsInt64BinaryFuncCost)
{
  auto aligned_context = MakeRaContext(kInt64, {CreateExpr(2176), CreateExpr(64)});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_context, perf), ge::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(perf, true), "99911");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArB64UnalignedSmallLastUsesOneVectorBlock)
{
  auto small_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(9)});
  auto one_block_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(63)});
  auto two_blocks_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)});

  PerfOutputInfo small_perf;
  PerfOutputInfo one_block_perf;
  PerfOutputInfo two_blocks_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(one_block_context, one_block_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(two_blocks_context, two_blocks_perf), ge::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(small_perf, false), TernaryChoiceString(one_block_perf, false));
  EXPECT_NE(TernaryChoiceString(small_perf, false), TernaryChoiceString(two_blocks_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArNormalUnalignedSmallLastUsesOneVectorBlock)
{
  auto small_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(65)});
  auto one_block_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(127)});
  auto two_blocks_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(129)});

  PerfOutputInfo small_perf;
  PerfOutputInfo one_block_perf;
  PerfOutputInfo two_blocks_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(small_context, small_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(one_block_context, one_block_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(two_blocks_context, two_blocks_perf), ge::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(small_perf, false), TernaryChoiceString(one_block_perf, false));
  EXPECT_NE(TernaryChoiceString(small_perf, false), TernaryChoiceString(two_blocks_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArUnalignedNonReuseDoesNotAddCopyCost)
{
  auto reuse_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)}, true);
  auto non_reuse_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArUnalignedDoesNotUseGlobalPenalty)
{
  auto b64_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)});
  auto normal_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(129)});

  PerfOutputInfo b64_perf;
  PerfOutputInfo normal_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(b64_context, b64_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(normal_context, normal_perf), ge::SUCCESS);

  const std::string b64_unaligned = TernaryChoiceString(b64_perf, false);
  const std::string normal_unaligned = TernaryChoiceString(normal_perf, false);
  EXPECT_EQ(b64_unaligned.find("2.5"), std::string::npos) << b64_unaligned;
  EXPECT_EQ(normal_unaligned.find("2.5"), std::string::npos) << normal_unaligned;
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64SymbolicAlignedLastUsesVectorBlockCeil)
{
  const Expr z0t_size = CreateExpr("z0t_size");
  const Expr z5t_size = CreateExpr("z5t_size");
  auto aligned_last = CreateExpr(4) * af::sym::Ceiling(z5t_size / CreateExpr(4));
  auto context = MakeRaContext(kInt64, {CreateExpr(136) * z0t_size, aligned_last});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), ge::SUCCESS);
  const std::string aligned_formula = TernaryChoiceString(perf, true);
  EXPECT_NE(aligned_formula.find("z5t_size"), std::string::npos);
  EXPECT_NE(aligned_formula.find("Rational(1 , 16)"), std::string::npos);
  EXPECT_EQ(aligned_formula.find("Rational(1 , 8)"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64UnalignedNonReuseMatchesReuseSource)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext reuse_context;
  reuse_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(9)});
  reuse_context.pattern = ReducePattern::kRA;
  reuse_context.is_reuse_source = true;
  reuse_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext non_reuse_context = reuse_context;
  non_reuse_context.is_reuse_source = false;

  PerfOutputInfo reuse_perf, non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(reuse_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(non_reuse_perf, PipeType::AIV_VEC).empty());

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaTreeReductionCostIncreasesWithFirst)
{
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // first=16 > first=8, same last=64, should produce higher cost
  ReduceApiPerfContext big_context;
  big_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(16), CreateExpr(64)});
  big_context.pattern = ReducePattern::kRA;
  big_context.is_reuse_source = true;
  big_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext small_context;
  small_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(64)});
  small_context.pattern = ReducePattern::kRA;
  small_context.is_reuse_source = true;
  small_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo big_perf, small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(big_context, big_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), ge::SUCCESS);

  // Different first dimensions should produce different formulas
  EXPECT_NE(PipeString(big_perf, PipeType::AIV_VEC), PipeString(small_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaPipeBarrierVPerfIsZeroInFirstPhase)
{
  auto base_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(64)});
  auto merge_context = base_context;
  merge_context.merge_mode = ascendcapi_v2::ReduceMergeMode::kMergeByElementwise;
  merge_context.merge_size = CreateExpr(128);

  PerfOutputInfo base_perf;
  PerfOutputInfo merge_perf;
  PerfOutputInfo elementwise_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(base_context, base_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(merge_context, merge_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcperf_v2::MaxPerf(MakeNodeDetail(kFloat16, {CreateExpr(128)}), elementwise_perf), ge::SUCCESS);

  Expr merge_delta = ResolvedPipeExpr(merge_perf, PipeType::AIV_VEC) - ResolvedPipeExpr(base_perf, PipeType::AIV_VEC);
  merge_delta.Simplify();
  EXPECT_EQ(Str(merge_delta), Str(ResolvedPipeExpr(elementwise_perf, PipeType::AIV_VEC)));
}

TEST_F(UTestReduceMinMaxApiPerfV2, MultiReduceCopyMergeUsesAlignedTempSizeAndElementwiseTail)
{
  auto base_context = MakeRaContext(kInt64, {CreateExpr(1224), CreateExpr(12)});
  auto merge_context = base_context;
  merge_context.merge_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
  merge_context.merge_size = CreateExpr(9);
  merge_context.merge_times = CreateExpr(9);

  PerfOutputInfo base_perf;
  PerfOutputInfo merge_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(base_context, base_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(merge_context, merge_perf), ge::SUCCESS);

  Expr ub_copy_res = CreateExpr(0);
  EXPECT_EQ(GetPerf({kUb2ub, kInt64, kInt64, {CreateExpr(32)}, CreateExpr(0)}, ub_copy_res), ge::SUCCESS);

  PerfOutputInfo elementwise_perf;
  EXPECT_EQ(ascendcperf_v2::MaxPerf(MakeNodeDetail(kInt64, {CreateExpr(32)}), elementwise_perf), ge::SUCCESS);

  Expr merge_delta = ResolvedPipeExpr(merge_perf, PipeType::AIV_VEC) - ResolvedPipeExpr(base_perf, PipeType::AIV_VEC);
  merge_delta.Simplify();
  Expr expected = (ub_copy_res + CreateExpr(8) * ResolvedPipeExpr(elementwise_perf, PipeType::AIV_VEC)) /
                  CreateExpr(9);
  expected.Simplify();
  EXPECT_EQ(Str(merge_delta), Str(expected));
}

TEST_F(UTestReduceMinMaxApiPerfV2, MultiReduceCopyMergeSymbolicTailUsesVectorBlock)
{
  const Expr z5t_size = CreateExpr("z5t_size");
  auto base_context = MakeRaContext(kInt64, {CreateExpr(2176), CreateExpr(4) * af::sym::Ceiling(z5t_size / kSymFour)});
  auto merge_context = base_context;
  merge_context.merge_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
  merge_context.merge_size = z5t_size;
  merge_context.merge_times = CreateExpr("z0z1z5Tb_size");

  PerfOutputInfo base_perf;
  PerfOutputInfo merge_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(base_context, base_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(merge_context, merge_perf), ge::SUCCESS);

  Expr merge_delta = ResolvedPipeExpr(merge_perf, PipeType::AIV_VEC) - ResolvedPipeExpr(base_perf, PipeType::AIV_VEC);
  merge_delta.Simplify();
  const std::string merge_formula = Str(merge_delta);
  EXPECT_NE(merge_formula.find("Rational(1 , 32)"), std::string::npos);
  EXPECT_EQ(merge_formula.find("Rational(1 , 4)"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64UnalignedNonReuseDoesNotAddCopyCost)
{
  auto reuse_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)}, true);
  auto non_reuse_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalUnalignedNonReuseDoesNotAddCopyCost)
{
  auto reuse_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(65)}, true);
  auto non_reuse_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(65)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalAlignedSubpathsProduceDistinctFormulas)
{
  auto concat_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(32)});
  auto less_than_vl_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(96)});
  auto over_vl_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(144)});

  PerfOutputInfo concat_perf;
  PerfOutputInfo less_than_vl_perf;
  PerfOutputInfo over_vl_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(concat_context, concat_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(less_than_vl_context, less_than_vl_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(over_vl_context, over_vl_perf), ge::SUCCESS);

  EXPECT_NE(TernaryChoiceString(concat_perf, true), TernaryChoiceString(less_than_vl_perf, true));
  EXPECT_NE(TernaryChoiceString(less_than_vl_perf, true), TernaryChoiceString(over_vl_perf, true));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64LargeUnalignedDoesNotCollapseToSmallFirstFormula)
{
  auto large_context = MakeRaContext(kInt64, {CreateExpr(1224), CreateExpr(9)});
  auto small_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)});

  PerfOutputInfo large_perf;
  PerfOutputInfo small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(large_context, large_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), ge::SUCCESS);

  EXPECT_NE(TernaryChoiceString(large_perf, false), TernaryChoiceString(small_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaAlignedNonReuseChangesTreeStructure)
{
  auto reuse_context = MakeRaContext(kFloat16, {CreateExpr(16), CreateExpr(96)}, true);
  auto non_reuse_context = MakeRaContext(kFloat16, {CreateExpr(16), CreateExpr(96)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), ge::SUCCESS);

  EXPECT_NE(TernaryChoiceString(reuse_perf, true), TernaryChoiceString(non_reuse_perf, true));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRaReduceUsesCodegenMergedShapeForMultiAAxes)
{
  const Expr z1t_size = CreateExpr("z1t_size");
  const Expr z2t_size = CreateExpr("z2t_size");
  const std::vector<Expr> repeats = {z2t_size, CreateExpr(17), z1t_size, CreateExpr(16), CreateExpr(9)};
  const std::vector<Expr> input_strides = {CreateExpr(3264) * z1t_size, CreateExpr(192) * z1t_size,
                                           CreateExpr(192), CreateExpr(12), CreateExpr(1)};
  const std::vector<Expr> output_strides = {CreateExpr(0), CreateExpr(0), CreateExpr(192), CreateExpr(12),
                                            CreateExpr(1)};
  const std::vector<TensorShapeInfo> inputs = {MakeTensorShape(
      kInt64, 8U, {CreateExpr(272) * z1t_size * z2t_size, CreateExpr(12)}, repeats, input_strides)};
  const std::vector<TensorShapeInfo> outputs = {
      MakeTensorShape(kInt64, 8U, {CreateExpr(192) * z1t_size}, repeats, output_strides)};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kRA, codegen::ReduceMergeMode::kCopy,
                          CreateExpr(192) * z1t_size, CreateExpr("z0z2Tt_size"), true);
  // 该用例绕过 parser，直接注入 codegen 侧合并结果，覆盖 ATT 读取 merged_dims 的路径。
  reduce_node.reduce_specific_params.merged_dims = {true, CreateExpr(17) * z2t_size, CreateExpr(192) * z1t_size};
  PerfOutputInfo perf;
  auto max_v2 = ApiPerfFactory::Instance().Create(kMax + "V2");
  ASSERT_NE(max_v2, nullptr);
  EXPECT_EQ(max_v2->GetPerfFunc()(inputs, outputs, reduce_node, perf), ge::SUCCESS);

  const std::string formula = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_EQ(formula.find("272 * z1t_size * z2t_size"), std::string::npos) << formula;
  EXPECT_NE(formula.find("17 * z2t_size"), std::string::npos) << formula;
  EXPECT_EQ(formula.find("Rational(17 , 8) * z2t_size"), std::string::npos) << formula;
  EXPECT_NE(formula.find("Ceiling((6 * z1t_size))"), std::string::npos) << formula;
  EXPECT_NE(formula.find("Ceiling((3 * z1t_size))"), std::string::npos) << formula;
  EXPECT_EQ(formula.find("Ceiling((48 * z1t_size))"), std::string::npos) << formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirReduceRebuildsMergedDimsFromCurrentShape)
{
  const std::vector<TensorShapeInfo> tail_inputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(4), CreateExpr(16)}, {CreateExpr(4), CreateExpr(16)},
                      {CreateExpr(16), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> tail_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(16)}, {CreateExpr(4), CreateExpr(16)},
                      {CreateExpr(0), CreateExpr(1)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kRA, codegen::ReduceMergeMode::kNone,
                          CreateExpr(16), CreateExpr(1), true);
  reduce_node.reduce_specific_params.merged_dims = {true, CreateExpr(64), CreateExpr(16)};

  PerfOutputInfo perf;
  auto max_v2 = ApiPerfFactory::Instance().Create(kMax + "V2");
  ASSERT_NE(max_v2, nullptr);
  EXPECT_EQ(max_v2->GetPerfFunc()(tail_inputs, tail_outputs, reduce_node, perf), ge::SUCCESS);

  auto expected_context = MakeRaContext(kFloat16, {CreateExpr(4), CreateExpr(16)}, true);
  PerfOutputInfo expected_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(expected_context, expected_perf), ge::SUCCESS);
  EXPECT_EQ(PipeString(perf, PipeType::AIV_VEC), PipeString(expected_perf, PipeType::AIV_VEC));
}

// --- ASCIR registration tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRegistersReduceMinMaxOpsAndAliases)
{
  const std::vector<TensorShapeInfo> reduce_inputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> reduce_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kAR, codegen::ReduceMergeMode::kNone, CreateExpr(8),
                          CreateExpr(1), false);
  PerfOutputInfo reduce_min_perf;
  PerfOutputInfo reduce_max_perf;
  auto reduce_min_v2 = ApiPerfFactory::Instance().Create(kReduceMin + "V2");
  auto reduce_max_v2 = ApiPerfFactory::Instance().Create(kReduceMax + "V2");
  ASSERT_NE(reduce_min_v2, nullptr);
  ASSERT_NE(reduce_max_v2, nullptr);
  EXPECT_EQ(reduce_min_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, reduce_min_perf), ge::SUCCESS);
  EXPECT_EQ(reduce_max_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, reduce_max_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(reduce_min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(reduce_max_perf, PipeType::AIV_VEC).empty());

  PerfOutputInfo min_reduce_alias_perf;
  PerfOutputInfo max_reduce_alias_perf;
  auto min_v2 = ApiPerfFactory::Instance().Create(kMin + "V2");
  auto max_v2 = ApiPerfFactory::Instance().Create(kMax + "V2");
  ASSERT_NE(min_v2, nullptr);
  ASSERT_NE(max_v2, nullptr);
  EXPECT_EQ(min_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, min_reduce_alias_perf), ge::SUCCESS);
  EXPECT_EQ(max_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, max_reduce_alias_perf), ge::SUCCESS);
  EXPECT_EQ(PipeString(reduce_min_perf, PipeType::AIV_VEC), PipeString(min_reduce_alias_perf, PipeType::AIV_VEC));
  EXPECT_EQ(PipeString(reduce_max_perf, PipeType::AIV_VEC), PipeString(max_reduce_alias_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRegistersReduceAnyAllOps)
{
  const std::vector<TensorShapeInfo> logical_reduce_inputs = {
      MakeTensorShape(kFloat32, 4U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> logical_reduce_outputs = {
      MakeTensorShape(kFloat32, 4U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kAR, codegen::ReduceMergeMode::kNone, CreateExpr(8),
                          CreateExpr(1), false);
  PerfOutputInfo reduce_all_perf;
  PerfOutputInfo reduce_any_perf;
  auto all_v2 = ApiPerfFactory::Instance().Create(kReduceAll + "V2");
  auto any_v2 = ApiPerfFactory::Instance().Create(kReduceAny + "V2");
  ASSERT_NE(all_v2, nullptr);
  ASSERT_NE(any_v2, nullptr);
  EXPECT_EQ(all_v2->GetPerfFunc()(logical_reduce_inputs, logical_reduce_outputs, reduce_node, reduce_all_perf),
            ge::SUCCESS);
  EXPECT_EQ(any_v2->GetPerfFunc()(logical_reduce_inputs, logical_reduce_outputs, reduce_node, reduce_any_perf),
            ge::SUCCESS);
  EXPECT_FALSE(PipeString(reduce_all_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(reduce_any_perf, PipeType::AIV_VEC).empty());
  EXPECT_EQ(PipeString(reduce_all_perf, PipeType::AIV_VEC).find("reduce_ar_normal_align_case"), std::string::npos);
  EXPECT_EQ(PipeString(reduce_any_perf, PipeType::AIV_VEC).find("reduce_ar_normal_align_case"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRegistersElementwiseMinMaxOps)
{
  const std::vector<TensorShapeInfo> elementwise_inputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(128)}, {CreateExpr(1)}),
      MakeTensorShape(kFloat16, 2U, {CreateExpr(128)}, {CreateExpr(1)})};
  const std::vector<TensorShapeInfo> elementwise_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(128)}, {CreateExpr(1)})};

  NodeInfo node;
  PerfOutputInfo minimum_perf;
  PerfOutputInfo maximum_perf;
  auto minimum_v2 = ApiPerfFactory::Instance().Create(kMinimum + "V2");
  auto maximum_v2 = ApiPerfFactory::Instance().Create(kMaximum + "V2");
  ASSERT_NE(minimum_v2, nullptr);
  ASSERT_NE(maximum_v2, nullptr);
  EXPECT_EQ(minimum_v2->GetPerfFunc()(elementwise_inputs, elementwise_outputs, node, minimum_perf), ge::SUCCESS);
  EXPECT_EQ(maximum_v2->GetPerfFunc()(elementwise_inputs, elementwise_outputs, node, maximum_perf), ge::SUCCESS);
  EXPECT_FALSE(PipeString(minimum_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(maximum_perf, PipeType::AIV_VEC).empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceAnyAllUseLogicalReduceEntrypoints)
{
  auto any_context = MakeRaContext(kUInt8, {CreateExpr(8), CreateExpr(64)});
  auto all_context = MakeArContext(kFloat32, {CreateExpr(8), CreateExpr(64)});

  PerfOutputInfo any_perf;
  PerfOutputInfo all_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceAnyPerf(any_context, any_perf), ge::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceAllPerf(all_context, all_perf), ge::SUCCESS);

  EXPECT_FALSE(PipeString(any_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(all_perf, PipeType::AIV_VEC).empty());
  EXPECT_EQ(PipeString(any_perf, PipeType::AIV_VEC).find("reduce_ra_normal_align_case"), std::string::npos);
  EXPECT_EQ(PipeString(all_perf, PipeType::AIV_VEC).find("reduce_ar_normal_align_case"), std::string::npos);

  auto unsupported_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(64)});
  PerfOutputInfo unsupported_perf;
  EXPECT_NE(ascendcapi_v2::ReduceAnyPerf(unsupported_context, unsupported_perf), ge::SUCCESS);
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirMaxMinAnyAllFallbackToElementwiseWhenReduceInfoUnknown)
{
  const std::vector<TensorShapeInfo> inputs = {
      MakeTensorShape(kFloat32, 2U, {CreateExpr(129)}, {CreateExpr(1)}),
      MakeTensorShape(kFloat32, 2U, {CreateExpr(129)}, {CreateExpr(1)})};
  const std::vector<TensorShapeInfo> outputs = {
      MakeTensorShape(kFloat32, 2U, {CreateExpr(129)}, {CreateExpr(1)})};
  const std::vector<std::string> tags = {kMax, kMin, kAny, kAll};

  NodeInfo node;
  for (const auto &tag : tags) {
    PerfOutputInfo perf;
    auto api_perf = ApiPerfFactory::Instance().Create(tag + "V2");
    ASSERT_NE(api_perf, nullptr) << tag;
    EXPECT_EQ(api_perf->GetPerfFunc()(inputs, outputs, node, perf), ge::SUCCESS) << tag;
    EXPECT_EQ(PipeString(perf, PipeType::AIV_VEC), "26") << tag;
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirReduceFailsWhenCodegenModeUnknown)
{
  const std::vector<TensorShapeInfo> reduce_inputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> reduce_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo node;
  PerfOutputInfo reduce_min_perf;
  auto min_v2 = ApiPerfFactory::Instance().Create(kReduceMin + "V2");
  ASSERT_NE(min_v2, nullptr);
  EXPECT_NE(min_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, node, reduce_min_perf), ge::SUCCESS);
}

// --- dtype registration tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, MinAscirUsesMinAttImpl)
{
  auto att_impl = af::ascir::AscirRegistry::GetInstance().GetIrAttImpl("3510", "Min");
  ASSERT_NE(att_impl, nullptr);
  EXPECT_STREQ(reinterpret_cast<const char *>(att_impl->GetApiPerf()), "MinV2");
}
