/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_gen_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <cstring>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "graph_construct_utils.h"
#include "result_checker_utils.h"
#include "gen_tiling_impl.h"
#include "graph/utils/graph_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "test_common_utils.h"
#include "runtime_stub.h"
#include "common/platform_context.h"
using namespace af::ascir_op;
namespace ascir {
using namespace ge;
}

std::string kRunTilingFuncMainTwoGroup = R"RAW(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

int main() {
  graph_two_group_testTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);

  // Group0: z0_0 = 2, z1 = 32
  tilingData.graph0_result0_g0_tiling_data.set_Z0_0(2);
  tilingData.graph0_result0_g0_tiling_data.set_Z1(256);

  // Group1: z0_1 = 128, z1 = 32
  tilingData.graph0_result0_g1_tiling_data.set_Z0_1(128);
  tilingData.graph0_result0_g1_tiling_data.set_Z1(256);

  std::cout << "=== Two Group Test with Different Axis Names ===" << std::endl;
  std::cout << "Group0: z0_0 = 2, z1 = 32" << std::endl;
  std::cout << "Group1: z0_1 = 128, z1 = 32" << std::endl;

  if (GetTiling(tilingData)) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)RAW";

using namespace att;
using att::test::CombineTilings;

namespace {
// 辅助函数：从单行文本中提取block_dim值
bool ExtractBlockDimFromLine(const std::string &line, uint32_t &block_dim) {
  const char *search_str = "The value of block_dim is ";
  size_t pos = line.find(search_str);
  if (pos == std::string::npos) {
    return false;
  }
  size_t value_start = pos + strlen(search_str);
  size_t value_end = line.find(" in", value_start);
  if (value_end == std::string::npos) {
    return false;
  }
  std::string value_str = line.substr(value_start, value_end - value_start);
  try {
    block_dim = std::stoul(value_str);
    return true;
  } catch (...) {
    return false;
  }
}

// 辅助函数：从日志中解析block_dim值
// 返回格式：{{first_g0, first_g1}, {second_g0, second_g1}}
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> ParseBlockDimFromLog(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return {};
  }

  std::vector<uint32_t> first_tiling_blocks;
  std::vector<uint32_t> second_tiling_blocks;
  std::string line;

  while (std::getline(file, line)) {
    uint32_t block_dim = 0;
    if (ExtractBlockDimFromLine(line, block_dim)) {
      if (first_tiling_blocks.size() < 2) {
        first_tiling_blocks.push_back(block_dim);
      } else {
        second_tiling_blocks.push_back(block_dim);
      }
    }
  }

  file.close();
  return {first_tiling_blocks, second_tiling_blocks};
}

// 辅助函数：验证二次tiling核数大于首次tiling且大于总核数的80%
bool VerifySecondaryTilingCoreUsage(const std::string &filename, uint32_t total_cores = 64) {
  auto [first_blocks, second_blocks] = ParseBlockDimFromLog(filename);

  if (first_blocks.size() < 2 || second_blocks.size() < 2) {
    std::cerr << "无法从日志中解析出足够的block_dim值，first.size()=" << first_blocks.size()
              << ", second.size()=" << second_blocks.size() << std::endl;
    return false;
  }

  uint32_t first_total = first_blocks[0] + first_blocks[1];
  uint32_t second_total = second_blocks[0] + second_blocks[1];
  uint32_t threshold = static_cast<uint32_t>(total_cores * 0.8);

  std::cout << "首次tiling核数: Group0=" << first_blocks[0] << ", Group1=" << first_blocks[1]
            << ", 总和=" << first_total << std::endl;
  std::cout << "二次tiling核数: Group0=" << second_blocks[0] << ", Group1=" << second_blocks[1]
            << ", 总和=" << second_total << std::endl;
  std::cout << "总核数阈值(80%): " << threshold << std::endl;

  bool condition1 = second_total > first_total;
  bool condition2 = second_total > threshold;

  if (!condition1) {
    std::cerr << "验证失败: 二次tiling核数(" << second_total << ") 不大于首次tiling核数(" << first_total << ")"
              << std::endl;
  }
  if (!condition2) {
    std::cerr << "验证失败: 二次tiling核数(" << second_total << ") 不大于总核数的80%(" << threshold << ")" << std::endl;
  }

  return condition1 && condition2;
}
}  // namespace

class STestGenConcatV2 : public ::testing::Test {
 public:
  static af::RuntimeStubV2 stub_v_2;
  static void TearDownTestCase() {
    ge::PlatformContext::GetInstance().Reset();
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    ge::PlatformContext::GetInstance().Reset();
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    ge::RuntimeStub::Install(&stub_v_2);
    AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    AutoFuseConfig::MutableAttStrategyConfig().force_template_op_name = "";
    AutoFuseConfig::MutableAttStrategyConfig().force_tiling_case = "";
    AutoFuseConfig::MutableAttStrategyConfig().force_schedule_result = -1L;
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    ge::RuntimeStub::UnInstall(&stub_v_2);
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    unsetenv("AUTOFUSE_DFX_FLAGS");
  }
};
af::RuntimeStubV2 STestGenConcatV2::stub_v_2;

namespace af{
namespace ascir {
namespace cg {
Status BuildVectorFunctionSubgraph(af::AscGraph &subgraph) {
  auto ND = af::Symbol("ND");
  auto nd = subgraph.CreateAxis("nd", ND);
  auto [ndB, ndb] = subgraph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = subgraph.TileSplit(ndb->id);
  auto data1 = subgraph.CreateContiguousData("input1", DT_FLOAT, {*ndbt});
  auto load1 = Load("load1", data1);
  auto abs1 = Abs("abs1", load1);
  auto sub1 = Sub("sub1", abs1, abs1);
  auto store1 = Store("store1", sub1);
  GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt}, {load1, sub1, store1}, 2));
  auto output1 = Output("output1", store1);
  return af::SUCCESS;
}

// 内联函数：添加VectorFunc子图（Lambda版本）
static auto AddVectorFuncSubgraphLambda = [](af::AscGraph &graph, const char_t *node_name,
                                             auto loop_axis_ptr) -> Status {
  af::AscGraph subgraph(node_name);
  GE_ASSERT_SUCCESS(BuildVectorFunctionSubgraph(subgraph));
  graph.AddSubGraph(subgraph);
  auto node = graph.FindNode(node_name);
  GE_ASSERT_NOTNULL(node);
  node->attr.sched.axis = {loop_axis_ptr->id};
  node->attr.sched.loop_axis = loop_axis_ptr->id;
  af::AttrUtils::SetStr(node->GetOpDescBarePtr(), "sub_graph_name", node_name);
  return af::SUCCESS;
};

// Concat测试：构建VectorFunc图（BlockSplit先于TileSplit）
static Status BuildVectorFuncGraphS0(af::AscGraph &graph) {
  auto S0 = af::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});

  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto vector_func = ascir_op::VectorFunc("vector_func");
      vector_func.SetAttr("sub_graph_name", "vector_func");
      vector_func.InstanceOutputy(1);
      vector_func.x = {load1};
      auto store1 = Store("store1", vector_func.y[0]);
      *store1.axis = {z0bT->id, z0bt->id};
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0b, *z0bt},
                                                                    {load1, vector_func.y[0], store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  GE_ASSERT_SUCCESS(AddVectorFuncSubgraphLambda(graph, "vector_func", z0bT));
  return af::SUCCESS;
}

// Concat测试：构建VectorFunc图（TileSplit先于BlockSplit）
static Status BuildVectorFuncGraphS0V1(af::AscGraph &graph) {
  auto S0 = af::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto vector_func = ascir_op::VectorFunc("vector_func");
      vector_func.SetAttr("sub_graph_name", "vector_func");
      vector_func.InstanceOutputy(1);
      vector_func.x = {load1};
      auto store1 = Store("store1", vector_func.y[0]);
      *store1.axis = {z0TB->id, z0Tb->id};
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0T, *z0TB, *z0t, *z0Tb},
                                                                    {load1, vector_func.y[0], store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  GE_ASSERT_SUCCESS(AddVectorFuncSubgraphLambda(graph, "vector_func", z0TB));
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0WithVectorFunc(af::AscGraph &graph) {
  return BuildVectorFuncGraphS0(graph);
}

Status BuildConcatGroupAscendGraphS0WithVectorFuncV1(af::AscGraph &graph) {
  return BuildVectorFuncGraphS0V1(graph);
}

// 辅助函数：设置GM属性
Status SetGmHardwareAttr(af::AscGraph &graph) {
  auto load_asc_node = graph.FindNode("load1");
  GE_ASSERT_NOTNULL(load_asc_node);
  load_asc_node->inputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;

  auto store_asc_node = graph.FindNode("store1");
  GE_ASSERT_NOTNULL(store_asc_node);
  store_asc_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;

  return af::SUCCESS;
}

// 两个Group测试用例的图构建通用函数
// z0_symbol_name: 大写的Symbol名称（如"Z0_0", "Z0_1"）
// z0_axis_name: 小写的axis名称（如"z0_0", "z0_1"）
Status BuildTwoGroupTestAscGraphCommon(af::AscGraph &graph, const std::string &z0_symbol_name,
                                       const std::string &z0_axis_name) {
  auto Z0 = af::Symbol(z0_symbol_name.c_str());
  auto Z1 = af::Symbol("Z1");
  auto z0_axis = graph.CreateAxis(z0_axis_name, Z0);
  auto z1 = graph.CreateAxis("z1", Z1);

  // 先z0和z1合轴成z0z1
  std::vector<int64_t> merge_ids = {z0_axis.id, z1.id};
  auto z0z1 = graph.MergeAxis(merge_ids);

  // 对z0z1做Tile切分
  auto [z0z1T, z0z1t] = graph.TileSplit(z0z1->id);

  // 对z0z1T做Block切分
  auto [z0z1TB, z0z1Tb] = graph.BlockSplit(z0z1T->id);

  std::vector<af::Axis> axes = {z0_axis, z1};
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, axes);

  LOOP(*z0z1TB) {
    LOOP(*z0z1Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto abs1 = Abs("abs1", load1).TQue(Position::kPositionVecOut, 1, 1);
      auto store1 = Store("store1", abs1);

      std::vector<af::Axis> output_axes = {*z0z1TB, *z0z1Tb, *z0z1t};
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(output_axes, {load1, abs1, store1}, 1));

      *load1.axis = {z0z1Tb->id, z0z1t->id};
      *load1.repeats = {z0z1Tb->size, z0z1t->size};
      *load1.strides = {z0z1t->size, CreateExpr(1)};
      *load1.vectorized_axis = {z0z1t->id};

      *abs1.axis = {z0z1Tb->id, z0z1t->id};
      *abs1.repeats = {z0z1Tb->size, z0z1t->size};
      *abs1.strides = {z0z1t->size, CreateExpr(1)};
      *abs1.vectorized_axis = {z0z1t->id};

      *store1.axis = {z0z1Tb->id, z0z1t->id};
      *store1.repeats = {z0z1Tb->size, z0z1t->size};
      *store1.strides = {z0z1t->size, CreateExpr(1)};
      *store1.vectorized_axis = {z0z1t->id};

      auto output1 = Output("output1", store1);
    }
  }

  return SetGmHardwareAttr(graph);
}

Status BuildTwoGroupTestAscGraph_Z0_0(af::AscGraph &graph) {
  return BuildTwoGroupTestAscGraphCommon(graph, "Z0_0", "z0_0");
}

Status BuildTwoGroupTestAscGraph_Z0_1(af::AscGraph &graph) {
  return BuildTwoGroupTestAscGraphCommon(graph, "Z0_1", "z0_1");
}
}  // namespace cg
}  // namespace ascir
}  // namespace ge
const std::string kFirstGraphName = "case0";
const std::string kSecondGraphName = "case1";

af::Status BuildVectorFuncFusedResult(bool tile_key, ascir::FusedScheduledResult &fused_scheduled_result,
                                      const std::string &graph_name) {
  ascir::ScheduleGroup schedule_group;
  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results;
  af::AscGraph graph_s0(graph_name.c_str());
  if (tile_key) {
    GE_ASSERT_SUCCESS(af::ascir::cg::BuildConcatGroupAscendGraphS0WithVectorFuncV1(graph_s0));
  } else {
    GE_ASSERT_SUCCESS(af::ascir::cg::BuildConcatGroupAscendGraphS0WithVectorFunc(graph_s0));
  }
  graph_s0.SetTilingKey(1U);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph_s0);
  schedule_group.impl_graphs.emplace_back(graph_s0);
  schedule_result.schedule_groups.emplace_back(schedule_group);
  schedule_results.emplace_back(schedule_result);
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  return af::SUCCESS;
}

af::Status GenTilingCodeFromResult(const ascir::FusedScheduledResult &fused_scheduled_result,
                                   const std::string &graph_name) {
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace(kTilingDataTypeName, graph_name + "TilingData");
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_EQ(generator.GenTilingCode("Concat", all_model_infos, generator_config, tiling_res), af::SUCCESS);
  std::ofstream oss;
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
  return af::SUCCESS;
}

af::Status GenTilingImplForGraphS0WithVectorFunc(bool tile_key = false) {
  const std::string graph_name = "graph_nd";
  ascir::FusedScheduledResult fused_scheduled_result;
  GE_ASSERT_SUCCESS(BuildVectorFuncFusedResult(tile_key, fused_scheduled_result, graph_name));

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3("Concat", fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n" << tiling_func;
  oss.close();
  GE_ASSERT_EQ(res, true);

  GE_ASSERT_SUCCESS(GenTilingCodeFromResult(fused_scheduled_result, graph_name));
  auto ret = std::system(
      std::string("cp ").append(TOP_DIR).append("/autofuse/tests/st/att/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  GE_ASSERT_EQ(ret, 0);
  return af::SUCCESS;
}

TEST_F(STestGenConcatV2, test_vector_function_parse) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(GenTilingImplForGraphS0WithVectorFunc(), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_z0bt_size(), 10);
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_ub_size(), 245760);
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(10);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret =
      std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
}

TEST_F(STestGenConcatV2, test_vector_function_parse_with_auto_tuning) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(GenTilingImplForGraphS0WithVectorFunc(true), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_z0t_size(), 1);
  MY_ASSERT_EQ(tilingData.get_block_dim(), 10);
  MY_ASSERT_EQ(tilingData.get_ub_size(), 245760);
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(10);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "concat tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret =
      std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 辅助函数：构建两Group测试的ScheduleResult
void BuildTwoGroupScheduleResult(ascir::FusedScheduledResult &fused_scheduled_result, const std::string &graph_name) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;

  af::AscGraph graph_0(graph_name.c_str());
  af::AscGraph graph_1("graph_1");

  ASSERT_EQ(af::ascir::cg::BuildTwoGroupTestAscGraph_Z0_0(graph_0), af::SUCCESS);
  ASSERT_EQ(af::ascir::cg::BuildTwoGroupTestAscGraph_Z0_1(graph_1), af::SUCCESS);

  graph_0.SetTilingKey(0U);
  std::vector<af::AscGraph> graphs_0 = {graph_0};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs_0);
  schedule_group1.impl_graphs.emplace_back(graph_0);

  graph_1.SetTilingKey(1U);
  std::vector graphs_1 = {graph_1};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs_1);
  schedule_group2.impl_graphs.emplace_back(graph_1);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result1.enable_group_parallel = true;
  schedule_results.emplace_back(schedule_result1);

  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
}

// 辅助函数：生成tiling函数和数据
void GenerateTwoGroupTilingFuncAndData(const ascir::FusedScheduledResult &fused_scheduled_result,
                                       const std::string &op_name, const std::string &graph_name) {
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");

  // 生成tiling函数
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  // 生成tiling数据
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), af::SUCCESS);

  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
}

// 辅助函数：准备测试环境
void PrepareTwoGroupTestEnv() {
  auto ret = std::system(
      std::string("cp ").append(TOP_DIR).append("/autofuse/tests/st/att/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::ReplaceLogMacrosGeneric("Concat_tiling_func.cpp"), true);

  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kRunTilingFuncMainTwoGroup;
  oss.close();
}

// 辅助函数：编译和运行测试
void CompileAndRunTwoGroupTest() {
  auto ret = std::system(
      "g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
}

// 辅助函数：验证测试输出
void VerifyTwoGroupTestOutput() {
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Two Group Test"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Test passed"), true);
  EXPECT_TRUE(VerifySecondaryTilingCoreUsage("./info.log", 64))
      << "二次tiling核数应该大于首次tiling且大于总核数(64)的80%";
}

// 测试用例：两个Group，每个Group有一个AscGraph
// - 每个AscGraph有两个轴：z0（动态）和z1（固定32）
// - 两个Group的z0轴名不同：Group0为z0_0，Group1为z0_1
// - 调度策略：先z1做Tile切分，然后z1T和z0合轴成z0z1T，再Block切分
// - 图结构：Load -> Abs -> Store
TEST_F(STestGenConcatV2, two_group_different_z0_axis_name) {
  const std::string kFirstGraphName = "graph_two_group_test";
  const std::string kOpName = "Concat";

  ascir::FusedScheduledResult fused_scheduled_result;
  BuildTwoGroupScheduleResult(fused_scheduled_result, kFirstGraphName);

  GenerateTwoGroupTilingFuncAndData(fused_scheduled_result, kOpName, kFirstGraphName);
  PrepareTwoGroupTestEnv();
  CompileAndRunTwoGroupTest();
  VerifyTwoGroupTestOutput();
}
