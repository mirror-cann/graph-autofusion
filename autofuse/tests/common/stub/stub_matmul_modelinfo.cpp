/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "stub_matmul_modelinfo.h"

namespace {
att::Expr GetSafeDivisor(const att::Expr &expr) {
  return af::sym::Max(af::sym::kSymbolOne, expr);
}

struct MatmulExprContext {
  att::Expr expr_corenum;
  att::Expr expr_m;
  att::Expr expr_tilem;
  att::Expr expr_basem;
  att::Expr expr_n;
  att::Expr expr_tilen;
  att::Expr expr_basen;
  att::Expr expr_k;
  att::Expr expr_stepka;
  att::Expr expr_stepkb;
  att::Expr expr_basek;
};

void BuildCoreAxis(att::ModelInfo &model_info, MatmulExprContext &ctx) {
  ctx.expr_corenum = att::CreateExpr("block_dim");
  att::SymVarInfoPtr sym_corenum = std::make_shared<att::SymVarInfo>(ctx.expr_corenum);
  att::AttAxisPtr core = std::make_shared<att::AttAxis>();
  core->name = "corenum";
  core->axis_pos = att::AxisPosition::ORIGIN;
  core->bind_multicore = false;
  core->is_last = false;
  core->is_node_innerest_dim = false;
  core->size = sym_corenum;
  model_info.arg_list.emplace_back(core);
}

void BuildMAxes(MatmulExprContext &ctx, att::AttAxisPtr &m, att::AttAxisPtr &tilem, att::AttAxisPtr &basem) {
  ctx.expr_m = att::CreateExpr("m_size");
  ctx.expr_tilem = att::CreateExpr("tilem_size");
  ctx.expr_basem = att::CreateExpr("basem_size");

  att::SymVarInfoPtr sym_m = std::make_shared<att::SymVarInfo>(ctx.expr_m);
  att::SymVarInfoPtr sym_tilem = std::make_shared<att::SymVarInfo>(ctx.expr_tilem);
  sym_tilem->align = ge::Symbol(16);
  sym_tilem->related_scope = {att::HardwareDef::L2};
  att::SymVarInfoPtr sym_basem = std::make_shared<att::SymVarInfo>(ctx.expr_basem);
  sym_basem->align = ge::Symbol(16);
  sym_basem->related_scope = {att::HardwareDef::L0A, att::HardwareDef::L0C, att::HardwareDef::L1};

  m = std::make_shared<att::AttAxis>();
  tilem = std::make_shared<att::AttAxis>();
  basem = std::make_shared<att::AttAxis>();

  m->name = "m";
  m->axis_pos = att::AxisPosition::ORIGIN;
  m->bind_multicore = false;
  m->is_last = false;
  m->is_node_innerest_dim = false;
  m->size = sym_m;

  tilem->name = "tilem";
  tilem->axis_pos = att::AxisPosition::INNER;
  tilem->bind_multicore = false;
  tilem->is_last = false;
  tilem->is_node_innerest_dim = true;
  tilem->size = sym_tilem;
  tilem->orig_axis.push_back(m.get());
  tilem->from_axis = {m.get()};

  basem->name = "basem";
  basem->axis_pos = att::AxisPosition::INNER;
  basem->bind_multicore = false;
  basem->is_last = true;
  basem->is_node_innerest_dim = false;
  basem->size = sym_basem;
  basem->orig_axis.push_back(m.get());
  basem->from_axis = {tilem.get()};
}

void BuildNAxes(MatmulExprContext &ctx, att::AttAxisPtr &n, att::AttAxisPtr &tilen, att::AttAxisPtr &basen) {
  ctx.expr_n = att::CreateExpr("n_size");
  ctx.expr_tilen = att::CreateExpr("tilen_size");
  ctx.expr_basen = att::CreateExpr("basen_size");

  att::SymVarInfoPtr sym_n = std::make_shared<att::SymVarInfo>(ctx.expr_n);
  att::SymVarInfoPtr sym_tilen = std::make_shared<att::SymVarInfo>(ctx.expr_tilen);
  sym_tilen->align = ge::Symbol(16);
  sym_tilen->related_scope = {att::HardwareDef::L2};
  att::SymVarInfoPtr sym_basen = std::make_shared<att::SymVarInfo>(ctx.expr_basen);
  sym_basen->align = ge::Symbol(16);
  sym_basen->related_scope = {att::HardwareDef::L0B, att::HardwareDef::L0C, att::HardwareDef::L1};

  n = std::make_shared<att::AttAxis>();
  tilen = std::make_shared<att::AttAxis>();
  basen = std::make_shared<att::AttAxis>();

  n->name = "n";
  n->axis_pos = att::AxisPosition::ORIGIN;
  n->bind_multicore = false;
  n->is_last = false;
  n->is_node_innerest_dim = false;
  n->size = sym_n;

  tilen->name = "tilen";
  tilen->axis_pos = att::AxisPosition::INNER;
  tilen->bind_multicore = false;
  tilen->is_last = false;
  tilen->is_node_innerest_dim = true;
  tilen->size = sym_tilen;
  tilen->orig_axis.push_back(n.get());
  tilen->from_axis = {n.get()};

  basen->name = "basen";
  basen->axis_pos = att::AxisPosition::INNER;
  basen->bind_multicore = false;
  basen->is_last = true;
  basen->is_node_innerest_dim = true;
  basen->size = sym_basen;
  basen->orig_axis.push_back(n.get());
  basen->from_axis = {tilen.get()};
}

void BuildKAxes(MatmulExprContext &ctx, att::AttAxisPtr &k, att::AttAxisPtr &stepka, att::AttAxisPtr &stepkb,
                att::AttAxisPtr &basek) {
  ctx.expr_k = att::CreateExpr("k_size");
  ctx.expr_stepka = att::CreateExpr("stepka_size");
  ctx.expr_stepkb = att::CreateExpr("stepkb_size");
  ctx.expr_basek = att::CreateExpr("basek_size");

  att::SymVarInfoPtr sym_k = std::make_shared<att::SymVarInfo>(ctx.expr_k);
  att::SymVarInfoPtr sym_stepka = std::make_shared<att::SymVarInfo>(ctx.expr_stepka);
  sym_stepka->align = ge::Symbol(256);
  sym_stepka->related_scope = {att::HardwareDef::L1};
  att::SymVarInfoPtr sym_stepkb = std::make_shared<att::SymVarInfo>(ctx.expr_stepkb);
  sym_stepkb->align = ge::Symbol(16);
  sym_stepkb->related_scope = {att::HardwareDef::L1};
  att::SymVarInfoPtr sym_basek = std::make_shared<att::SymVarInfo>(ctx.expr_basek);
  sym_basek->align = ge::Symbol(16);
  sym_basek->related_scope = {att::HardwareDef::L0A, att::HardwareDef::L0B};

  k = std::make_shared<att::AttAxis>();
  stepka = std::make_shared<att::AttAxis>();
  stepkb = std::make_shared<att::AttAxis>();
  basek = std::make_shared<att::AttAxis>();

  k->name = "k";
  k->axis_pos = att::AxisPosition::ORIGIN;
  k->bind_multicore = false;
  k->is_last = false;
  k->is_node_innerest_dim = false;
  k->size = sym_k;

  stepka->name = "stepka";
  stepka->axis_pos = att::AxisPosition::INNER;
  stepka->bind_multicore = false;
  stepka->is_last = false;
  stepka->is_node_innerest_dim = true;
  stepka->size = sym_stepka;
  stepka->orig_axis.push_back(k.get());
  stepka->from_axis = {k.get()};

  stepkb->name = "stepkb";
  stepkb->axis_pos = att::AxisPosition::INNER;
  stepkb->bind_multicore = false;
  stepkb->is_last = false;
  stepkb->is_node_innerest_dim = true;
  stepkb->size = sym_stepkb;
  stepkb->orig_axis.push_back(k.get());
  stepkb->from_axis = {stepka.get()};

  basek->name = "basek";
  basek->axis_pos = att::AxisPosition::INNER;
  basek->bind_multicore = false;
  basek->is_last = true;
  basek->is_node_innerest_dim = false;
  basek->size = sym_basek;
  basek->orig_axis.push_back(k.get());
  basek->from_axis = {stepkb.get()};
}

void AppendArgList(att::ModelInfo &model_info, const att::AttAxisPtr &m, const att::AttAxisPtr &tilen,
                   const att::AttAxisPtr &tilem, const att::AttAxisPtr &stepka, const att::AttAxisPtr &stepkb,
                   const att::AttAxisPtr &basek, const att::AttAxisPtr &basen, const att::AttAxisPtr &basem,
                   const att::AttAxisPtr &n, const att::AttAxisPtr &k) {
  model_info.arg_list.emplace_back(m);
  model_info.arg_list.emplace_back(tilen);
  model_info.arg_list.emplace_back(tilem);
  model_info.arg_list.emplace_back(stepka);
  model_info.arg_list.emplace_back(stepkb);
  model_info.arg_list.emplace_back(basek);
  model_info.arg_list.emplace_back(basen);
  model_info.arg_list.emplace_back(basem);
  model_info.arg_list.emplace_back(n);
  model_info.arg_list.emplace_back(k);
}

void FillMatmulHardwareCons(att::ModelInfo &model_info, const MatmulExprContext &ctx) {
  model_info.hardware_cons[att::HardwareDef::L0A] = ctx.expr_basem * ctx.expr_basek * att::CreateExpr(4);
  model_info.hardware_cons[att::HardwareDef::L0B] = ctx.expr_basek * ctx.expr_basen * att::CreateExpr(4);
  model_info.hardware_cons[att::HardwareDef::L0C] = ctx.expr_basem * ctx.expr_basen * att::CreateExpr(4);
  model_info.hardware_cons[att::HardwareDef::L1] =
      (ctx.expr_stepka * ctx.expr_basem * att::CreateExpr(4)) + (ctx.expr_stepkb * ctx.expr_basen * att::CreateExpr(4));
  model_info.hardware_cons[att::HardwareDef::L2] =
      (ctx.expr_tilen * ctx.expr_tilem * att::CreateExpr(2)) +
      ((ctx.expr_tilen + ctx.expr_tilem) * ctx.expr_k * att::CreateExpr(2));
  model_info.hardware_cons[att::HardwareDef::UB] = att::CreateExpr(0L);
}

struct MatmulPerfContext {
  att::Expr tile_cnt;
  att::Expr base_cnt;
  att::Expr al1_cnt;
  att::Expr bl1_cnt;
  att::Expr l0_cnt;
};

MatmulPerfContext CalcMatmulLoopCnts(const MatmulExprContext &ctx) {
  MatmulPerfContext perf;
  perf.tile_cnt = ((ctx.expr_n / GetSafeDivisor(ctx.expr_tilen)) * (ctx.expr_m / GetSafeDivisor(ctx.expr_tilem)));
  perf.base_cnt = af::sym::Max(af::sym::kSymbolOne, (((ctx.expr_tilem / GetSafeDivisor(ctx.expr_basem)) *
                                                      (ctx.expr_tilen / GetSafeDivisor(ctx.expr_basen))) /
                                                     GetSafeDivisor(ctx.expr_corenum)));
  perf.al1_cnt = ctx.expr_k / GetSafeDivisor(ctx.expr_stepka);
  perf.bl1_cnt = ctx.expr_stepka / GetSafeDivisor(ctx.expr_stepkb);
  perf.l0_cnt = ctx.expr_stepkb / GetSafeDivisor(ctx.expr_basek);
  return perf;
}

void FillMatmulPerfObjects(att::ModelInfo &model_info, const MatmulExprContext &ctx, const MatmulPerfContext &perf) {
  att::Expr l1_cnt = perf.al1_cnt * perf.bl1_cnt;
  att::Expr al0_mte1 =
      (((ctx.expr_basem * ctx.expr_basek) * att::CreateExpr(2)) / att::CreateExpr(512)) + att::CreateExpr(26);
  att::Expr bl0_mte1 =
      (((ctx.expr_basek * ctx.expr_basen) * att::CreateExpr(2)) / att::CreateExpr(256)) + att::CreateExpr(26);
  att::Expr mte1 = (perf.tile_cnt * perf.base_cnt * l1_cnt * perf.l0_cnt) * (al0_mte1 + bl0_mte1);
  std::cout << "mte1: " << mte1 << std::endl;

  att::Expr l0_mac = af::sym::Ceiling(ctx.expr_basem / att::CreateExpr(16)) *
                     af::sym::Ceiling(ctx.expr_basek / att::CreateExpr(16)) *
                     af::sym::Ceiling(ctx.expr_basen / att::CreateExpr(16));
  att::Expr mac = (perf.tile_cnt * perf.base_cnt * l1_cnt * perf.l0_cnt) * l0_mac;
  std::cout << "mac: " << mac << std::endl;

  att::Expr al1_mte2 =
      (((ctx.expr_basem * ctx.expr_stepka) * att::CreateExpr(2)) /
       (att::CreateExpr(32) / af::sym::Max(af::sym::kSymbolOne, (att::CreateExpr(256) / ctx.expr_stepka)))) +
      att::CreateExpr(210);
  att::Expr bl1_mte2 =
      (((ctx.expr_stepkb * ctx.expr_basen) * att::CreateExpr(2)) /
       (att::CreateExpr(32) / af::sym::Max(af::sym::kSymbolOne, (att::CreateExpr(256) / ctx.expr_basen)))) +
      att::CreateExpr(210);
  att::Expr mte2 = perf.tile_cnt * perf.base_cnt * perf.al1_cnt * (al1_mte2 + (perf.bl1_cnt * bl1_mte2));
  std::cout << "mte2: " << mte2 << std::endl;

  att::Expr base_fixpipe = ((ctx.expr_basem * ctx.expr_basen) * att::CreateExpr(4)) / att::CreateExpr(32);
  att::Expr fixpipe = (perf.tile_cnt * perf.base_cnt) * base_fixpipe;

  model_info.objects[att::PipeType::AIC_MAC] = mac;
  model_info.objects[att::PipeType::AIC_MTE1] = mte1;
  model_info.objects[att::PipeType::AIC_MTE2] = mte2;
  model_info.objects[att::PipeType::AIC_FIXPIPE] = fixpipe;
}

void FillModelInfo(att::ModelInfo &model_info, const MatmulExprContext &ctx) {
  FillMatmulHardwareCons(model_info, ctx);
  MatmulPerfContext perf = CalcMatmulLoopCnts(ctx);
  FillMatmulPerfObjects(model_info, ctx, perf);

  model_info.tiling_case_id = 1;
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_stepka, ctx.expr_stepkb));
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_stepkb, ctx.expr_basek));
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_tilen, ctx.expr_basen));
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_tilem, ctx.expr_basem));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_tilem - ctx.expr_m));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_tilen - ctx.expr_n));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_stepka - ctx.expr_k));
  model_info.container_exprs["Q1"] = (ctx.expr_m + ctx.expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (ctx.expr_m + ctx.expr_n);
  model_info.output_size = 1;
}
}  // namespace

namespace att {
ModelInfo GenMatmulModelInfo() {
  ModelInfo model_info;
  MatmulExprContext ctx;
  BuildCoreAxis(model_info, ctx);

  AttAxisPtr m;
  AttAxisPtr tilem;
  AttAxisPtr basem;
  BuildMAxes(ctx, m, tilem, basem);

  AttAxisPtr n;
  AttAxisPtr tilen;
  AttAxisPtr basen;
  BuildNAxes(ctx, n, tilen, basen);

  AttAxisPtr k;
  AttAxisPtr stepka;
  AttAxisPtr stepkb;
  AttAxisPtr basek;
  BuildKAxes(ctx, k, stepka, stepkb, basek);

  AppendArgList(model_info, m, tilen, tilem, stepka, stepkb, basek, basen, basem, n, k);
  FillModelInfo(model_info, ctx);
  return model_info;
}
}  // namespace att
