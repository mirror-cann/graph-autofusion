/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub_solver_model_info.h"

namespace {
struct SolverExprContext {
  att::Expr expr_m;
  att::Expr expr_tilem;
  att::Expr expr_stepm;
  att::Expr expr_basem;
  att::Expr expr_n;
  att::Expr expr_tilen;
  att::Expr expr_stepn;
  att::Expr expr_basen;
  att::Expr expr_k;
};


void InitDefaultExpr(const ge::ExprType expr_type, att::Expr &default_expr, bool &is_const)
{
  is_const = true;
  if (expr_type == ge::ExprType::kExprConstantRation) {
    default_expr = ge::Symbol(8, "tmp") / ge::Symbol(3, "tmp");
    return;
  }
  if (expr_type == ge::ExprType::kExprConstantInteger) {
    default_expr = ge::Symbol(16, "tmp");
    return;
  }
  if (expr_type == ge::ExprType::kExprVariable) {
    is_const = false;
  }
}

void InitSymVar(att::SymVarInfoPtr &sym, const att::Expr &expr, uint32_t align_val = 0,
                const std::vector<att::HardwareDef> &scope = {})
{
  sym = std::make_shared<att::SymVarInfo>(expr);
  if (align_val > 0) {
    sym->align = ge::Symbol(align_val);
  }
  if (!scope.empty()) {
    sym->related_scope = scope;
  }
}

void SetAxisOrigin(att::AttAxisPtr &axis, const std::string &name, const att::SymVarInfoPtr &size)
{
  axis = std::make_shared<att::AttAxis>();
  axis->name = name;
  axis->axis_pos = att::AxisPosition::ORIGIN;
  axis->bind_multicore = false;
  axis->is_last = false;
  axis->is_node_innerest_dim = false;
  axis->size = size;
}

void SetAxisInner(att::AttAxisPtr &axis, const std::string &name, const att::SymVarInfoPtr &size,
                  bool bind_multicore, bool is_last, att::AttAxis *orig, att::AttAxis *from)
{
  axis = std::make_shared<att::AttAxis>();
  axis->name = name;
  axis->axis_pos = att::AxisPosition::INNER;
  axis->bind_multicore = bind_multicore;
  axis->is_last = is_last;
  axis->is_node_innerest_dim = true;
  axis->size = size;
  axis->orig_axis.push_back(orig);
  axis->from_axis = {from};
}

void BuildMArgList(att::ModelInfo &model_info, const bool is_const, const att::Expr &default_expr,
                   const uint32_t m_align, SolverExprContext &ctx)
{
  ctx.expr_m = is_const ? default_expr : att::CreateExpr("m_size");
  ctx.expr_tilem = is_const ? default_expr : att::CreateExpr("tilem_size");
  ctx.expr_stepm = is_const ? default_expr : att::CreateExpr("stepm_size");
  ctx.expr_basem = att::CreateExpr("basem_size");

  att::SymVarInfoPtr sym_m, sym_tilem, sym_stepm, sym_basem;
  InitSymVar(sym_m, ctx.expr_m);
  sym_m->value_range.first = 1;
  sym_m->value_range.second = 10000;
  sym_m->align = ge::Symbol(m_align);
  InitSymVar(sym_tilem, ctx.expr_tilem, 16, {att::HardwareDef::L2});
  InitSymVar(sym_stepm, ctx.expr_stepm, 16, {att::HardwareDef::L1, att::HardwareDef::CORENUM});
  InitSymVar(sym_basem, ctx.expr_basem, 16, {att::HardwareDef::L0A, att::HardwareDef::L0C});

  att::AttAxisPtr m, tilem, stepm, basem;
  SetAxisOrigin(m, "m", sym_m);
  SetAxisInner(tilem, "tilem", sym_tilem, false, false, m.get(), m.get());
  SetAxisInner(stepm, "stepm", sym_stepm, true, false, m.get(), tilem.get());
  SetAxisInner(basem, "basem", sym_basem, false, true, m.get(), stepm.get());

  model_info.arg_list.emplace_back(m);
  model_info.arg_list.emplace_back(tilem);
  model_info.arg_list.emplace_back(stepm);
  model_info.arg_list.emplace_back(basem);
}

void BuildNArgList(att::ModelInfo &model_info, const bool is_const, const att::Expr &default_expr, SolverExprContext &ctx)
{
  ctx.expr_n = is_const ? default_expr : att::CreateExpr("n_size");
  ctx.expr_tilen = is_const ? default_expr : att::CreateExpr("tilen_size");
  ctx.expr_stepn = is_const ? default_expr : att::CreateExpr("stepn_size");
  ctx.expr_basen = is_const ? default_expr : att::CreateExpr("basen_size");

  att::SymVarInfoPtr sym_n, sym_tilen, sym_stepn, sym_basen;
  InitSymVar(sym_n, ctx.expr_n);
  InitSymVar(sym_tilen, ctx.expr_tilen, 16, {att::HardwareDef::L2});
  InitSymVar(sym_stepn, ctx.expr_stepn, 128, {att::HardwareDef::L1, att::HardwareDef::CORENUM});
  InitSymVar(sym_basen, ctx.expr_basen, 16, {att::HardwareDef::L0B, att::HardwareDef::L0C});

  att::AttAxisPtr n, tilen, stepn, basen;
  SetAxisOrigin(n, "n", sym_n);
  SetAxisInner(tilen, "tilen", sym_tilen, false, false, n.get(), n.get());
  SetAxisInner(stepn, "stepn", sym_stepn, true, false, n.get(), tilen.get());
  SetAxisInner(basen, "basen", sym_basen, false, true, n.get(), stepn.get());

  model_info.arg_list.emplace_back(n);
  model_info.arg_list.emplace_back(tilen);
  model_info.arg_list.emplace_back(stepn);
  model_info.arg_list.emplace_back(basen);
}

void BuildKArg(att::ModelInfo &model_info, SolverExprContext &ctx)
{
  ctx.expr_k = att::CreateExpr("k_size");
  att::SymConstInfoPtr sym_k = std::make_shared<att::SymConstInfo>(ctx.expr_k);
  sym_k->const_value = 128u;
  att::AttAxisPtr k = std::make_shared<att::AttAxis>();
  k->name = "k";
  k->axis_pos = att::AxisPosition::ORIGIN;
  k->bind_multicore = false;
  k->is_last = false;
  k->is_node_innerest_dim = false;
  k->size = sym_k;
  model_info.arg_list.emplace_back(k);
}

void FillModelInfo(att::ModelInfo &model_info, const SolverExprContext &ctx)
{
  att::Expr l0a_occupy = ctx.expr_basem * ctx.expr_k * att::CreateExpr(4);
  att::Expr l0b_occupy = ctx.expr_k * ctx.expr_basen * att::CreateExpr(4);
  att::Expr l0c_occupy = ctx.expr_basem * ctx.expr_basen * att::CreateExpr(4);
  att::Expr l1_occupy = (ctx.expr_k * ctx.expr_stepm * att::CreateExpr(4)) +
                        (ctx.expr_k * ctx.expr_stepn * att::CreateExpr(4));
  att::Expr l2_occupy = (ctx.expr_tilen * ctx.expr_tilem * att::CreateExpr(2)) +
                        ((ctx.expr_tilen + ctx.expr_tilem) * ctx.expr_k * att::CreateExpr(2));
  att::Expr core_num = ((ctx.expr_tilem / ctx.expr_stepm) * (ctx.expr_tilen / ctx.expr_stepn));

  model_info.hardware_cons[att::HardwareDef::L0A] = l0a_occupy;
  model_info.hardware_cons[att::HardwareDef::L0B] = l0b_occupy;
  model_info.hardware_cons[att::HardwareDef::L0C] = l0c_occupy;
  model_info.hardware_cons[att::HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[att::HardwareDef::L2] = l2_occupy;
  model_info.hardware_cons[att::HardwareDef::UB] = ctx.expr_m * att::CreateExpr(10);
  model_info.hardware_cons[att::HardwareDef::CORENUM] = core_num;

  att::Expr mac = ((ctx.expr_basem * ctx.expr_basen * ctx.expr_k) / (att::CreateExpr(16) * att::CreateExpr(256)));
  att::Expr mte = (((ctx.expr_stepm * ctx.expr_k) / att::CreateExpr(32)) +
                   ((ctx.expr_stepn * ctx.expr_k) / att::CreateExpr(32)));
  model_info.objects[att::PipeType::AIC_MAC] = mac;
  model_info.objects[att::PipeType::AIC_MTE2] = mte;
  model_info.tiling_case_id = 0;
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_stepm, ctx.expr_basem));
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_stepn, ctx.expr_basen));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_tilem - ctx.expr_stepm));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_tilen - ctx.expr_stepn));
  model_info.container_exprs["Q1"] = (ctx.expr_m + ctx.expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (ctx.expr_m + ctx.expr_n);
  model_info.output_size = 1;
}
}  // namespace

namespace att {
ModelInfo CreateModelInfo(const uint32_t m_align, const ge::ExprType expr_type)
{
  ModelInfo model_info;
  Expr default_expr;
  bool is_const = true;
  InitDefaultExpr(expr_type, default_expr, is_const);

  SolverExprContext ctx;
  BuildMArgList(model_info, is_const, default_expr, m_align, ctx);

  Optional test_optional;
  test_optional.optional_name = "test";
  test_optional.data_type = "int32_t";
  test_optional.min_value = "1";
  test_optional.max_value = "100";
  InputTensor input_tensor;
  input_tensor.data_type = 1;

  BuildNArgList(model_info, is_const, default_expr, ctx);
  BuildKArg(model_info, ctx);
  FillModelInfo(model_info, ctx);
  return model_info;
}
}  // namespace att
