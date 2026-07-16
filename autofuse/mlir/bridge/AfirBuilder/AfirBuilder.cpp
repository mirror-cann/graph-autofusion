/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AfirBuilder.cpp - Build an AFIR module from neutral schedule info -*- C++ -*-===//

#include "AfirBuilder/AfirBuilder.h"

#include "AFIR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace af_demo {
namespace {

Type parseDtype(MLIRContext *ctx, StringRef dtype) {
  return llvm::StringSwitch<Type>(dtype)
      .Cases("float16", "f16", Float16Type::get(ctx))
      .Cases("float32", "f32", Float32Type::get(ctx))
      .Cases("bfloat16", "bf16", BFloat16Type::get(ctx))
      .Case("int8", IntegerType::get(ctx, 8))
      .Case("int16", IntegerType::get(ctx, 16))
      .Case("int32", IntegerType::get(ctx, 32))
      .Case("int64", IntegerType::get(ctx, 64))
      .Case("uint8", IntegerType::get(ctx, 8, IntegerType::Unsigned))
      .Case("uint16", IntegerType::get(ctx, 16, IntegerType::Unsigned))
      .Case("uint32", IntegerType::get(ctx, 32, IntegerType::Unsigned))
      .Case("uint64", IntegerType::get(ctx, 64, IntegerType::Unsigned))
      .Default(Type());
}

RankedTensorType makeTensorType(MLIRContext *ctx, const TensorInfo &t) {
  Type elem = parseDtype(ctx, t.dtype);
  if (!elem) {
    return {};
  }
  return RankedTensorType::get(t.shape, elem);
}

afir::AxisType parseAxisType(StringRef s) {
  return llvm::StringSwitch<afir::AxisType>(s)
      .Case("original", afir::AxisType::Original)
      .Case("block_outer", afir::AxisType::BlockOuter)
      .Case("block_inner", afir::AxisType::BlockInner)
      .Case("tile_outer", afir::AxisType::TileOuter)
      .Case("tile_inner", afir::AxisType::TileInner)
      .Case("merged", afir::AxisType::Merged)
      .Default(afir::AxisType::Invalid);
}

// Build the per-op `outputs` attribute (AscTensorGroups) carrying tensor_id +
// memory position — this is where per-tensor schedule/mem info rides on AFIR.
// (vectorized_axis is emitted separately as a discardable op attr in buildNodeOp:
// the AscTensorGroups assemblyFormat binds vectorized_axis/strides in one group
// and can't round-trip a non-empty axis list with empty strides.)
ArrayAttr makeOutputsAttr(MLIRContext *ctx, const TensorInfo &t) {
  auto pos = static_cast<afir::Position>(t.position);
  auto posCfg = afir::PositionConfigAttr::get(ctx, pos, /*depth=*/0, /*is_double_buffer=*/false);
  auto group = afir::AscTensorGroupsAttr::get(ctx, /*vectorized_axis=*/{}, /*vectorized_strides=*/{},
                                              /*tensor_id=*/t.tensor_id, /*reuse_id=*/-1, posCfg,
                                              /*position_id=*/-1);
  return ArrayAttr::get(ctx, {group});
}

// Attach the graph's loop axes as an `afir.asc_graph` attribute on the func —
// carrying SchedInfo (axis order/type/size) for later tiling passes.
Attribute makeGraphAttr(MLIRContext *ctx, const GraphInfo &g) {
  SmallVector<afir::AxisAttr> axes;
  for (const auto &a : g.axes) {
    axes.push_back(afir::AxisAttr::get(ctx, a.id, StringAttr::get(ctx, a.name), parseAxisType(a.type),
                                       /*bind_block=*/false, StringAttr::get(ctx, a.size),
                                       /*align=*/StringAttr(), /*from=*/{}));
  }
  SmallVector<StringAttr> sizeVars;
  for (const auto &sv : g.size_vars) {
    sizeVars.push_back(StringAttr::get(ctx, sv));
  }
  return afir::AscGraphAttrGroupsAttr::get(ctx, g.tiling_key, axes, afir::AscGraphType::Compute, sizeVars);
}

// Create the AFIR op for one node. Returns the produced Value (null on error).
Value buildNodeOp(OpBuilder &b, Location loc, const NodeInfo &node, ArrayRef<Value> operands,
                  RankedTensorType resultTy) {
  MLIRContext *ctx = b.getContext();
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(StringAttr::get(ctx, "outputs"), makeOutputsAttr(ctx, node.output));

  // Per-op schedule info for the tiling-aware lowering: the loop-nest axis ids
  // (outer->inner) and the innermost loop axis. Carried as discardable attrs
  // `afir.sched_axis` / `afir.loop_axis` (SchedInfo has no dedicated AFIR attr).
  if (!node.sched_axis.empty()) {
    attrs.emplace_back(StringAttr::get(ctx, "afir.sched_axis"), b.getDenseI64ArrayAttr(node.sched_axis));
  }
  if (node.loop_axis != -1) {
    attrs.emplace_back(StringAttr::get(ctx, "afir.loop_axis"), b.getI64IntegerAttr(node.loop_axis));
  }
  // vectorized_axis (the buffer's API vector-length axes) as a discardable op
  // attr — see makeOutputsAttr for why it's not in the AscTensorGroups attr.
  if (!node.output.vectorized_axis.empty()) {
    attrs.emplace_back(StringAttr::get(ctx, "afir.vectorized_axis"),
                       b.getDenseI64ArrayAttr(node.output.vectorized_axis));
  }

  const StringRef ty = node.type;
  auto make = [&](auto tag) -> Value {
    using OpTy = decltype(tag);
    Operation *op = b.create<OpTy>(loc, TypeRange{resultTy}, ValueRange(operands), attrs);
    return op->getResult(0);
  };

  if (ty == "Load") return make(afir::LoadOp{});
  if (ty == "Store") return make(afir::StoreOp{});
  if (ty == "Broadcast") return make(afir::BroadcastOp{});
  if (ty == "Cast") return make(afir::CastOp{});
  if (ty == "Add") return make(afir::AddOp{});
  if (ty == "Sub") return make(afir::SubOp{});
  if (ty == "Mul") return make(afir::MulOp{});
  if (ty == "Div") return make(afir::DivOp{});
  // unary elementwise
  if (ty == "Abs") return make(afir::AbsOp{});
  if (ty == "Exp") return make(afir::ExpOp{});
  if (ty == "Relu") return make(afir::ReluOp{});
  if (ty == "Sqrt") return make(afir::SqrtOp{});
  if (ty == "Neg") return make(afir::NegOp{});

  emitError(loc) << "AfirBuilder: unsupported node type '" << node.type << "'";
  return {};
}

LogicalResult appendTensorTypes(MLIRContext *ctx, ArrayRef<IoTensor> tensors, SmallVectorImpl<Type> &types) {
  for (const auto &tensor : tensors) {
    auto type = makeTensorType(ctx, tensor.info);
    if (!type) return failure();
    types.push_back(type);
  }
  return success();
}

llvm::StringMap<const TensorInfo *> makeLogicalOverrides(const GraphInfo &graph) {
  llvm::StringMap<const TensorInfo *> overrides;
  for (const auto &node : graph.nodes) {
    if (node.type == "Output" && node.input_names.size() == 1) {
      overrides[node.input_names[0]] = &node.output;
    }
  }
  return overrides;
}

FailureOr<bool> handleMarkerNode(const NodeInfo &node, llvm::StringMap<Value> &valueMap, Location loc) {
  if (node.type == "Data") {
    if (!valueMap.count(node.name)) {
      emitError(loc) << "AfirBuilder: Data node '" << node.name << "' is not a kernel input";
      return failure();
    }
    return true;
  }
  if (node.type != "Output") return false;
  if (node.input_names.size() != 1 || !valueMap.count(node.input_names[0])) {
    emitError(loc) << "AfirBuilder: Output node '" << node.name << "' needs one known input";
    return failure();
  }
  valueMap[node.name] = valueMap[node.input_names[0]];
  return true;
}

LogicalResult buildComputeNode(OpBuilder &builder, Location loc, const NodeInfo &node,
                               const llvm::StringMap<const TensorInfo *> &logicalOverrides,
                               llvm::StringMap<Value> &valueMap) {
  SmallVector<Value> operands;
  for (const auto &inputName : node.input_names) {
    auto input = valueMap.find(inputName);
    if (input == valueMap.end()) {
      return emitError(loc) << "AfirBuilder: node '" << node.name << "' references undefined value '" << inputName
                            << "'",
             failure();
    }
    operands.push_back(input->second);
  }

  const TensorInfo *resultInfo = &node.output;
  if (auto override = logicalOverrides.find(node.name); override != logicalOverrides.end()) {
    resultInfo = override->second;
  }
  auto resultType = makeTensorType(builder.getContext(), *resultInfo);
  if (!resultType) return failure();
  Value result = buildNodeOp(builder, loc, node, operands, resultType);
  if (!result) return failure();
  valueMap[node.name] = result;
  return success();
}

LogicalResult buildGraphNodes(OpBuilder &builder, Location loc, const GraphInfo &graph,
                              llvm::StringMap<Value> &valueMap) {
  auto logicalOverrides = makeLogicalOverrides(graph);
  for (const auto &node : graph.nodes) {
    FailureOr<bool> handled = handleMarkerNode(node, valueMap, loc);
    if (failed(handled)) return failure();
    if (*handled) continue;
    if (failed(buildComputeNode(builder, loc, node, logicalOverrides, valueMap))) return failure();
  }
  return success();
}

LogicalResult buildReturn(OpBuilder &builder, Location loc, ArrayRef<IoTensor> outputs,
                          const llvm::StringMap<Value> &valueMap) {
  SmallVector<Value> returns;
  for (const auto &output : outputs) {
    auto value = valueMap.find(output.name);
    if (value == valueMap.end()) {
      return emitError(loc) << "AfirBuilder: kernel output '" << output.name << "' was never produced", failure();
    }
    returns.push_back(value->second);
  }
  func::ReturnOp::create(builder, loc, ValueRange(returns));
  return success();
}

LogicalResult buildGraphFunc(OpBuilder &moduleBuilder, const KernelInfo &kernel, const GraphInfo &graph) {
  MLIRContext *ctx = moduleBuilder.getContext();
  SmallVector<Type> argTypes, resultTypes;
  if (failed(appendTensorTypes(ctx, kernel.inputs, argTypes)) ||
      failed(appendTensorTypes(ctx, kernel.outputs, resultTypes))) {
    return failure();
  }

  auto funcTy = FunctionType::get(ctx, argTypes, resultTypes);
  auto func = func::FuncOp::create(moduleBuilder, moduleBuilder.getUnknownLoc(), graph.name, funcTy);
  func->setAttr("afir.asc_graph", makeGraphAttr(ctx, graph));

  Block *entry = func.addEntryBlock();
  OpBuilder b(entry, entry->end());
  Location loc = func.getLoc();

  llvm::StringMap<Value> valueMap;
  for (auto [idx, in] : llvm::enumerate(kernel.inputs)) {
    valueMap[in.name] = entry->getArgument(idx);
  }
  if (failed(buildGraphNodes(b, loc, graph, valueMap))) return failure();
  return buildReturn(b, loc, kernel.outputs, valueMap);
}

}  // namespace

OwningOpRef<ModuleOp> BuildAfirModule(MLIRContext &context, const KernelInfo &kernel) {
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc(), kernel.name);
  builder.setInsertionPointToStart(module->getBody());

  for (const auto &graph : kernel.graphs) {
    if (failed(buildGraphFunc(builder, kernel, graph))) {
      return nullptr;
    }
  }

  if (failed(verify(*module))) {
    return nullptr;
  }
  return module;
}

}  // namespace af_demo
