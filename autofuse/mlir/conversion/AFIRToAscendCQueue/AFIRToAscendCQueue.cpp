/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRToAscendCQueue.cpp - queue-faithful AFIR lowering -----*- C++ -*-===//
//
// Autofuse MLIR migration demo — lowers a buffer-annotated AFIR func to the
// full queue-driven ascendc form that a real Ascend C vector kernel compiles
// from:
//
//   func(%gm..., %ws, %tiling) {ascendc.aicore, ascendc.global, cann.num_inputs}
//     %pipe = ascendc.pipe
//     per afir.load  : queue<vecin>  + init_queue + alloc + global_tensor
//                      + set_global_buffer + data_copy_l2(local,global)
//                      + enque + deque
//     per afir.<bin> : queue<pos>    + init_queue + alloc
//                      + <op>_l2 {ascendc.unit="AiCore.Vector"} + enque + deque
//     per afir.store : global_tensor + set_global_buffer
//                      + data_copy_l2(global,local) + free
//
// The lowering *honors* the schedule's positions (read from each op's `outputs`
// AscTensorGroups attr) — it does not re-decide placement. This is the demo's
// central teaching point: codegen is a faithful emitter of upstream decisions.
//
// Scope: single vector kernel, one func, elementwise + load/store. Tiling is
// a fixed calCount for the demo (real tiling comes from the host tiling func).
//
//===----------------------------------------------------------------------===//

#include "AFIRToAscendCQueue/AFIRToAscendCQueue.h"

#include "AFIR/AFIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

using namespace mlir;

namespace {

// GM memory space marker used on kernel-arg memrefs (matches mlir-af: 22).
constexpr int64_t kGmMemSpace = 22;

// Map an AFIR scheduled Position to an ascendc TPosition.
ascendc::TPosition toTPosition(afir::Position pos) {
  switch (pos) {
    case afir::Position::VECTOR_IN:
      return ascendc::TPosition::VECIN;
    case afir::Position::VECTOR_OUT:
      return ascendc::TPosition::VECOUT;
    default:
      return ascendc::TPosition::VECCALC;
  }
}

// Read the scheduled position off an AFIR op's `outputs` attribute.
afir::Position scheduledPosition(Operation *op) {
  auto outs = op->getAttrOfType<ArrayAttr>("outputs");
  if (!outs || outs.empty()) {
    return afir::Position::VECTOR_CALC;
  }
  if (auto g = dyn_cast<afir::AscTensorGroupsAttr>(outs[0])) {
    return g.getPositionConfig().getPosition();
  }
  return afir::Position::VECTOR_CALC;
}

// The ascendc L2 binary op mnemonic for an AFIR elementwise op.
StringRef l2Mnemonic(Operation *op) {
  return llvm::TypeSwitch<Operation *, StringRef>(op)
      .Case<afir::AddOp>([](auto) { return "ascendc.add_l2"; })
      .Case<afir::SubOp>([](auto) { return "ascendc.sub_l2"; })
      .Case<afir::MulOp>([](auto) { return "ascendc.mul_l2"; })
      .Case<afir::DivOp>([](auto) { return "ascendc.div_l2"; })
      .Default([](auto) { return StringRef(); });
}

// The ascendc L2 unary op mnemonic for an AFIR elementwise unary op. Only the
// unary ops that have a direct ascendc L2 counterpart are mapped (see the Asc
// dialect's OpVecUnary.td); the rest fall through to the unsupported-op path.
StringRef l2UnaryMnemonic(Operation *op) {
  return llvm::TypeSwitch<Operation *, StringRef>(op)
      .Case<afir::AbsOp>([](auto) { return "ascendc.abs_l2"; })
      .Case<afir::ExpOp>([](auto) { return "ascendc.exp_l2"; })
      .Case<afir::LnOp>([](auto) { return "ascendc.ln_l2"; })
      .Case<afir::SqrtOp>([](auto) { return "ascendc.sqrt_l2"; })
      .Case<afir::RsqrtOp>([](auto) { return "ascendc.rsqrt_l2"; })
      .Case<afir::ReciprocalOp>([](auto) { return "ascendc.reciprocal_l2"; })
      .Case<afir::ReluOp>([](auto) { return "ascendc.relu_l2"; })
      .Case<afir::NegOp>([](auto) { return "ascendc.neg_l2"; })
      .Case<afir::LogicalNotOp>([](auto) { return "ascendc.not_l2"; })
      .Default([](auto) { return StringRef(); });
}

// Static element count of a shaped type (product of its dims). The demo assumes
// fully static shapes.
int64_t numElems(ShapedType t) {
  int64_t n = 1;
  for (int64_t d : t.getShape()) n *= d;
  return n;
}

struct QueueLowering {
  OpBuilder &b;
  MLIRContext *ctx;
  Value pipe;
  // Value-keyed constant caches. The emitter names constants by value and emits
  // one `constexpr` per distinct SSA constant, so reusing a Value avoids emitting
  // duplicate C++ declarations (e.g. two tensors of the same size, or a shape
  // dim that equals the queue depth).
  llvm::DenseMap<int64_t, Value> i32Cache;
  llvm::DenseMap<int64_t, Value> idxCache;
  // Copy-param caches. DataCopy(Pad)ExtParams for a given (count, elem) are
  // identical for every load/store in a tile, and the emitter materializes one
  // C++ declaration per distinct SSA value — so caching both dedups the params
  // AND the `constexpr`/`half` constants they embed (isPad, paddingValue),
  // which is what used to force a post-processing "hoist the duplicate constant"
  // workaround. Keyed on the (loop-invariant) count Value and elem type.
  llvm::DenseMap<Value, Value> copyExtCache;
  llvm::DenseMap<Type, Value> copyPadExtCache;

  Value i32Const(Location loc, int64_t v) {
    auto it = i32Cache.find(v);
    if (it != i32Cache.end()) return it->second;
    Value c = b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(v));
    i32Cache[v] = c;
    return c;
  }
  Value idxConst(Location loc, int64_t v) {
    auto it = idxCache.find(v);
    if (it != idxCache.end()) return it->second;
    Value c = b.create<arith::ConstantIndexOp>(loc, v);
    idxCache[v] = c;
    return c;
  }

  ascendc::LocalTensorType localTy(ShapedType t) {
    return ascendc::LocalTensorType::get(t.getShape(), t.getElementType());
  }
  ascendc::GlobalTensorType globalTy(ShapedType t) {
    return ascendc::GlobalTensorType::get(t.getShape(), t.getElementType());
  }
  ascendc::QueueType queueTy(ascendc::TPosition pos) {
    return ascendc::QueueType::get(ctx, pos, /*depth=*/1);
  }

  // Per-tensor element count (index) for data_copy / compute calCount.
  Value countOf(Location loc, ShapedType t) {
    return idxConst(loc, numElems(t));
  }
  // Per-tensor buffer byte length (index) for init_queue.
  Value byteLenOf(Location loc, ShapedType t) {
    return idxConst(loc, numElems(t) * (t.getElementTypeBitWidth() / 8));
  }

  //=== Tiling-aware helpers (M2-3): counts/offsets are SSA values read from the
  //=== runtime TilingData struct, never static constants. ==================

  // Read a scalar field off the tiling py_struct as an `index` SSA value:
  //   %m = emitasc.member %tiling "field" : <py_struct>, i64
  //   %i = arith.index_cast %m : i64 to index
  Value tilingField(Location loc, Value tiling, StringRef field) {
    Value m = b.create<emitasc::MemberOp>(loc, b.getIntegerType(64), tiling, b.getStringAttr(field));
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), m);
  }

  // 1-D dynamic local/global tensor of `elem` (a tile view; extent is runtime).
  ascendc::LocalTensorType localTy1D(Type elem) {
    return ascendc::LocalTensorType::get({ShapedType::kDynamic}, elem);
  }
  ascendc::GlobalTensorType globalTy1D(Type elem) {
    return ascendc::GlobalTensorType::get({ShapedType::kDynamic}, elem);
  }

  // Build a queue + init_queue sized for `bufLen` elements of `elem` (byte length
  // = bufLen * elemBytes, SSA). bufLen is the MAX tile size (tile_len) so the
  // buffer fits every iteration incl. the (smaller) tail — only the copy/compute
  // `count` shrinks on the tail, never the buffer.
  //
  // MUST be hoisted OUTSIDE the tile loop: InitBuffer is a linear allocation
  // against the pipe's UB pool, so calling it per tile leaks a buffer every
  // iteration and exhausts the pool after a fixed tile count (the CPU functional
  // sim catches this; real HW only survives because its 192KB UB is large enough
  // for the demo's tile counts). The buffer is allocated once and recycled each
  // tile via AllocTensor/FreeTensor.
  Value makeQueueTiled(Location loc, ascendc::TPosition pos, Value bufLen, Type elem) {
    Value q = b.create<ascendc::QueueOp>(loc, queueTy(pos));
    int64_t elemBytes = elem.getIntOrFloatBitWidth() / 8;
    Value bytes = b.create<arith::MulIOp>(loc, bufLen, idxConst(loc, elemBytes));
    b.create<ascendc::TPipeInitQueueOp>(loc, pipe, q, /*depth=*/i32Const(loc, 1), bytes);
    return q;
  }

  // DataCopyExtParams for one contiguous runtime-sized block:
  // {blockCount=1, blockLen=count*elemBytes, srcStride=0, dstStride=0, rsv=0}.
  // C++ struct has {uint16, uint32, uint32, uint32, uint32} fields; MLIR operands
  // are signless i32, so we attach `types` attr → emitter casts via static_cast.
  Value makeCopyExtParams(Location loc, Value count, Type elem) {
    if (Value c = copyExtCache.lookup(count)) return c;
    Value bytesIdx = b.create<arith::MulIOp>(loc, count, idxConst(loc, elem.getIntOrFloatBitWidth() / 8));
    Type ui16 = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    Type ui32 = IntegerType::get(ctx, 32, IntegerType::Unsigned);
    Value blockCount = i32Const(loc, 1);
    Value blockLenI32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), bytesIdx);
    Value zero = i32Const(loc, 0);
    Type paramsTy = ascendc::DataCopyExtParamsType::get(ctx);
    auto typesAttr = ArrayAttr::get(
        ctx, {TypeAttr::get(ui16), TypeAttr::get(ui32), TypeAttr::get(ui32), TypeAttr::get(ui32), TypeAttr::get(ui32)});
    Value c =
        b.create<ascendc::ConstructOp>(loc, paramsTy, ValueRange{blockCount, blockLenI32, zero, zero, zero}, typesAttr);
    copyExtCache[count] = c;
    return c;
  }

  // Build the padding parameters with the scalar types expected by the emitter.
  Value makeCopyPadExtParams(Location loc, Type elem) {
    if (Value c = copyPadExtCache.lookup(elem)) return c;
    Type i1 = b.getI1Type();
    Type ui8 = IntegerType::get(ctx, 8, IntegerType::Unsigned);
    Value isPad = b.create<arith::ConstantOp>(loc, IntegerAttr::get(i1, 1));
    Value zero8 = i32Const(loc, 0);  // compile-time 0; safe for uint8 fields
    Value padValue;
    if (isa<FloatType>(elem)) {
      padValue = b.create<arith::ConstantOp>(loc, FloatAttr::get(elem, 0.0));
    } else {
      padValue = b.create<arith::ConstantOp>(loc, IntegerAttr::get(elem, 0));
    }
    Type paramsTy = ascendc::DataCopyPadExtParamsType::get(ctx, elem);
    auto typesAttr =
        ArrayAttr::get(ctx, {TypeAttr::get(i1), TypeAttr::get(ui8), TypeAttr::get(ui8), TypeAttr::get(elem)});
    Value c = b.create<ascendc::ConstructOp>(loc, paramsTy, ValueRange{isPad, zero8, zero8, padValue}, typesAttr);
    copyPadExtCache[elem] = c;
    return c;
  }

  // Free a tile tensor back to its queue, recycling the depth-1 buffer for the
  // next iteration. Without this the queue's single buffer is never returned and
  // the pipe's UB pool is exhausted after a fixed tile count.
  void freeTile(Location loc, Value q, Value tensor) {
    b.create<ascendc::TQueBindFreeTensorOp>(loc, q, tensor);
  }

  // afir.load, one tile: GM[offset] -> VECIN local tile. `q` is created once
  // outside the loop (buffer sized to tile_len max); each tile only Alloc/copy/
  // Enque/Deque, then the caller frees the returned tensor after it's consumed.
  // DataCopyPad handles an arbitrary non-32B-aligned tail.
  Value lowerLoadTiled(Location loc, Value q, Value gmMemref, Type elem, Value count, Value offset) {
    Value t = b.create<ascendc::TQueBindAllocTensorOp>(loc, localTy1D(elem), q);
    Value g = b.create<ascendc::GlobalTensorOp>(loc, globalTy1D(elem));
    b.create<ascendc::GlobalTensorSetGlobalBufferOp>(loc, g, gmMemref, /*offset=*/i32Const(loc, 0));
    // operator[] (subindex) returns an OFFSET VIEW; operator() returns a scalar
    // ref — so we must use subindex here for the per-tile GM window.
    Value gAtOff = b.create<ascendc::GlobalTensorSubIndexOp>(loc, globalTy1D(elem), g, offset);
    Value ext = makeCopyExtParams(loc, count, elem);
    Value pad = makeCopyPadExtParams(loc, elem);
    b.create<ascendc::DataCopyPadExtL0Op>(loc, t, gAtOff, ext, pad);
    b.create<ascendc::TQueBindEnqueTensorOp>(loc, q, t);
    return b.create<ascendc::TQueBindDequeTensorOp>(loc, localTy1D(elem), q);
  }

  // afir.<binary>, one tile. `q` pre-made outside the loop.
  Value lowerBinaryTiled(Operation *op, Value q, Value lhs, Value rhs, Type elem, Value count) {
    Location loc = op->getLoc();
    Value dst = b.create<ascendc::TQueBindAllocTensorOp>(loc, localTy1D(elem), q);
    OperationState st(loc, l2Mnemonic(op));
    st.addOperands({dst, lhs, rhs, count});
    st.addAttribute("ascendc.unit", b.getStringAttr("AiCore.Vector"));
    b.create(st);
    b.create<ascendc::TQueBindEnqueTensorOp>(loc, q, dst);
    return b.create<ascendc::TQueBindDequeTensorOp>(loc, localTy1D(elem), q);
  }

  // afir.<unary>, one tile. `q` pre-made outside the loop.
  Value lowerUnaryTiled(Operation *op, Value q, Value src, Type elem, Value count) {
    Location loc = op->getLoc();
    Value dst = b.create<ascendc::TQueBindAllocTensorOp>(loc, localTy1D(elem), q);
    OperationState st(loc, l2UnaryMnemonic(op));
    st.addOperands({dst, src, count});
    st.addAttribute("ascendc.unit", b.getStringAttr("AiCore.Vector"));
    b.create(st);
    b.create<ascendc::TQueBindEnqueTensorOp>(loc, q, dst);
    return b.create<ascendc::TQueBindDequeTensorOp>(loc, localTy1D(elem), q);
  }

  // afir.store, one tile: local tile -> GM[offset], `count` elems.
  // DataCopyPad ext handles an arbitrary non-32B-aligned tail.
  void lowerStoreTiled(Location loc, Value src, Value gmMemref, Type elem, Value count, Value offset) {
    Value g = b.create<ascendc::GlobalTensorOp>(loc, globalTy1D(elem));
    b.create<ascendc::GlobalTensorSetGlobalBufferOp>(loc, g, gmMemref, /*offset=*/i32Const(loc, 0));
    // operator[] (subindex) = offset view, not operator() (scalar ref).
    Value gAtOff = b.create<ascendc::GlobalTensorSubIndexOp>(loc, globalTy1D(elem), g, offset);
    Value ext = makeCopyExtParams(loc, count, elem);
    b.create<ascendc::DataCopyPadExtL2Op>(loc, gAtOff, src, ext);
  }

  // Build a queue + init_queue sized for `shaped`, returning the queue value.
  Value makeQueue(Location loc, ascendc::TPosition pos, ShapedType shaped) {
    Value q = b.create<ascendc::QueueOp>(loc, queueTy(pos));
    b.create<ascendc::TPipeInitQueueOp>(loc, pipe, q, /*depth=*/i32Const(loc, 1), byteLenOf(loc, shaped));
    return q;
  }

  // afir.load : GM memref arg -> VECIN local tensor (returns dequed tensor).
  Value lowerLoad(Location loc, Value gmMemref, ShapedType shaped, ascendc::TPosition pos) {
    Value q = makeQueue(loc, pos, shaped);
    Value t = b.create<ascendc::TQueBindAllocTensorOp>(loc, localTy(shaped), q);
    Value g = b.create<ascendc::GlobalTensorOp>(loc, globalTy(shaped));
    b.create<ascendc::GlobalTensorSetGlobalBufferOp>(loc, g, gmMemref, /*offset=*/i32Const(loc, 0));
    b.create<ascendc::DataCopyL2Op>(loc, t, g, countOf(loc, shaped));
    b.create<ascendc::TQueBindEnqueTensorOp>(loc, q, t);
    return b.create<ascendc::TQueBindDequeTensorOp>(loc, localTy(shaped), q);
  }

  // afir.<binary> : two local tensors -> local tensor at scheduled position.
  Value lowerBinary(Operation *op, Value lhs, Value rhs, ShapedType shaped, ascendc::TPosition pos) {
    Location loc = op->getLoc();
    Value q = makeQueue(loc, pos, shaped);
    Value dst = b.create<ascendc::TQueBindAllocTensorOp>(loc, localTy(shaped), q);

    OperationState st(loc, l2Mnemonic(op));
    st.addOperands({dst, lhs, rhs, countOf(loc, shaped)});
    st.addAttribute("ascendc.unit", b.getStringAttr("AiCore.Vector"));
    b.create(st);

    b.create<ascendc::TQueBindEnqueTensorOp>(loc, q, dst);
    return b.create<ascendc::TQueBindDequeTensorOp>(loc, localTy(shaped), q);
  }

  // afir.<unary> : one local tensor -> local tensor at scheduled position.
  Value lowerUnary(Operation *op, Value src, ShapedType shaped, ascendc::TPosition pos) {
    Location loc = op->getLoc();
    Value q = makeQueue(loc, pos, shaped);
    Value dst = b.create<ascendc::TQueBindAllocTensorOp>(loc, localTy(shaped), q);

    OperationState st(loc, l2UnaryMnemonic(op));
    st.addOperands({dst, src, countOf(loc, shaped)});
    st.addAttribute("ascendc.unit", b.getStringAttr("AiCore.Vector"));
    b.create(st);

    b.create<ascendc::TQueBindEnqueTensorOp>(loc, q, dst);
    return b.create<ascendc::TQueBindDequeTensorOp>(loc, localTy(shaped), q);
  }

  // afir.broadcast : one local tensor (srcShaped) -> larger local tensor
  // (dstShaped) at the scheduled position. Shapes ride as i32 constant operands;
  // the ascendc broadcast_l2 op carries the rank, and the emitter recovers the
  // stretched axis from the tensor types.
  Value lowerBroadcast(Operation *op, Value src, ShapedType dstShaped, ShapedType srcShaped, ascendc::TPosition pos) {
    Location loc = op->getLoc();
    Value q = makeQueue(loc, pos, dstShaped);
    Value dst = b.create<ascendc::TQueBindAllocTensorOp>(loc, localTy(dstShaped), q);

    SmallVector<Value> dstShapeVals, srcShapeVals;
    for (int64_t d : dstShaped.getShape()) dstShapeVals.push_back(i32Const(loc, d));
    for (int64_t d : srcShaped.getShape()) srcShapeVals.push_back(i32Const(loc, d));

    auto bc = b.create<ascendc::BroadcastL2Op>(loc, dst, src, ValueRange(dstShapeVals), ValueRange(srcShapeVals),
                                               static_cast<uint32_t>(dstShaped.getRank()));
    bc->setAttr("ascendc.unit", b.getStringAttr("AiCore.Vector"));

    b.create<ascendc::TQueBindEnqueTensorOp>(loc, q, dst);
    return b.create<ascendc::TQueBindDequeTensorOp>(loc, localTy(dstShaped), q);
  }

  // afir.store : local tensor -> GM memref arg.
  void lowerStore(Location loc, Value src, Value gmMemref, ShapedType shaped, Value queueForFree) {
    Value g = b.create<ascendc::GlobalTensorOp>(loc, globalTy(shaped));
    b.create<ascendc::GlobalTensorSetGlobalBufferOp>(loc, g, gmMemref, /*offset=*/i32Const(loc, 0));
    b.create<ascendc::DataCopyL2Op>(loc, g, src, countOf(loc, shaped));
    if (queueForFree) {
      b.create<ascendc::TQueBindFreeTensorOp>(loc, queueForFree, src);
    }
  }
};

struct ConvertAFIRToAscendCQueuePass : public PassWrapper<ConvertAFIRToAscendCQueuePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAFIRToAscendCQueuePass)

  StringRef getArgument() const final {
    return "convert-afir-to-ascendc-queue";
  }
  StringRef getDescription() const final {
    return "Lower buffer-annotated AFIR to queue-faithful ascendc (demo: vector elewise)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ascendc::AscendCDialect, emitasc::EmitAscDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto func : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (failed(lowerFunc(func))) {
        signalPassFailure();
        return;
      }
    }
  }

  struct FunctionState {
    MLIRContext *ctx;
    func::FuncOp func;
    Block &block;
    ArrayRef<Operation *> afirOps;
    func::ReturnOp returnOp;
    const llvm::DenseMap<Value, unsigned> &resultArgIndex;
    unsigned originalArgumentCount;
    unsigned inputCount;
    OpBuilder &builder;
    QueueLowering &queueLowering;
    IRMapping &mapping;

    Value gmArgument(unsigned index) const {
      return block.getArgument(originalArgumentCount + index);
    }
  };

  struct TiledOperationState {
    FunctionState &function;
    QueueLowering &lowering;
    llvm::DenseMap<Operation *, Value> &operationQueues;
    llvm::DenseMap<Value, Value> &tensorQueues;
    Value actual;
    Value offset;
  };

  static bool containsAfir(func::FuncOp func) {
    bool found = false;
    func.walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "afir") found = true;
    });
    return found;
  }

  static SmallVector<Type> buildKernelArgumentTypes(func::FuncOp func, MLIRContext *ctx) {
    SmallVector<Type> types;
    auto toGmMemref = [&](Type type) {
      auto shaped = cast<ShapedType>(type);
      return MemRefType::get({ShapedType::kDynamic}, shaped.getElementType(), MemRefLayoutAttrInterface{},
                             IntegerAttr::get(IntegerType::get(ctx, 32), kGmMemSpace));
    };
    for (Type type : func.getArgumentTypes()) types.push_back(toGmMemref(type));
    for (Type type : func.getResultTypes()) types.push_back(toGmMemref(type));
    types.push_back(MemRefType::get({}, IntegerType::get(ctx, 8, IntegerType::Unsigned), MemRefLayoutAttrInterface{},
                                    IntegerAttr::get(IntegerType::get(ctx, 32), kGmMemSpace)));
    auto i64Type = IntegerType::get(ctx, 64);
    types.push_back(emitasc::PyStructType::get(
        ctx, StringAttr::get(ctx, "TilingData"), ArrayAttr::get(ctx, {TypeAttr::get(i64Type), TypeAttr::get(i64Type)}),
        ArrayAttr::get(ctx, {StringAttr::get(ctx, "tile_len"), StringAttr::get(ctx, "total")})));
    return types;
  }

  static LogicalResult collectAfirOperations(Block &block, SmallVectorImpl<Operation *> &afirOps,
                                             func::ReturnOp &returnOp) {
    for (Operation &op : block) {
      if (op.getDialect() && op.getDialect()->getNamespace() == "afir") {
        afirOps.push_back(&op);
      } else if (auto candidate = dyn_cast<func::ReturnOp>(op)) {
        returnOp = candidate;
      }
    }
    return returnOp ? success() : failure();
  }

  static llvm::DenseMap<Value, unsigned> mapResultArguments(func::ReturnOp returnOp, unsigned inputCount) {
    llvm::DenseMap<Value, unsigned> resultArgIndex;
    for (auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
      resultArgIndex[value] = inputCount + index;
    }
    return resultArgIndex;
  }

  static void finalizeFunction(FunctionState &state) {
    state.builder.setInsertionPoint(state.returnOp);
    state.builder.create<func::ReturnOp>(state.func.getLoc());
    state.returnOp.erase();
    for (Operation *op : llvm::reverse(state.afirOps)) op->erase();
    BitVector droppedArguments(state.block.getNumArguments(), false);
    for (unsigned i = 0; i < state.originalArgumentCount; ++i) droppedArguments.set(i);
    state.block.eraseArguments(droppedArguments);
    state.func->setAttr("ascendc.aicore", UnitAttr::get(state.ctx));
    state.func->setAttr("ascendc.global", UnitAttr::get(state.ctx));
    state.func->setAttr("cann.num_inputs", IntegerAttr::get(IntegerType::get(state.ctx, 32), state.inputCount));
    state.func->removeAttr("afir.asc_graph");
  }

  static LogicalResult lowerTiledOperation(Operation *op, TiledOperationState &state) {
    Type elementType = cast<ShapedType>(op->getResult(0).getType()).getElementType();
    if (isa<afir::LoadOp>(op)) {
      auto argument = dyn_cast<BlockArgument>(op->getOperand(0));
      if (!argument || argument.getArgNumber() >= state.function.originalArgumentCount) {
        return op->emitError("queue lowering: load input must be a kernel input arg");
      }
      Value lowered = state.lowering.lowerLoadTiled(op->getLoc(), state.operationQueues[op],
                                                    state.function.gmArgument(argument.getArgNumber()), elementType,
                                                    state.actual, state.offset);
      state.function.mapping.map(op->getResult(0), lowered);
      state.tensorQueues[lowered] = state.operationQueues[op];
      return success();
    }
    if (l2Mnemonic(op) != StringRef()) {
      Value lhs = state.function.mapping.lookupOrDefault(op->getOperand(0));
      Value rhs = state.function.mapping.lookupOrDefault(op->getOperand(1));
      Value lowered =
          state.lowering.lowerBinaryTiled(op, state.operationQueues[op], lhs, rhs, elementType, state.actual);
      if (Value queue = state.tensorQueues.lookup(lhs)) state.lowering.freeTile(op->getLoc(), queue, lhs);
      if (Value queue = state.tensorQueues.lookup(rhs)) state.lowering.freeTile(op->getLoc(), queue, rhs);
      state.function.mapping.map(op->getResult(0), lowered);
      state.tensorQueues[lowered] = state.operationQueues[op];
      return success();
    }
    if (l2UnaryMnemonic(op) != StringRef()) {
      Value source = state.function.mapping.lookupOrDefault(op->getOperand(0));
      Value lowered = state.lowering.lowerUnaryTiled(op, state.operationQueues[op], source, elementType, state.actual);
      if (Value queue = state.tensorQueues.lookup(source)) state.lowering.freeTile(op->getLoc(), queue, source);
      state.function.mapping.map(op->getResult(0), lowered);
      state.tensorQueues[lowered] = state.operationQueues[op];
      return success();
    }
    if (!isa<afir::StoreOp>(op)) {
      return op->emitError("queue lowering (tiled): unsupported AFIR op '") << op->getName() << "'";
    }
    Value source = state.function.mapping.lookupOrDefault(op->getOperand(0));
    auto output = state.function.resultArgIndex.find(op->getResult(0));
    if (output == state.function.resultArgIndex.end()) {
      return op->emitError("queue lowering: store result must feed a kernel output");
    }
    state.lowering.lowerStoreTiled(op->getLoc(), source, state.function.gmArgument(output->second), elementType,
                                   state.actual, state.offset);
    if (Value queue = state.tensorQueues.lookup(source)) state.lowering.freeTile(op->getLoc(), queue, source);
    state.function.mapping.map(op->getResult(0), source);
    return success();
  }

  static LogicalResult lowerTiled(FunctionState &state) {
    Location loc = state.func.getLoc();
    unsigned outputCount = state.resultArgIndex.size();
    Value tiling = state.gmArgument(state.inputCount + outputCount + 1);
    Value tileLength = state.queueLowering.tilingField(loc, tiling, "tile_len");
    Value total = state.queueLowering.tilingField(loc, tiling, "total");
    Value tripCount = state.builder.create<arith::CeilDivSIOp>(loc, total, tileLength);
    Value blockIndex = state.builder.create<ascendc::GetBlockIdxOp>(loc, state.builder.getIndexType());
    Value blockCount = state.builder.create<ascendc::GetBlockNumOp>(loc, state.builder.getIndexType());

    llvm::DenseMap<Operation *, Value> operationQueues;
    for (Operation *op : state.afirOps) {
      if (isa<afir::StoreOp>(op)) continue;
      Type elementType = cast<ShapedType>(op->getResult(0).getType()).getElementType();
      operationQueues[op] =
          state.queueLowering.makeQueueTiled(loc, toTPosition(scheduledPosition(op)), tileLength, elementType);
    }

    auto loop = state.builder.create<scf::ForOp>(loc, blockIndex, tripCount, blockCount);
    OpBuilder loopBuilder(loop.getBody(), loop.getBody()->begin());
    QueueLowering loopLowering{loopBuilder, state.ctx, state.queueLowering.pipe, {}, {}};
    Value offset = loopBuilder.create<arith::MulIOp>(loc, loop.getInductionVar(), tileLength);
    Value remaining = loopBuilder.create<arith::SubIOp>(loc, total, offset);
    Value actual = loopBuilder.create<arith::MinSIOp>(loc, tileLength, remaining);
    llvm::DenseMap<Value, Value> tensorQueues;
    TiledOperationState operationState{state, loopLowering, operationQueues, tensorQueues, actual, offset};
    for (Operation *op : state.afirOps) {
      if (failed(lowerTiledOperation(op, operationState))) return failure();
    }
    return success();
  }

  static LogicalResult lowerUntiledOperation(Operation *op, FunctionState &state) {
    state.builder.setInsertionPoint(op);
    auto shaped = cast<ShapedType>(op->getResult(0).getType());
    ascendc::TPosition position = toTPosition(scheduledPosition(op));
    if (isa<afir::LoadOp>(op)) {
      auto argument = dyn_cast<BlockArgument>(op->getOperand(0));
      if (!argument || argument.getArgNumber() >= state.originalArgumentCount) {
        return op->emitError("queue lowering: load input must be a kernel input arg");
      }
      state.mapping.map(
          op->getResult(0),
          state.queueLowering.lowerLoad(op->getLoc(), state.gmArgument(argument.getArgNumber()), shaped, position));
      return success();
    }
    if (l2Mnemonic(op) != StringRef()) {
      Value lhs = state.mapping.lookupOrDefault(op->getOperand(0));
      Value rhs = state.mapping.lookupOrDefault(op->getOperand(1));
      state.mapping.map(op->getResult(0), state.queueLowering.lowerBinary(op, lhs, rhs, shaped, position));
      return success();
    }
    if (l2UnaryMnemonic(op) != StringRef()) {
      Value source = state.mapping.lookupOrDefault(op->getOperand(0));
      state.mapping.map(op->getResult(0), state.queueLowering.lowerUnary(op, source, shaped, position));
      return success();
    }
    if (isa<afir::BroadcastOp>(op)) {
      Value source = state.mapping.lookupOrDefault(op->getOperand(0));
      auto sourceType = cast<ShapedType>(op->getOperand(0).getType());
      state.mapping.map(op->getResult(0), state.queueLowering.lowerBroadcast(op, source, shaped, sourceType, position));
      return success();
    }
    if (!isa<afir::StoreOp>(op)) {
      return op->emitError("queue lowering: unsupported AFIR op '") << op->getName() << "'";
    }
    Value source = state.mapping.lookupOrDefault(op->getOperand(0));
    auto output = state.resultArgIndex.find(op->getResult(0));
    if (output == state.resultArgIndex.end()) {
      return op->emitError("queue lowering: store result must feed a kernel output");
    }
    state.queueLowering.lowerStore(op->getLoc(), source, state.gmArgument(output->second), shaped, {});
    state.mapping.map(op->getResult(0), source);
    return success();
  }

  static LogicalResult lowerUntiled(FunctionState &state) {
    for (Operation *op : state.afirOps) {
      if (failed(lowerUntiledOperation(op, state))) return failure();
    }
    return success();
  }

  LogicalResult lowerFunc(func::FuncOp func) {
    if (!containsAfir(func)) return success();
    MLIRContext *ctx = &getContext();
    auto &body = func.getBody();
    if (!llvm::hasSingleElement(body)) {
      return func.emitError("queue lowering expects a single-block func");
    }
    Block &block = body.front();
    unsigned inputCount = func.getNumArguments();
    SmallVector<Type> kernelArgumentTypes = buildKernelArgumentTypes(func, ctx);

    SmallVector<Operation *> afirOps;
    func::ReturnOp returnOp;
    if (failed(collectAfirOperations(block, afirOps, returnOp))) {
      return func.emitError("missing return");
    }
    llvm::DenseMap<Value, unsigned> resultArgIndex = mapResultArguments(returnOp, inputCount);

    func.setType(FunctionType::get(ctx, kernelArgumentTypes, {}));
    unsigned originalArgumentCount = block.getNumArguments();
    for (Type type : kernelArgumentTypes) block.addArgument(type, func.getLoc());

    OpBuilder builder(ctx);
    builder.setInsertionPoint(afirOps.empty() ? returnOp.getOperation() : afirOps.front());
    QueueLowering queueLowering{builder, ctx, {}, {}, {}};
    queueLowering.pipe = builder.create<ascendc::PipeOp>(func.getLoc());
    IRMapping mapping;
    FunctionState state{ctx,        func,    block,         afirOps, returnOp, resultArgIndex, originalArgumentCount,
                        inputCount, builder, queueLowering, mapping};

    bool tiled = llvm::any_of(afirOps, [](Operation *op) { return op->hasAttr("afir.sched_axis"); });
    LogicalResult lowered = tiled ? lowerTiled(state) : lowerUntiled(state);
    if (failed(lowered)) return failure();
    finalizeFunction(state);
    return success();
  }
};

}  // namespace

namespace mlir {
namespace afir {

std::unique_ptr<Pass> createConvertAFIRToAscendCQueuePass() {
  return std::make_unique<ConvertAFIRToAscendCQueuePass>();
}
void registerAFIRToAscendCQueuePass() {
  PassRegistration<ConvertAFIRToAscendCQueuePass>();
}

}  // namespace afir
}  // namespace mlir
