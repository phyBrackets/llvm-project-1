//===- SourceExprPass.cpp - Mapping Source Expression
//---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the mapping between LLVM Value and Source level
// expression, by utilizing the debug intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <llvm/ADT/DenseMapInfo.h>
#include <stack>

using namespace llvm;

#define DEBUG_TYPE "source_pass"

namespace {

llvm::DenseMap<Value *, std::string> llvmValtoLocalVarMap;

// This function translates LLVM opcodes to source-level expressions using DWARF
// operation encodings. It returns the source-level expression corresponding to
// the input opcode or "unknown" if the opcode is unsupported.

std::string getExpressionFromOpcode(StringRef opcode) {
  // Map LLVM opcodes to source-level expressions
  std::string expression;

  switch (llvm::StringSwitch<unsigned>(opcode)
              .Case("add", llvm::dwarf::DW_OP_plus)
              .Case("sub", llvm::dwarf::DW_OP_minus)
              .Case("mul", llvm::dwarf::DW_OP_mul)
              .Case("div", llvm::dwarf::DW_OP_div)
              .Case("shl", llvm::dwarf::DW_OP_shl)
              .Default(0)) {
  case llvm::dwarf::DW_OP_plus:
    expression = "+";
    break;
  case llvm::dwarf::DW_OP_minus:
    expression = "-";
    break;
  case llvm::dwarf::DW_OP_mul:
    expression = "*";
    break;
  case llvm::dwarf::DW_OP_div:
    expression = "/";
    break;
  case llvm::dwarf::DW_OP_shl:
    expression = "<<";
    break;
  default:
    // Handle unknown opcodes or unsupported operations
    expression = "unknown";
    break;
  }

  return expression;
}

llvm::DenseMap<llvm::Value *, llvm::SmallVector<std::string>>
    sourceExpressionsMap;

// This function generates the source-level expression for a given operand by
// recursively traversing the operand's instructions. If the operand is a binary
// operator, it constructs the expression using the names of its operands and
// the operator symbol. If the operand is not an instruction or doesn't match
// any specific case, it returns its name or operand representation.
std::string getSourceExpressionForOperand(
    llvm::Value *operand,
    llvm::DenseMap<llvm::Value *, std::string> sourceExpressionsMap,
    StringRef symbol) {
  if (llvm::Instruction *operandInst =
          llvm::dyn_cast<llvm::Instruction>(operand)) {
    if (llvm::BinaryOperator *binaryOp =
            llvm::dyn_cast<llvm::BinaryOperator>(operandInst)) {
      // Binary operator - build source expression using two operands
      llvm::Value *operand1 = binaryOp->getOperand(0);
      llvm::Value *operand2 = binaryOp->getOperand(1);

      // Check if source expressions exist fo r the operands
      std::string name1 = llvmValtoLocalVarMap.count(operand1)
                              ? llvmValtoLocalVarMap[operand1]
                              : getSourceExpressionForOperand(
                                    operand1, llvmValtoLocalVarMap, symbol);
      std::string name2 = llvmValtoLocalVarMap.count(operand2)
                              ? llvmValtoLocalVarMap[operand2]
                              : getSourceExpressionForOperand(
                                    operand2, llvmValtoLocalVarMap, symbol);
      return "(" + name1 + " " +
             getExpressionFromOpcode(binaryOp->getOpcodeName()) + " " + name2 +
             ")";
    } else if (llvm::LoadInst *loadInst =
                   llvm::dyn_cast<llvm::LoadInst>(operandInst)) {
      llvm::Value *operandVal = loadInst->getOperand(0);
      std::string operandName =
          sourceExpressionsMap.count(operandVal)
              ? sourceExpressionsMap[operandVal]
              : getSourceExpressionForOperand(operandVal, sourceExpressionsMap,
                                              symbol);
      return "(" + operandName + ")";
    } else if (llvm::StoreInst *storeInst =
                   llvm::dyn_cast<llvm::StoreInst>(operandInst)) {
      // Store instruction - return the source expression for its value operand
      llvm::Value *operandVal = storeInst->getValueOperand();

      // Check if a source expression exists for the value operand
      std::string operandName =
          llvmValtoLocalVarMap.count(operandVal)
              ? llvmValtoLocalVarMap[operandVal]
              : getSourceExpressionForOperand(operandVal, llvmValtoLocalVarMap,
                                              symbol);
      // Construct the source expression using the operand name and the symbol
      return "(" + operandName + ")";
    }
  }
  return operand->getNameOrAsOperand();
}

// This function extracts the derived type information from a metadata node.
// If the node does not match the expected format, it returns "unknown".

std::string getDerivedType(const llvm::MDNode *node) {
  if (auto derivedType = llvm::dyn_cast<llvm::DIDerivedType>(node)) {
    if (derivedType->getTag() == llvm::dwarf::DW_TAG_pointer_type) {
      if (auto baseType =
              llvm::dyn_cast<llvm::DIBasicType>(derivedType->getBaseType())) {
        return baseType->getName().str() + "*";
      }
    }
  }

  return "unknown";
}

// These functions generates a source expression for a variable with a basic
// type and derived type. It retrieves the value of the LLVM instruction and
// formats it into a string representation. Then, based on whether the variable
// is a parameter or not, it constructs a source expression string including the
// variable's type, name, and assigned value. The resulting source expression is
// returned.

std::string getSourceExpressionForBasicType(llvm::DILocalVariable *localVar,
                                            llvm::Value *llvmVal,
                                            StringRef varName) {
  std::string value;
  value = llvmVal->getNameOrAsOperand();
  std::string SourceExpression;
  if (localVar->isParameter()) {
    SourceExpression = "arg " + std::to_string(localVar->getArg()) + ": " +
                       localVar->getType()->getName().str() + " " +
                       varName.str() + " = " + value + "\n";
  } else {
    SourceExpression = localVar->getType()->getName().str() + " " +
                       varName.str() + " = " + value + "\n";
  }

  return SourceExpression;
}

std::string getSourceExpressionForDerivedType(llvm::DILocalVariable *localVar,
                                              llvm::Value *llvmVal,
                                              StringRef varName) {
  std::string derivedType = getDerivedType(localVar->getType());

  std::string value;
  // llvmValtoLocalVarMap[llvmVal] = varName.str();
  value = llvmVal->getNameOrAsOperand();
  std::string SourceExpression;
  if (localVar->isParameter()) {
    SourceExpression = "arg " + std::to_string(localVar->getArg()) + ": " +
                       derivedType + " " + varName.str() + " = " +
                       varName.str() + "\n";
  } else {
    SourceExpression = derivedType + " " + value + " = " + value + "\n";
  }

  return SourceExpression;
}

// Helper function to get the source expression for a variable based on its type

std::string getSourceExpressionForVariable(llvm::DILocalVariable *localVar,
                                           llvm::Value *value,
                                           StringRef varName) {
  std::string sourceExpression;
  DIType *localVarType = localVar->getType();

  llvmValtoLocalVarMap[value] = varName.str();

  if (auto *basicType = dyn_cast<DIBasicType>(localVarType)) {
    // Basic type case
    sourceExpression =
        getSourceExpressionForBasicType(localVar, value, varName);
  } else {
    // Derived type case
    sourceExpression =
        getSourceExpressionForDerivedType(localVar, value, varName);
  }

  return sourceExpression;
}

llvm::SmallVector<std::string> getSourceExpressionsForValue(
    llvm::Value *llvmVal,
    llvm::DenseMap<llvm::Value *, llvm::SmallVector<std::string>>
        &sourceExpressionsMap,
    StringRef symbol, llvm::DILocalVariable *localVar = nullptr) {
  llvm::SmallVector<std::string> sourceExpressions;
  llvm::DenseMap<llvm::LoadInst *, llvm::StoreInst *> loadStoreMap;

  for (llvm::User *user : llvmVal->users()) {
    if (llvm::LoadInst *loadInst = llvm::dyn_cast<llvm::LoadInst>(user)) {

      if (loadStoreMap.count(loadInst) > 0) {
        std::string sourceExpression;
        llvm::Value *loadedValue = loadInst->getPointerOperand();
        sourceExpression = getSourceExpressionForVariable(localVar, loadedValue,
                                                          localVar->getName());
        sourceExpressions.push_back(sourceExpression);
        sourceExpressionsMap[loadInst] = sourceExpressions;
        // Load instruction already processed inside the store, skip it
        continue;
      }

      llvm::SmallVector<std::string> expressions = getSourceExpressionsForValue(
          loadInst, sourceExpressionsMap, symbol, localVar);
      sourceExpressions.insert(sourceExpressions.end(), expressions.begin(),
                               expressions.end());
      std::string sourceExpression = getSourceExpressionForVariable(
          localVar, loadInst, localVar->getName());
      sourceExpressions.push_back(sourceExpression);
      sourceExpressionsMap[loadInst] = sourceExpressions;
    } else if (llvm::StoreInst *storeInst =
                   llvm::dyn_cast<llvm::StoreInst>(user)) {
      llvm::Value *storedValue = storeInst->getValueOperand();
      std::string sourceExpression;
      // llvm::Value* loadedValue;
      if (llvm::LoadInst *loadInst =
              llvm::dyn_cast<llvm::LoadInst>(storedValue)) {
        llvm::Value *loadedValue = loadInst->getPointerOperand();

        if (llvm::GetElementPtrInst *gepInst =
                llvm::dyn_cast<llvm::GetElementPtrInst>(loadInst)) {
          continue; // Skip GEP instructions for now
        } else {
          sourceExpression = getSourceExpressionForVariable(
              localVar, loadedValue, localVar->getName());
          sourceExpressions.push_back(sourceExpression);
          sourceExpressionsMap[loadInst] = sourceExpressions;

          //      continue;
        }
      } else {

        sourceExpression = getSourceExpressionForVariable(localVar, storedValue,
                                                          localVar->getName());
        sourceExpressions.push_back(sourceExpression);
        sourceExpressionsMap[storeInst] = sourceExpressions;
      }
    }
  }

  return sourceExpressions;
}

std::string processDbgDeclareInst(
    llvm::DbgDeclareInst *dbgDeclare,
    llvm::DenseMap<llvm::Value *, llvm::SmallVector<std::string>>
        &sourceExpressionsMap,
    StringRef symbol) {
  llvm::Value *llvmVal = cast<llvm::Value>(dbgDeclare->getAddress());

  llvm::DILocalVariable *localVar = dbgDeclare->getVariable();
  std::string varName = localVar->getName().str();

  llvm::SmallVector<std::string> sourceExpressions;
  std::string sourceExpression;
  llvm::SmallVector<std::string> expressions;

  // sourceExpression = getSourceExpressionForVariable(localVar, llvmVal,
  // varName);
  expressions = getSourceExpressionsForValue(llvmVal, sourceExpressionsMap,
                                             symbol, localVar);

  sourceExpressions.insert(sourceExpressions.end(), expressions.begin(),
                           expressions.end());
  // sourceExpressions.push_back(sourceExpression);

  for (std::string expression : sourceExpressions) {
    llvm::dbgs() << "Source-level expression is : " << expression << "\n";
  }

  return "";
}

void processDbgValueInst(
    llvm::DbgValueInst *dbgVal,
    llvm::DenseMap<llvm::Value *, llvm::SmallVector<std::string>>
        &sourceExpressionsMap,
    StringRef symbol) {
  llvm::Value *llvmVal = dbgVal->getValue();

  llvm::DILocalVariable *localVar = dbgVal->getVariable();
  llvm::DIExpression *expr = dbgVal->getExpression();
  std::string varName = localVar->getName().str();
  llvm::SmallVector<std::string> sourceExpressions;

  if (!expr->getNumElements()) {
    std::string sourceExpression =
        getSourceExpressionForVariable(localVar, llvmVal, varName);
    llvm::SmallVector<std::string> expressions = getSourceExpressionsForValue(
        llvmVal, sourceExpressionsMap, symbol, localVar);
    sourceExpressions.insert(sourceExpressions.end(), expressions.begin(),
                             expressions.end());
    sourceExpressions.push_back(sourceExpression);
  } else {
    std::string exprStr;
    llvm::raw_string_ostream exprStream(exprStr);
    llvmVal->printAsOperand(exprStream, false /* PrintType */);

    std::stack<uint64_t> exprStack;

    for (int i = 0; i < expr->getNumElements(); ++i) {
      uint64_t element = expr->getElement(i);
      exprStack.push(element);
    }

    while (!exprStack.empty()) {
      uint64_t element = exprStack.top();
      uint64_t store;

      if (element == llvm::dwarf::DW_OP_constu) {
        if (!exprStack.empty()) {
          store = exprStack.top();
          exprStream << store;
        }
        exprStack.pop();
      } else if (element == llvm::dwarf::DW_OP_mul) {
        exprStream << " * ";
        exprStack.pop();
      } else if (element == llvm::dwarf::DW_OP_minus) {
        exprStream << " - ";
        exprStack.pop();
      } else if (element == llvm::dwarf::DW_OP_stack_value) {
        exprStack.pop();
      } else {
        store = element;
        exprStack.pop();
      }
    }

    std::string expr = varName + " = " + exprStream.str();
    sourceExpressions.push_back(expr);
  }

  for (std::string expression : sourceExpressions) {
    llvm::dbgs() << "Source-level expression is: " << expression << "\n";
  }
  // return "";
}

// This function builds source-level expressions for LLVM instructions related
// to debug information. It handles two cases: DbgValueInst and DbgDeclareInst.
// For DbgValueInst, it retrieves the value, local variable, and expression
// associated with the instruction. Based on the expression, it generates
// source-level expressions either by processing the expression elements or by
// constructing expressions using the instructions and it's operand. For
// DbgDeclareInst, it retrieves the address, local variable, and source
// expressions associated with StoreInst instructions using the address.

std::string buildSourceLevelExpressionFromIntrinsic(llvm::Instruction &I,
                                                    std::string symbol) {
  llvm::SmallVector<std::string> sourceExpressions;
  if (auto *dbgVal = llvm::dyn_cast<llvm::DbgValueInst>(&I)) {
    dbgs() << I << "\n";
    processDbgValueInst(dbgVal, sourceExpressionsMap, symbol);

  } else if (auto *dbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
    dbgs() << I << "\n";
    processDbgDeclareInst(dbgDeclare, sourceExpressionsMap, symbol);
  }

  return "";
}

void processGEPInstructions(
    llvm::DenseMap<llvm::Value *, std::string> &llvmValtoLocalVarMap,
    llvm::DenseMap<llvm::Value *, llvm::SmallVector<std::string>>
        &sourceExpressionsMap) {
  for (const auto &entry : llvmValtoLocalVarMap) {

    llvm::Value *llvmVal = entry.first;
    auto localVar = entry.second;
    for (llvm::User *user : llvmVal->users()) {
      if (llvm::GetElementPtrInst *gepInst =
              llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {

        llvm::Value *basePointer = gepInst->getOperand(0);
        llvm::Value *offset = gepInst->getOperand(gepInst->getNumIndices());

        std::string basePointerName =
            llvmValtoLocalVarMap.count(basePointer)
                ? llvmValtoLocalVarMap[basePointer]
                : getSourceExpressionForOperand(
                      basePointer, llvmValtoLocalVarMap,
                      getExpressionFromOpcode(llvm::Instruction::getOpcodeName(
                          gepInst->getOpcode())));
        std::string offsetName =
            llvmValtoLocalVarMap.count(offset)
                ? llvmValtoLocalVarMap[offset]
                : getSourceExpressionForOperand(
                      offset, llvmValtoLocalVarMap,
                      getExpressionFromOpcode(llvm::Instruction::getOpcodeName(
                          gepInst->getOpcode())));

        // Construct the source expression for the address computation
        std::string expression = "&" + basePointerName + "[" + offsetName + "]";
        llvm::SmallVector<std::string> sourceExpression;
        sourceExpression.push_back(expression);
        // Store the source expression for the current instruction in
        // sourceExpressionsMap
        sourceExpressionsMap[gepInst] = sourceExpression;

        // Process the users of the GEP instruction recursively
        //  std::vector<std::string> expressions =
        //    getSourceExpressionsForValue(gepInst, sourceExpressionsMap,
        //    getExpressionFromOpcode(llvm::Instruction::getOpcodeName(gepInst->getOpcode())));
        // sourceExpressions.insert(sourceExpressions.end(),
        // expressions.begin(),
        //               expressions.end());
      }
    }
  }
}

// This method implements what the pass does
void visitor(Function &F) {
  errs() << "(llvm-tutor) Hello from: " << F.getName() << "\n";
  errs() << "(llvm-tutor)   number of arguments: " << F.arg_size() << "\n";
  std::string SourceExpression = "null";
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      //     errs() << I << "foo";
      std::string operation = llvm::Instruction::getOpcodeName(I.getOpcode());
      std::string symbol = getExpressionFromOpcode(operation);

      std::string srcExx = buildSourceLevelExpressionFromIntrinsic(I, symbol);
    }
  }

  processGEPInstructions(llvmValtoLocalVarMap, sourceExpressionsMap);

  for (const auto &pair : sourceExpressionsMap) {
    llvm::Value *value = pair.first;
    llvm::SmallVector<std::string> sourceExprs = pair.second;

    llvm::outs() << "Value: " << value->getNameOrAsOperand() << "\n";
    llvm::outs() << "Source Expressions:\n";
    for (std::string sourceExpr : sourceExprs) {
      llvm::outs() << "  " << sourceExpr << "\n";
    }
    llvm::outs() << "--------------------------\n";
  }
}

// New PM implementation
struct SourceExpressionPass : PassInfoMixin<SourceExpressionPass> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    visitor(F);
    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

} // namespace

llvm::PassPluginLibraryInfo getSourceExprPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SourceExprMap", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "source-expr") {
                    FPM.addPass(SourceExpressionPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSourceExprPluginInfo();
}