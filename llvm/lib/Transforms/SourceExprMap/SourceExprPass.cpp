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
#include <stack>

using namespace llvm;

#define DEBUG_TYPE "source_pass"

namespace {

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

llvm::DenseMap<llvm::Value *, std::string> sourceExpressionsMap;

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

      // Check if source expressions exist for the operands
      std::string name1 = sourceExpressionsMap.count(operand1)
                              ? sourceExpressionsMap[operand1]
                              : getSourceExpressionForOperand(
                                    operand1, sourceExpressionsMap, symbol);
      std::string name2 = sourceExpressionsMap.count(operand2)
                              ? sourceExpressionsMap[operand2]
                              : getSourceExpressionForOperand(
                                    operand2, sourceExpressionsMap, symbol);

      // Construct the source expression using variable names and the symbol
      return "(" + name1 + " " +
             getExpressionFromOpcode(binaryOp->getOpcodeName()) + " " + name2 +
             ")";
    }
  }

  // If the operand is not an instruction or doesn't match any specific case,
  // return its name or operand representation
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
  value = llvmVal->getNameOrAsOperand();
  std::string SourceExpression;
  if (localVar->isParameter()) {
    SourceExpression = "arg " + std::to_string(localVar->getArg()) + ": " +
                       derivedType + " " + varName.str() + " = " + value + "\n";
  } else {
    SourceExpression = derivedType + " " + varName.str() + " = " + value + "\n";
  }

  return SourceExpression;
}

// Helper function to get the source expression for a variable based on its type

std::string getSourceExpressionForVariable(llvm::DILocalVariable *localVar,
                                           llvm::Value *value,
                                           StringRef varName) {
  std::string sourceExpression;
  DIType *localVarType = localVar->getType();

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

// This function builds source-level expressions for LLVM instructions related
// to debug information. It handles two cases: DbgValueInst and DbgDeclareInst.
// For DbgValueInst, it retrieves the value, local variable, and expression
// associated with the instruction. Based on the expression, it generates
// source-level expressions either by processing the expression elements or by
// constructing expressions using the instructions and it's operand. For
// DbgDeclareInst, it retrieves the address, local variable, and source
// expressions associated with StoreInst instructions using the address.

std::string buildSourceLevelExpressionFromIntrinsic(llvm::Instruction &I,
                                                    StringRef symbol) {
  if (auto *dbgVal = llvm::dyn_cast<llvm::DbgValueInst>(&I)) {
    llvm::dbgs() << I << "\n";
    llvm::Value *llvmVal = dbgVal->getValue();
    llvm::DILocalVariable *localVar = dbgVal->getVariable();
    llvm::DIExpression *expr = dbgVal->getExpression();
    std::vector<std::string> sourceExpressions;
    std::string varName = localVar->getName().str();

    if (!expr->getNumElements()) {
      std::string sourceExpression;

      sourceExpression = // getTypeExpression(localVarType);
          getSourceExpressionForVariable(localVar, llvmVal, varName);

      for (User *user : llvmVal->users()) {
        if (llvm::GetElementPtrInst *gepInst =
                llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {

          llvm::Value *basePointer = gepInst->getOperand(0);
          llvm::Value *offset = gepInst->getOperand(gepInst->getNumIndices());

          std::string basePointerName =
              sourceExpressionsMap.count(basePointer)
                  ? sourceExpressionsMap[basePointer]
                  : getSourceExpressionForOperand(basePointer,
                                                  sourceExpressionsMap, symbol);
          std::string offsetName =
              sourceExpressionsMap.count(offset)
                  ? sourceExpressionsMap[offset]
                  : getSourceExpressionForOperand(offset, sourceExpressionsMap,
                                                  symbol);

          // Construct the source expression for the address computation
          std::string expression =
              "&" + basePointerName + "[" + offsetName + "]";

          // Store the source expression for the current instruction
          sourceExpressionsMap[&I] = expression;
          sourceExpressions.push_back(expression);
        }
      }

      sourceExpressions.push_back(sourceExpression);
    }

    else {
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
      //   dbgs() << "Source-level expression is " << SourceExpression << " = "
      //   << expr << "\n";
      // return "";
    }

    for (StringRef expression : sourceExpressions) {
      dbgs() << "Source-level expression is: " << expression.str() << "\n";
    }

    return "";

  } else if (auto *dbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
    dbgs() << I << "\n";
    // errs() << "Decl";
    Value *llvmVal = cast<Value>(dbgDeclare->getAddress());
    std::string sourceExpression;
    DILocalVariable *localVar = dbgDeclare->getVariable();
    std::string varName = localVar->getName().str();
    AllocaInst *allocaInst = cast<AllocaInst>(llvmVal);
    std::vector<std::string> sourceExpressions;
    auto localVarType = localVar->getType();

    for (User *user : allocaInst->users()) {
      if (auto *storeInst = dyn_cast<StoreInst>(user)) {
        Value *storedValue = storeInst->getValueOperand();
        std::string sourceExpression;

        if (auto *loadInst = dyn_cast<LoadInst>(storedValue)) {
          Value *loadedValue = loadInst->getPointerOperand();
          sourceExpression =
              getSourceExpressionForVariable(localVar, loadedValue, varName);
        } else {
          sourceExpression =
              getSourceExpressionForVariable(localVar, storedValue, varName);
        }

        sourceExpressions.push_back(sourceExpression);
      }
    }

    for (StringRef expression : sourceExpressions) {
      dbgs() << "Source-level expression is: " << expression.str() << "\n";
    }
    return "";
  }
  return "";
}

std::string
buildSourceLevelExpressionWithRawInstruction(llvm::Instruction &I,
                                             const std::string &symbol) {
  std::string sourceExpression = "nullptr";

  if (llvm::BinaryOperator *binaryOp =
          llvm::dyn_cast<llvm::BinaryOperator>(&I)) {
    // Binary operator - build source expression using two operands
    llvm::Value *operand1 = binaryOp->getOperand(0);
    llvm::Value *operand2 = binaryOp->getOperand(1);

    // Check if source expressions exist for the operands
    std::string name1 = sourceExpressionsMap.count(operand1)
                            ? sourceExpressionsMap[operand1]
                            : operand1->getNameOrAsOperand();
    std::string name2 = sourceExpressionsMap.count(operand2)
                            ? sourceExpressionsMap[operand2]
                            : operand2->getNameOrAsOperand();

    // Construct the source expression using variable names and the symbol
    sourceExpression = "(" + name1 + " " + symbol + " " + name2 + ")";

    // Store the source expression for the current instruction
    sourceExpressionsMap[&I] = sourceExpression;
  } else if (llvm::GetElementPtrInst *gepInst =
                 llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {

    llvm::Value *basePointer = gepInst->getOperand(0);
    llvm::Value *offset = gepInst->getOperand(gepInst->getNumIndices());

    std::string basePointerName = sourceExpressionsMap.count(basePointer)
                                      ? sourceExpressionsMap[basePointer]
                                      : basePointer->getNameOrAsOperand();
    std::string offsetName = sourceExpressionsMap.count(offset)
                                 ? sourceExpressionsMap[offset]
                                 : offset->getNameOrAsOperand();

    // Construct the source expression for the address computation
    sourceExpression = "&" + basePointerName + "[" + offsetName + "]";

    // Store the source expression for the current instruction
    sourceExpressionsMap[&I] = sourceExpression;
  } else if (llvm::LoadInst *loadInst = llvm::dyn_cast<llvm::LoadInst>(&I)) {
    llvm::Value *address = loadInst->getOperand(0);

    // Check if source expressions exist for the address operand
    std::string addressName = sourceExpressionsMap.count(address)
                                  ? sourceExpressionsMap[address]
                                  : address->getNameOrAsOperand();

    // Construct the source expression for accessing the value
    sourceExpression = "*(" + addressName + ")";

    // Store the source expression for the current instruction
    sourceExpressionsMap[&I] = sourceExpression;
  } else if (llvm::StoreInst *storeInst = llvm::dyn_cast<llvm::StoreInst>(&I)) {
    llvm::Value *value = storeInst->getValueOperand();
    llvm::Value *address = storeInst->getPointerOperand();

    std::string valueName = sourceExpressionsMap.count(value)
                                ? sourceExpressionsMap[value]
                                : value->getNameOrAsOperand();
    std::string addressName = sourceExpressionsMap.count(address)
                                  ? sourceExpressionsMap[address]
                                  : address->getNameOrAsOperand();

    sourceExpression = /*"*" +*/ addressName + " = " + valueName;
  } else if (llvm::UnaryOperator *unaryOp =
                 llvm::dyn_cast<llvm::UnaryOperator>(&I)) {
    dbgs() << "gelement";
    llvm::Value *operand = unaryOp->getOperand(0);

    std::string name = operand->getNameOrAsOperand();

    // Construct the source expression using the variable name and the symbol
    sourceExpression = name + symbol;
  }

  return sourceExpression;
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