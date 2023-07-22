//===- SourceExpressionAnalysis.cpp - Mapping Source Expression
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

#include "llvm/Analysis/SourceExpressionAnalysis.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include <unordered_map>
using namespace llvm;

#define DEBUG_TYPE "source_expr"

// This function translates LLVM opcodes to source-level expressions using DWARF
// operation encodings. It returns the source-level expression corresponding to
// the input opcode or "unknown" if the opcode is unsupported.

std::string
LoadStoreSourceExpression::getExpressionFromOpcode(unsigned opcode) {
  // Map LLVM opcodes to source-level expressions

  switch (opcode) {
  case Instruction::Add:
    return "+";
    break;
  case Instruction::Sub:
    return "-";
    break;
  case Instruction::Mul:
    return "*";
    break;
  case Instruction::UDiv:
    return "/";
    break;
  case Instruction::Shl:
    return "<<";
    break;
  default:
    // Handle unknown opcodes or unsupported operations
    return "unknown";
    break;
  }
}

// Function to remove the '&' character from a string

std::string LoadStoreSourceExpression::removeAmpersand(StringRef AddrStr) {
  std::string result = AddrStr.str();

  size_t found = result.find('&');
  if (found != std::string::npos) {
    result.erase(found, 1);
  }
  return result;
}

/**
 * Process the debug metadata for the given stored value. This function
 * retrieves the corresponding debug values (DbgValueInst) and debug declare
 * instructions (DbgDeclareInst) associated with the stored value. If a
 * DbgDeclareInst is found, the associated DILocalVariable is retrieved and
 * returned. If a DbgValueInst is found, the associated DILocalVariable is
 * retrieved and the source expression is stored in the 'sourceExpressionsMap'
 * for the stored value. This function is used to extract debug information for
 * the source expressions.
 *
 * @param storedValue The stored value to process.
 * @return The DILocalVariable associated with the stored value, or nullptr if
 * no debug metadata is found.
 */

DILocalVariable *
LoadStoreSourceExpression::processDbgMetadata(Value *storedValue) {
  if (storedValue->isUsedByMetadata()) {
    // Find the corresponding DbgValues and DbgDeclareInsts
    SmallVector<DbgValueInst *, 8> DbgValues;
    findDbgValues(DbgValues, storedValue);

    TinyPtrVector<DbgDeclareInst *> dbgDeclareInsts =
        FindDbgDeclareUses(storedValue);

    if (!dbgDeclareInsts.empty()) {
      assert(dbgDeclareInsts.size() == 1);

      // Handle the case where DbgDeclareInst is found
      DbgDeclareInst *dbgDeclareInst = dbgDeclareInsts[0];
      DILocalVariable *localVar = dbgDeclareInst->getVariable();
      sourceExpressionsMap[storedValue] = localVar->getName().str();
      return localVar;
    } else if (!DbgValues.empty()) {
      assert(DbgValues.size() == 1);

      // Handle the case where DbgValueInst is found
      DbgValueInst *dbgValueInst = DbgValues[0];
      DILocalVariable *localVar = dbgValueInst->getVariable();
      sourceExpressionsMap[storedValue] = localVar->getName().str();
      return localVar;
    }
  }

  return nullptr;
}

/**
 * Process the given DIType.
 * This function recursively traverses the type hierarchy to handle basic types,
 * derived types, and composite types (such as structures and arrays). For basic
 * types, information about the member is stored in the memberInfo map. For
 * derived types, the `processDIType` function is called recursively with the
 * base type. For structures, each element is processed recursively using the
 * `processDIType` function. For arrays, the base type (which is a structure) is
 * processed recursively with the same member name.
 *
 * @param diType The DIType to process.
 * @param basePointer The base pointer value associated with the DIType.
 * @param memberName The name of the member, if applicable.
 */

void LoadStoreSourceExpression::processDIType(DIType *diType,
                                              Value *basePointer,
                                              std::string memberName) {
  // Check if the DIType is valid
  if (!diType) {
    return;
  }

  // Process basic type associated with the type
  if (auto *basicType = dyn_cast<DIBasicType>(diType)) {
    // Store information about the member
    memberInfo[basePointer->getName()].push_back(
        {basicType->getName().str(), memberName});
  }
  // Process derived type associated with the type
  else if (auto *derivedType = dyn_cast<DIDerivedType>(diType)) {
    std::string derivedMemberName = derivedType->getName().str();
    auto *type = derivedType->getBaseType();

    // Recursively process the nested type
    processDIType(type, basePointer, derivedMemberName);
  }
  // Process composite types (structures, arrays, etc.)
  else if (auto *compositeType = dyn_cast<DICompositeType>(diType)) {
    // Check if the composite type is a structure
    if (compositeType->getTag() == dwarf::DW_TAG_structure_type) {
      auto nodeArray = compositeType->getElements();
      if (nodeArray) {
        // Iterate over the elements of the structure
        for (unsigned i = 0; i < nodeArray.size(); ++i) {
          Metadata *metadata = nodeArray[i];
          DIType *nestedDINode = dyn_cast<DIType>(metadata);
          processDIType(nestedDINode, basePointer, memberName);
        }
      }
    }
    // Check if the composite type is an array
    else if (compositeType->getTag() == dwarf::DW_TAG_array_type) {
      auto *baseType = compositeType->getBaseType();
      // Recursively process the base type (which is a structure)
      processDIType(baseType, basePointer, memberName);
    }
  }
}

/**
 * Get the source-level expression for an LLVM value.
 *
 * @param operand The LLVM value to generate the source-level expression for.
 * @param symbol The symbol associated with the value, if applicable.
 * @return The source-level expression for the value.
 */

std::string LoadStoreSourceExpression::getSourceExpression(Value *operand,
                                                           StringRef symbol) {
  // Check if the operand has debug metadata associated with it
  if (!isa<ConstantInt>(operand)) {
    DILocalVariable *localVar = processDbgMetadata(operand);
    if (localVar) {
      return localVar->getName().str();
    }
  }

  if (GetElementPtrInst *gepInst = dyn_cast<GetElementPtrInst>(operand)) {
    return getSourceExpressionForGetElementPtr(gepInst);
  } else if (BinaryOperator *binaryOp = dyn_cast<BinaryOperator>(operand)) {
    return getSourceExpressionForBinaryOperator(binaryOp, operand);
  } else if (LoadInst *loadInst = dyn_cast<LoadInst>(operand)) {
    return getSourceExpressionForLoadInst(loadInst);
  } else if (StoreInst *storeInst = dyn_cast<StoreInst>(operand)) {
    return getSourceExpressionForStoreInst(storeInst);
  } else if (SExtInst *sextInst = dyn_cast<SExtInst>(operand)) {
    return getSourceExpressionForSExtInst(sextInst);
  }

  // If no specific case matches, return the name of the operand or its
  // representation
  return operand->getNameOrAsOperand();
}

// Get the type tag from the given DIType
// Returns:
//   0: If the DIType is null or the type tag is unknown or unsupported
//   DW_TAG_base_type, DW_TAG_pointer_type, DW_TAG_const_type, etc.: The type
//   tag

static uint16_t getTypeTag(DIType *diType) {
  if (!diType)
    return 0;

  if (auto *basicType = dyn_cast<DIBasicType>(diType)) {
    return basicType->getTag();
  } else if (auto *derivedType = dyn_cast<DIDerivedType>(diType)) {
    return derivedType->getTag();
  } else if (auto *compositeType = dyn_cast<DICompositeType>(diType)) {
    return compositeType->getTag();
  }

  // Return 0 for unknown or unsupported type tags
  return 0;
}

/**
 * Get the source-level expression for a GetElementPtr instruction.
 *
 * @param gepInst The GetElementPtr instruction.
 * @return The source-level expression for the address computation.
 */

std::string LoadStoreSourceExpression::getSourceExpressionForGetElementPtr(
    GetElementPtrInst *gepInst) {
  // GetElementPtr instruction - construct source expression for address
  // computation

  Value *basePointer = gepInst->getOperand(0);
  Value *offset = gepInst->getOperand(gepInst->getNumIndices());

  int offsetVal = 0;
  if (ConstantInt *offsetConstant = dyn_cast<ConstantInt>(offset)) {
    // Retrieve the value of the constant integer as an integer
    offsetVal = offsetConstant->getSExtValue();
  }

  DILocalVariable *localVar = processDbgMetadata(basePointer);
  DIType *type = localVar ? localVar->getType() : nullptr;
  processDIType(type, basePointer);

  std::string basePointerName =
      sourceExpressionsMap.count(basePointer)
          ? sourceExpressionsMap[basePointer]
          : getSourceExpression(basePointer, getExpressionFromOpcode(

                                                 gepInst->getOpcode()));
  std::string offsetName =
      sourceExpressionsMap.count(offset)
          ? sourceExpressionsMap[offset]
          : getSourceExpression(offset,
                                getExpressionFromOpcode(gepInst->getOpcode()));

  SmallString<32> expression;
  raw_svector_ostream OS(expression);

  uint16_t tag = getTypeTag(type);
  if (tag == dwarf::DW_TAG_structure_type) {
    // It's a struct type
    OS << basePointerName << "."
       << memberInfo[basePointer->getName()].at(offsetVal).second;
  } else if (tag == dwarf::DW_TAG_array_type ||
             isa<PointerType>(basePointer->getType())) {
    if (basePointerName.find('[') == std::string::npos) {
      // Construct the source expression for the address computation with
      // square brackets
      OS << "&" << basePointerName << "[" << offsetName << "]";
    } else if (basePointerName.find('[') != std::string::npos &&
               basePointerName.find('&') != std::string::npos) {
      size_t found = basePointerName.find('&');
      if (found != std::string::npos) {
        basePointerName.erase(found, 1);
      }
      // If basePointerName already contains square brackets, combine it
      // with offsetName directly
      OS << basePointerName << "[" << offsetName << "]";
    } else if (basePointerName.find('[') != std::string::npos &&
               !checkArrayType[basePointer->getNameOrAsOperand()]) {
      // If basePointerName already contains square brackets, combine it
      // with offsetName directly
      OS << basePointerName << " + " << offsetName;
    }
  }

  sourceExpressionsMap[gepInst] = expression.str().str();

  // Return the constructed source expression
  return expression.str().str();
}

/**
 * Get the source-level expression for a binary operator instruction.
 *
 * @param binaryOp The binary operator instruction.
 * @param operand The operand associated with the instruction.
 * @return The source-level expression for the binary operation.
 */

std::string LoadStoreSourceExpression::getSourceExpressionForBinaryOperator(
    BinaryOperator *binaryOp, Value *operand) {
  // Binary operator - build source expression using two operands

  Value *operand1 = binaryOp->getOperand(0);
  Value *operand2 = binaryOp->getOperand(1);

  std::string name1 = sourceExpressionsMap.count(operand1)
                          ? sourceExpressionsMap[operand1]
                          : getSourceExpression(operand1);
  std::string name2 = sourceExpressionsMap.count(operand2)
                          ? sourceExpressionsMap[operand2]
                          : getSourceExpression(operand2);
  std::string opcode = binaryOp->getOpcodeName();

  SmallString<32> expression;
  raw_svector_ostream OS(expression);
  if (opcode == "+") {
    if (ConstantInt *constantInt = dyn_cast<ConstantInt>(operand2)) {
      if (constantInt->isNegative()) {
        // Modify the expression for addition with a negative value
        APInt absValue = constantInt->getValue().abs();
        SmallVector<char, 16> str;
        absValue.toString(str, 10, false);
        std::string absValueStr(str.begin(), str.end());
        OS << "(" << name1 << " - " << absValueStr << ")";
      }
    }
  }

  if (expression.empty()) {
    // If no modification is needed, use the original expression generation
    OS << "(" << name1 << " " << getExpressionFromOpcode(binaryOp->getOpcode())
       << " " << name2 << ")";
  }

  sourceExpressionsMap[operand] = expression.str().str();
  // Return the constructed source expression
  return expression.str().str();
}

/**
 * Get the source-level expression for a load instruction.
 *
 * @param loadInst The load instruction.
 * @return The source-level expression for the value operand.
 */

std::string
LoadStoreSourceExpression::getSourceExpressionForLoadInst(LoadInst *loadInst) {
  // Load instruction - return the source expression for its value operand

  Value *operandVal = loadInst->getPointerOperand();

  // Check if a source expression exists for the value operand
  std::string operandName = sourceExpressionsMap.count(operandVal)
                                ? sourceExpressionsMap[operandVal]
                                : getSourceExpression(operandVal);

  sourceExpressionsMap[operandVal] = operandName;

  // Return the source expression for the value operand
  return operandName;
}

/**
 * Get the source-level expression for a store instruction.
 *
 * @param storeInst The store instruction.
 * @return The source-level expression for the value operand.
 */

std::string LoadStoreSourceExpression::getSourceExpressionForStoreInst(
    StoreInst *storeInst) {
  // Store instruction - return the source expression for its value operand

  Value *operandVal = storeInst->getValueOperand();

  // Check if a source expression exists for the value operand
  std::string operandName = sourceExpressionsMap.count(operandVal)
                                ? sourceExpressionsMap[operandVal]
                                : getSourceExpression(operandVal);

  sourceExpressionsMap[storeInst->getPointerOperand()] = operandName;
  // Return the source expression for the value operand
  return operandName;
}

/**
 * Get the source-level expression for a sign extension instruction.
 *
 * @param sextInst The sign extension instruction.
 * @return The source-level expression for the operand.
 */

std::string
LoadStoreSourceExpression::getSourceExpressionForSExtInst(SExtInst *sextInst) {
  // Signed Extension instruction - return the source expression for its operand

  Value *operandVal = sextInst->getOperand(0);

  // Check if a source expression exists for the operand
  std::string operandName = sourceExpressionsMap.count(operandVal)
                                ? sourceExpressionsMap[operandVal]
                                : getSourceExpression(operandVal);

  sourceExpressionsMap[operandVal] = operandName;

  // Return the source expression for the operand
  return operandName;
}

// Process the StoreInst and generate the source expression for the stored
// value.
std::string LoadStoreSourceExpression::processStoreInst(StoreInst *I,
                                                        StringRef symbol,
                                                        bool loadFlag) {
  Value *storedValue = I->getPointerOperand();

  // Process associated metadata with the stored value to get the information
  // about variable name
  DILocalVariable *localVar = processDbgMetadata(storedValue);

  Value *operand = nullptr;
  if (isa<Instruction>(I->getValueOperand())) {
    // Check if the value operand is an instruction
    operand = I->getValueOperand();
  } else if (isa<Instruction>(I->getPointerOperand())) {
    // Check if the pointer operand is an instruction
    operand = I->getPointerOperand();
  }

  if (operand) {
    // Generate the source expression for the operand

    std::string expression;
    if (!sourceExpressionsMap.count(operand) ||
        isa<GetElementPtrInst>(operand)) {
      expression = getSourceExpression(operand, symbol);

      if (isa<GetElementPtrInst>(operand)) {
        sourceExpressionsMap[I->getPointerOperand()] = expression;
      }
    } else {
      expression = sourceExpressionsMap[operand];
    }

    return expression;
  }
  if (localVar) {
    // Return the name of the local variable
    return localVar->getName().str();
  }

  return {};
}

// Process the LoadInst and generate the source expressions for the loaded value
// and its corresponding store instruction (if applicable).
void LoadStoreSourceExpression::processLoadInst(LoadInst *I,

                                                StringRef symbol) {
  SmallVector<std::string> sourceExpressions;

  // Search for the corresponding StoreInst for the LoadInst and process it
  for (User *U : I->getPointerOperand()->users()) {
    if (StoreInst *storeInst = dyn_cast<StoreInst>(U)) {
      // Map the StoreInst to the current LoadInst in the loadStoreMap
      loadStoreMap[storeInst] = I;

      // Process the StoreInst and generate the source expression
      std::string expression = processStoreInst(storeInst, symbol, true);

      // Map the LoadInst to its source expression in the sourceExpressionsMap
      sourceExpressionsMap[I] = removeAmpersand(expression);

      break; // Assuming there is only one store instruction for the load
    }
  }

  // Check if the pointer operand of the LoadInst is an instruction
  if (isa<Instruction>(I->getPointerOperand())) {

    Value *val = I->getPointerOperand();

    // Get the source expression for the pointer operand
    std::string expression;
    if (!sourceExpressionsMap.count(val)) {
      expression = getSourceExpression(val, symbol);
    } else {
      expression = sourceExpressionsMap[val];
    }

    // Map the LoadInst to its source expression in the sourceExpressionsMap
    sourceExpressionsMap[I] = removeAmpersand(expression);
  }
}

// Build the source level expression for the given LLVM instruction
void LoadStoreSourceExpression::buildSourceLevelExpression(Instruction &I,
                                                           StringRef symbol) {
  SmallVector<std::string> sourceExpressions;

  // Check if the instruction is a LoadInst
  if (auto *loadInst = dyn_cast<LoadInst>(&I)) {
    // Process the LoadInst and generate the source expressions
    processLoadInst(loadInst, symbol);
  }
  // If it is a StoreInst
  else if (auto *storeInst = dyn_cast<StoreInst>(&I)) {

    // Check if the StoreInst has not been processed already
    if (loadStoreMap.count(storeInst) == 0) {
      // Process the StoreInst and generate the source expressions
      std::string expression = processStoreInst(storeInst, symbol);
      sourceExpressionsMap[storeInst->getPointerOperand()] = expression;
    }
  }
}

AnalysisKey SourceExpressionAnalysis::Key;

SourceExpressionAnalysis::Result
SourceExpressionAnalysis::run(Function &F, FunctionAnalysisManager &) {

  return LoadStoreSourceExpression(F);
}

void LoadStoreSourceExpression::print(raw_ostream &OS) const {

  for (const auto &entry : sourceExpressionsMap) {
    Value *key = entry.first;
    std::string value = entry.second;

    if (Instruction *keyInst = dyn_cast<Instruction>(key)) {
      keyInst->printAsOperand(dbgs(), /*PrintType=*/false);
    } else {
      OS << "<unknown>";
    }
    OS << " = " << value;

    OS << "\n";
  }
}

PreservedAnalyses
SourceExpressionAnalysisPrinterPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  OS << "Load Store Expression " << F.getName() << "\n";
  SourceExpressionAnalysis::Result &PI = AM.getResult<SourceExpressionAnalysis>(
      F); // Retrieve the correct analysis result type

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      std::string symbol = PI.getExpressionFromOpcode(I.getOpcode());
      PI.buildSourceLevelExpression(I, symbol);
    }
  }

  PI.print(OS);
  return PreservedAnalyses::all();
}
