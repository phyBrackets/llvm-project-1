//===- LoadStoreAnalysis.cpp - Mapping Source Expression
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

#include "llvm/Analysis/LoadStoreAnalysis.h"

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
LoadStoreSourceExpression::getExpressionFromOpcode(StringRef opcode) {
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

// Function to remove the '&' character from a string

std::string
LoadStoreSourceExpression::removeAmpersand(llvm::StringRef AddrStr) {
  std::string result = AddrStr.str();

  size_t found = result.find('&');
  if (found != std::string::npos) {
    result.erase(found, 1);
  }
  return result;
}

/**
 * Process the given DIType and perform type-specific actions based on its kind.
 * This function recursively traverses the type hierarchy to handle basic types,
 * derived types, and composite types (such as structures and arrays). For basic
 * types, information about the member is stored in the memberInfo map. For
 * derived types, the processDIType function is called recursively with the base
 * type. For structures, each element is processed recursively using the
 * processDIType function. For arrays, the base type is processed recursively
 * with the same member name.
 *
 * @param dinode The DIType to process.
 * @param basePointer The base pointer value associated with the DIType.
 * @param memberName The name of the member, if applicable.
 * @return An optional string indicating the type of the DIType, or an empty
 * optional if the type is not recognized.
 */

std::optional<std::string> LoadStoreSourceExpression::processDIType(
    llvm::DIType *diType, llvm::Value *basePointer, std::string memberName) {
  // Check if the DIType is valid
  if (!diType) {
    return {};
  }

  // Process basic types
  if (auto *basicType = llvm::dyn_cast<llvm::DIBasicType>(diType)) {
    // Store information about the member
    memberInfo[basePointer->getName()].push_back(
        {basicType->getName().str(), memberName});
  }
  // Process derived types
  else if (auto *derivedType = llvm::dyn_cast<llvm::DIDerivedType>(diType)) {
    std::string derivedMemberName = derivedType->getName().str();
    auto *type = derivedType->getBaseType();
    processDIType(type, basePointer, derivedMemberName);
  }
  // Process composite types (structures and arrays)
  else if (auto *compositeType =
               llvm::dyn_cast<llvm::DICompositeType>(diType)) {
    // Check if the composite type is a structure
    if (compositeType->getTag() == llvm::dwarf::DW_TAG_structure_type) {
      auto nodeArray = compositeType->getElements();
      if (nodeArray) {
        // Iterate over the elements of the structure
        for (unsigned i = 0; i < nodeArray.size(); ++i) {
          llvm::Metadata *metadata = nodeArray[i];
          llvm::DIType *nestedDINode = llvm::dyn_cast<llvm::DIType>(metadata);
          processDIType(nestedDINode, basePointer);
        }
      }
      return "Struct";
    }
    // Check if the composite type is an array
    else if (compositeType->getTag() == llvm::dwarf::DW_TAG_array_type) {
      auto *baseType = compositeType->getBaseType();
      processDIType(baseType, basePointer, memberName);
      // return "Array";
    }
  }

  // Return the result of processing the DIType with nullptr, basePointer, and
  // memberName
  return processDIType(nullptr, basePointer, memberName);
}

/**
 * Process the base pointer to determine its type and handle different
 * scenarios. If the base pointer is associated with debug metadata
 * (DbgDeclareInst or DbgValueInst), retrieve the corresponding DILocalVariable
 * and extract its type. Then, the type is processed further using the
 * processDIType function to obtain additional information.
 *
 * @param basePointer The base pointer value to process.
 * @return An optional string representing the type of the base pointer, or an
 * empty optional if the type is unknown.
 */

std::optional<std::string>
LoadStoreSourceExpression::processBasePointer(llvm::Value *basePointer) {
  llvm::SmallVector<llvm::DbgValueInst *, 8> DbgValues;
  llvm::DILocalVariable *localVar = nullptr;
  llvm::DIType *type = nullptr;

  if (basePointer->isUsedByMetadata()) {
    // Find the corresponding DbgValues and DbgDeclareInsts
    findDbgValues(DbgValues, basePointer);
    TinyPtrVector<DbgDeclareInst *> dbgDeclareInsts =
        FindDbgDeclareUses(basePointer);

    if (!dbgDeclareInsts.empty()) {
      // Handle the case where DbgDeclareInst is found
      DbgDeclareInst *dbgDeclareInst = dbgDeclareInsts[0];
      localVar = dbgDeclareInst->getVariable();
      sourceExpressionsMap[basePointer] = localVar->getName().str();
    } else if (!DbgValues.empty()) {
      // Handle the case where DbgValueInst is found
      DbgValueInst *dbgValueInst = DbgValues[0];
      localVar = dbgValueInst->getVariable();
      sourceExpressionsMap[basePointer] = localVar->getName().str();
    }
  }

  if (localVar) {
    type = localVar->getType();
  }

  if (type) {
    std::optional<std::string> result = processDIType(type, basePointer);
    if (result.has_value()) {
      std::string type = result.value();
      return type;
      // Process the obtained type as needed
    }
  }

  return std::nullopt;
}

// This function generates the source-level expression for a given operand
// by recursively traversing the operand's instructions. If the operand is a
// GetElementPtr instruction, it constructs the expression for the address
// computation using the names of the base pointer and offset. If the operand is
// a binary operator, it constructs the expression using the names of its
// operands and the operator symbol. If the operand is a LoadInst or StoreInst,
// it returns the source expression for its value operand. If the operand is a
// SExtInst, it returns the source expression for its operand. If the operand
// doesn't match any specific case, it returns its name or operand
// representation.

std::string LoadStoreSourceExpression::getSourceExpression(
    llvm::Value *operand,
    llvm::DenseMap<llvm::Value *, std::string> &sourceExpressionsMap,
    StringRef symbol) {

  if (llvm::GetElementPtrInst *gepInst =
          llvm::dyn_cast<llvm::GetElementPtrInst>(operand)) {
    // GetElementPtr instruction - construct source expression for address
    // computation
    llvm::Value *basePointer = gepInst->getOperand(0);
    llvm::Value *offset = gepInst->getOperand(gepInst->getNumIndices());

    int offsetVal = 0;
    if (llvm::ConstantInt *offsetConstant =
            llvm::dyn_cast<llvm::ConstantInt>(offset)) {
      // Retrieve the value of the constant integer as an integer
      offsetVal = offsetConstant->getSExtValue();
    }

    // Check if the base pointer is an AllocaInst and if its allocated type is
    // an ArrayType
    if (llvm::AllocaInst *allocaInst =
            llvm::dyn_cast<llvm::AllocaInst>(basePointer)) {
      llvm::Type *allocaType = allocaInst->getAllocatedType();
      if (llvm::ArrayType *arrType =
              llvm::dyn_cast<llvm::ArrayType>(allocaType)) {
        checkArrayType[basePointer->getNameOrAsOperand()] = true;
      }
    }
    // Check if the base pointer is of PointerType
    else if (isa<llvm::PointerType>(basePointer->getType())) {
      checkArrayType[basePointer->getNameOrAsOperand()] = true;
    }

    std::string basePointerName =
        sourceExpressionsMap.count(basePointer)
            ? sourceExpressionsMap[basePointer]
            : getSourceExpression(
                  basePointer, sourceExpressionsMap,
                  getExpressionFromOpcode(
                      llvm::Instruction::getOpcodeName(gepInst->getOpcode())));
    std::string offsetName =
        sourceExpressionsMap.count(offset)
            ? sourceExpressionsMap[offset]
            : getSourceExpression(
                  offset, sourceExpressionsMap,
                  getExpressionFromOpcode(
                      llvm::Instruction::getOpcodeName(gepInst->getOpcode())));

    llvm::SmallString<32> expression;
    llvm::raw_svector_ostream OS(expression);
    if (basePointer->getType()) {
      // Process the base pointer to determine its type
      std::optional<std::string> result = processBasePointer(basePointer);
      if (result && result.value() == "Struct") {
        // If the base pointer type is a structure, construct the source
        // expression using member info
        OS << basePointerName << "."
           << memberInfo[basePointer->getName()].at(offsetVal).second;
      } else {
        if (basePointerName.find('[') == std::string::npos &&
            checkArrayType[basePointer->getNameOrAsOperand()]) {
          // Construct the source expression for the address computation with
          // square brackets
          OS << "&" << basePointerName << "[" << offsetName << "]";
        } else if (basePointerName.find('[') != std::string::npos &&
                   basePointerName.find('&') != std::string::npos &&
                   checkArrayType[basePointer->getNameOrAsOperand()]) {
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
    }

    sourceExpressionsMap[gepInst] = expression.str().str();

    return expression.str().str();
  } else if (llvm::BinaryOperator *binaryOp =
                 llvm::dyn_cast<llvm::BinaryOperator>(operand)) {
    // Binary operator - build source expression using two operands
    llvm::Value *operand1 = binaryOp->getOperand(0);
    llvm::Value *operand2 = binaryOp->getOperand(1);

    std::string name1 =
        sourceExpressionsMap.count(operand1)
            ? sourceExpressionsMap[operand1]
            : getSourceExpression(operand1, sourceExpressionsMap, symbol);
    std::string name2 =
        sourceExpressionsMap.count(operand2)
            ? sourceExpressionsMap[operand2]
            : getSourceExpression(operand2, sourceExpressionsMap, symbol);
    std::string opcode = binaryOp->getOpcodeName();
    // std::string expression;
    llvm::SmallString<32> expression;
    llvm::raw_svector_ostream OS(expression);
    if (opcode == "add") {
      if (llvm::ConstantInt *constantInt =
              llvm::dyn_cast<llvm::ConstantInt>(operand2)) {
        if (constantInt->isNegative()) {
          // Modify the expression for addition with a negative value
          llvm::APInt absValue = constantInt->getValue().abs();
          llvm::SmallVector<char, 16> str;
          absValue.toString(str, 10, false);
          std::string absValueStr(str.begin(), str.end());
          OS << "(" << name1 << " - " << absValueStr << ")";
        }
      }
    }

    if (expression.empty()) {
      // If no modification is needed, use the original expression generation
      OS << "(" << name1 << " " << getExpressionFromOpcode(opcode) << " "
         << name2 << ")";
    }

    sourceExpressionsMap[operand] = expression.str().str();

    return expression.str().str();
  } else if (llvm::LoadInst *loadInst =
                 llvm::dyn_cast<llvm::LoadInst>(operand)) {
    // Load instruction - return the source expression for its value operand
    llvm::Value *operandVal = loadInst->getPointerOperand();

    // Check if a source expression exists for the value operand
    std::string operandName =
        sourceExpressionsMap.count(operandVal)
            ? sourceExpressionsMap[operandVal]
            : getSourceExpression(operandVal, sourceExpressionsMap, symbol);
    // Construct the source expression using the operand name and the symbol
    // std::string expression = operandName.str();
    sourceExpressionsMap[operandVal] = operandName;

    return operandName;
  } else if (llvm::StoreInst *storeInst =
                 llvm::dyn_cast<llvm::StoreInst>(operand)) {
    // Store instruction - return the source expression for its value operand
    llvm::Value *operandVal = storeInst->getValueOperand();

    // Check if a source expression exists for the value operand
    std::string operandName =
        sourceExpressionsMap.count(operandVal)
            ? sourceExpressionsMap[operandVal]
            : getSourceExpression(operandVal, sourceExpressionsMap, symbol);
    // Construct the source expression using the operand name and the symbol
    // std::string expression = operandName;
    sourceExpressionsMap[storeInst->getPointerOperand()] = operandName;

    return operandName;
  } else if (llvm::SExtInst *sextInst =
                 llvm::dyn_cast<llvm::SExtInst>(operand)) {
    // Signed Extension instruction - return the source expression for its
    // operand
    llvm::Value *operandVal = sextInst->getOperand(0);

    // Check if a source expression exists for the operand
    std::string operandName =
        sourceExpressionsMap.count(operandVal)
            ? sourceExpressionsMap[operandVal]
            : getSourceExpression(operandVal, sourceExpressionsMap, symbol);

    sourceExpressionsMap[operandVal] = operandName;
    return operandName;
  }

  // If no specific case matches, return the name of the operand or its
  // representation
  return operand->getNameOrAsOperand();
}

// Process the StoreInst and generate the source expression for the stored
// value.
std::string LoadStoreSourceExpression::processStoreInst(
    llvm::StoreInst *I,
    llvm::DenseMap<llvm::Value *, std::string> &sourceExpressionsMap,
    StringRef symbol, bool loadFlag) {
  llvm::Value *storedValue = I->getPointerOperand();
  llvm::SmallVector<llvm::DbgValueInst *, 8> DbgValues;
  std::string sourceExpression;
  llvm::DILocalVariable *localVar;

  // Check if the stored value is used by metadata (debug information)

  if (storedValue->isUsedByMetadata()) {
    // Find the corresponding DbgValues and DbgDeclareInsts
    findDbgValues(DbgValues, storedValue);
    TinyPtrVector<DbgDeclareInst *> dbgDeclareInsts =
        FindDbgDeclareUses(storedValue);

    if (!dbgDeclareInsts.empty()) {
      // Handle the case where DbgDeclareInst is found
      DbgDeclareInst *dbgDeclareInst = dbgDeclareInsts[0];
      localVar = dbgDeclareInst->getVariable();
      sourceExpressionsMap[storedValue] = localVar->getName().str();
    } else if (!DbgValues.empty()) {
      // Handle the case where DbgValueInst is found
      DbgValueInst *dbgValueInst = DbgValues[0];
      localVar = dbgValueInst->getVariable();
      sourceExpressionsMap[storedValue] = localVar->getName().str();
    }
  }

  llvm::Value *operand = nullptr;
  if (llvm::Instruction *operandInst =
          llvm::dyn_cast<llvm::Instruction>(I->getValueOperand())) {
    // Check if the value operand is an instruction
    operand = I->getValueOperand();
  } else if (llvm::Instruction *operandInst =
                 llvm::dyn_cast<llvm::Instruction>(I->getPointerOperand())) {
    // Check if the pointer operand is an instruction
    operand = I->getPointerOperand();
  }

  if (operand) {
    // Generate the source expression for the operand

    std::string expression;
    if (!sourceExpressionsMap.count(operand)) {
      expression = getSourceExpression(operand, sourceExpressionsMap, symbol);
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
void LoadStoreSourceExpression::processLoadInst(
    llvm::LoadInst *I,
    llvm::DenseMap<llvm::Value *, std::string> &sourceExpressionsMap,
    StringRef symbol) {
  llvm::SmallVector<std::string> sourceExpressions;

  // Search for the corresponding StoreInst for the LoadInst and process it
  for (llvm::User *U : I->getPointerOperand()->users()) {
    if (llvm::StoreInst *storeInst = llvm::dyn_cast<llvm::StoreInst>(U)) {
      // Map the StoreInst to the current LoadInst in the loadStoreMap
      loadStoreMap[storeInst] = I;

      // Process the StoreInst and generate the source expression
      std::string expression =
          processStoreInst(storeInst, sourceExpressionsMap, symbol, true);

      // expression = removeAmpersand(expression);

      // Map the LoadInst to its source expression in the sourceExpressionsMap
      sourceExpressionsMap[I] = removeAmpersand(expression);

      break; // Assuming there is only one store instruction for the load
    }
  }

  // Check if the pointer operand of the LoadInst is an instruction
  if (llvm::Instruction *operandInst =
          llvm::dyn_cast<llvm::Instruction>(I->getPointerOperand())) {

    llvm::Value *val = I->getPointerOperand();

    // Get the source expression for the pointer operand
    std::string expression;
    if (!sourceExpressionsMap.count(val)) {
      expression = getSourceExpression(val, sourceExpressionsMap, symbol);
    } else {
      expression = sourceExpressionsMap[val];
    }

    // Check if the expression contains '&'
    // expression = removeAmpersand(expression);

    // Map the LoadInst to its source expression in the sourceExpressionsMap
    sourceExpressionsMap[I] = removeAmpersand(expression);
  }
}

// Build the source level expression for the given LLVM instruction
void LoadStoreSourceExpression::buildSourceLevelExpression(llvm::Instruction &I,
                                                           StringRef symbol) {
  llvm::SmallVector<std::string> sourceExpressions;

  // Check if the instruction is a LoadInst
  if (auto *loadInst = llvm::dyn_cast<LoadInst>(&I)) {
    // Process the LoadInst and generate the source expressions
    processLoadInst(loadInst, sourceExpressionsMap, symbol);
  }
  // If it is a StoreInst
  else if (auto *storeInst = llvm::dyn_cast<StoreInst>(&I)) {

    // Check if the StoreInst has not been processed already
    if (loadStoreMap.count(storeInst) == 0) {
      // Process the StoreInst and generate the source expressions
      std::string expression =
          processStoreInst(storeInst, sourceExpressionsMap, symbol);
      sourceExpressionsMap[storeInst->getPointerOperand()] = expression;
    }
  }
}

AnalysisKey LoadStoreAnalysis::Key;

LoadStoreAnalysis::Result LoadStoreAnalysis::run(Function &F,
                                                 FunctionAnalysisManager &) {

  return LoadStoreSourceExpression(F);
}

void LoadStoreSourceExpression::print(raw_ostream &OS) const {

  for (const auto &entry : sourceExpressionsMap) {
    llvm::Value *key = entry.first;
    std::string value = entry.second;

    // Print the key
    OS << "Key: ";
    if (llvm::Instruction *keyInst = llvm::dyn_cast<llvm::Instruction>(key)) {
      keyInst->printAsOperand(dbgs(), /*PrintType=*/false);
    } else {
      OS << "<unknown>";
    }
    OS << " - Values: " << value;

    OS << "\n";
  }
}

PreservedAnalyses
LoadStoreAnalysisPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Load Store Expression " << F.getName() << "\n";
  LoadStoreAnalysis::Result &PI = AM.getResult<LoadStoreAnalysis>(
      F); // Retrieve the correct analysis result type
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      std::string operation = llvm::Instruction::getOpcodeName(I.getOpcode());
      std::string symbol = PI.getExpressionFromOpcode(operation);

      PI.buildSourceLevelExpression(I, symbol);
    }
  }

  PI.print(OS);
  return PreservedAnalyses::all();
}

llvm::PassPluginLibraryInfo getSourceExprPluginInfo() {
  return {

      LLVM_PLUGIN_API_VERSION, "LoadStore", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        // Register LoadStoreAnalysisPrinterPass so that it can be used when
        // specifying pass pipelines with `-passes=`.
        PB.registerPipelineParsingCallback(
            [&](StringRef Name, FunctionPassManager &FPM,
                ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "source-expr") {
                FPM.addPass(LoadStoreAnalysisPrinterPass(llvm::outs()));
                return true;
              }
              return false;
            });

        // REGISTRATION FOR "-O{1|2|3|s}"
        // Register LoadStoreAnalysisPrinterPass as a step of an existing
        // pipeline. The insertion point is specified by using the
        // 'registerVectorizerStartEPCallback' callback. To be more precise,
        // using this callback means that LoadStoreAnalysisPrinterPass will be
        // called whenever the vectoriser is used (i.e. when using
        // '-O{1|2|3|s}'.
        PB.registerVectorizerStartEPCallback(
            [](llvm::FunctionPassManager &PM, llvm::OptimizationLevel Level) {
              PM.addPass(LoadStoreAnalysisPrinterPass(llvm::outs()));
            });

        // Register LoadStoreAnalysis as an analysis pass. This is required so
        // that LoadStoreAnalysisPrinterPass (or any other pass) can request the
        // results of LoadStoreAnalysis.
        PB.registerAnalysisRegistrationCallback(
            [](FunctionAnalysisManager &FAM) {
              FAM.registerPass([&] { return LoadStoreAnalysis(); });
            });
      }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSourceExprPluginInfo();
}